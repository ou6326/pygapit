"""
pyGAPIT — GAPIT-style analysis tools for Python
(Genome Association and Prediction Integrated Tool)

Main entry point: GAPIT()

Uses a GAPIT-style interface while targeting selected GAPIT 3.5 workflows.
Not every R parameter or model is implemented yet.

R:      myGAPIT <- GAPIT(Y=myY, GD=myGD, GM=myGM, model="BLINK", PCA.total=3)
Python: myGAPIT  = GAPIT(Y=myY,  GD=myGD, GM=myGM, model="BLINK", PCA_total=3)

(R dots replaced with underscores in Python, e.g. PCA.total → PCA_total)
"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._typing import (
    Array,
    FloatMatrix,
    FloatVector,
    IntVector,
    LabelVector,
    Matrix,
    StrVector,
    as_float_matrix,
    as_float_vector,
    as_str_vector,
    readonly_copy,
    require_square,
)
from .gs.blup import cblup, gblup, sblup
from .gwas.blink import blink_gwas
from .gwas.farmcpu import farmcpu_gwas
from .gwas.glm import glm_gwas
from .gwas.mlm import cmlm_gwas, mlm_gwas
from .gwas.mlmm import mlmm_gwas
from .io.formats import (
    AlignedData,
    GenotypeData,
    PhenotypeData,
    align_inputs,
    maf_filter,
    read_hapmap,
    read_numeric,
    read_phenotype,
)
from .stats.kinship import vanraden_kinship, zhang_kinship
from .stats.pca import PCAResult, build_covariate_matrix, compute_pca
from .stats.testing import (
    benjamini_hochberg,
    bonferroni_threshold,
    genomic_inflation_factor,
)


@dataclass(frozen=True, slots=True)
class GAPITOutputFiles:
    """Paths written for one trait/model analysis."""

    gwas: Path
    prediction: Path | None
    kinship: Path
    pca: Path
    manhattan: Path | None
    qq: Path | None
    kinship_plot: Path | None
    pca_plot: Path | None

    def paths(self) -> tuple[Path, ...]:
        """Return all files that were successfully written."""
        candidates = (
            self.gwas,
            self.prediction,
            self.kinship,
            self.pca,
            self.manhattan,
            self.qq,
            self.kinship_plot,
            self.pca_plot,
        )
        return tuple(path for path in candidates if path is not None)


@dataclass(frozen=True, slots=True)
class ModelRunResult:
    """Normalized output shared by every top-level analysis model."""

    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    h2: float = 0.0
    vg: float = 0.0
    ve: float = 0.0
    selected_qtns: IntVector | None = None

    def __post_init__(self) -> None:
        marker_count = len(self.p_values)
        if self.p_values.ndim != 1:
            raise ValueError("Model p-values must be one-dimensional")
        if self.effects.shape != (marker_count,) or self.se.shape != (marker_count,):
            raise ValueError(
                "Model p-values, effects, and standard errors must have equal length"
            )
        for field in ("p_values", "effects", "se"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))
        if self.selected_qtns is not None:
            object.__setattr__(self, "selected_qtns", readonly_copy(self.selected_qtns))


@dataclass(frozen=True, slots=True)
class PreparedTrait:
    """Arrays and annotations shared by every model for one trait."""

    name: str
    y: FloatVector
    genotypes: FloatMatrix
    kinship: FloatMatrix
    design: FloatMatrix
    taxa: StrVector
    pca: PCAResult
    snp_names: StrVector
    chromosomes: LabelVector
    positions: FloatVector
    maf: FloatVector

    def __post_init__(self) -> None:
        for field in (
            "y",
            "genotypes",
            "kinship",
            "design",
            "taxa",
            "snp_names",
            "chromosomes",
            "positions",
            "maf",
        ):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))

    @property
    def n_obs(self) -> int:
        return len(self.y)

    @property
    def marker_count(self) -> int:
        return self.genotypes.shape[1]


_SUPPORTED_MODELS = frozenset(
    {"GLM", "MLM", "CMLM", "MLMM", "BLINK", "FARMCPU", "GBLUP", "CBLUP"}
)


@dataclass(slots=True)
class GAPITResult:
    """
    Return object from GAPIT().
    Mirrors the named list returned by R's GAPIT().
    """

    # GWAS outputs
    GWAS: pd.DataFrame | None = None  # full results table
    significant: pd.DataFrame | None = None  # significant SNPs only
    lambda_gc: float = 1.0  # genomic inflation factor

    # GS/prediction outputs
    Pred: pd.DataFrame | None = None  # prediction table per individual

    # Variance components
    h2: float = 0.0
    vg: float = 0.0
    ve: float = 0.0

    # QTNs identified (multi-locus methods)
    QTNs: IntVector | None = None

    # Intermediate objects (useful for custom downstream analysis)
    kinship: FloatMatrix | None = None
    pca: PCAResult | None = None
    taxa: StrVector | None = None
    output_files: GAPITOutputFiles | None = None
    multiple_output_files: tuple[Path, ...] = ()

    # Method used
    model: str = ""
    trait: str = ""
    runtime_seconds: float = 0.0


def GAPIT(
    # ── Input data ──────────────────────────────────────────────────────
    Y: pd.DataFrame | str | Path | None = None,  # phenotype
    G: pd.DataFrame | str | Path | None = None,  # HapMap genotype
    GD: pd.DataFrame | Matrix | str | Path | None = None,  # numeric genotype
    GM: pd.DataFrame | str | Path | None = None,  # SNP map
    KI: Matrix | pd.DataFrame | None = None,  # kinship
    CV: pd.DataFrame | Array | None = None,  # covariates
    Z: Matrix | pd.DataFrame | None = None,  # incidence matrix
    # ── Model selection ─────────────────────────────────────────────────
    model: str | list[str] = "BLINK",
    # ── PCA parameters ──────────────────────────────────────────────────
    PCA_total: int = 3,  # R: PCA.total
    # ── QC parameters ───────────────────────────────────────────────────
    maf_threshold: float = 0.05,  # R: maf.threshold
    SNP_impute: str = "middle",  # R: SNP.impute
    # ── GWAS parameters ─────────────────────────────────────────────────
    cutOff: float | None = None,  # significance threshold
    p_threshold: float | None = None,  # R: p.threshold (multi-locus)
    FDRcut: bool = False,  # R: filter BLINK pseudo-QTNs by an FDR cutoff
    LD: float = 0.7,  # LD threshold for BLINK
    # ── CMLM parameters ─────────────────────────────────────────────────
    group_from: int = 1,  # R: group.from
    group_to: int | None = None,  # R: group.to
    # ── FarmCPU/BLINK parameters ─────────────────────────────────────────
    bin_size: int = 5_000_000,  # R: bin.size (bp)
    maxLoop: int = 10,  # max iterations
    # ── Genomic Selection ────────────────────────────────────────────────
    buspred: bool = False,  # predict after GWAS
    prediction_model: str | None = None,
    # ── Simulation parameters ────────────────────────────────────────────
    h2: float | None = None,  # heritability for simulation
    NQTN: int | None = None,  # number of QTNs for simulation
    # ── Output parameters ────────────────────────────────────────────────
    file_output: bool = True,  # R: file.output
    output_dir: str | Path = ".",  # working directory for outputs
    Multiple_analysis: bool = False,  # R: Multiple_analysis
    # ── Kinship algorithm ────────────────────────────────────────────────
    kinship_algorithm: str = "VanRaden",  # R: kinship.algorithm
    # ── Trait to analyze (column name or index) ──────────────────────────
    trait: str | int | None = None,
) -> GAPITResult | dict[str, GAPITResult]:
    """
    GAPIT — Genome Association and Prediction Integrated Tool (Python)

    GAPIT-style Python pipeline targeting selected GAPIT 3.5 workflows.
    Currently dispatches GLM, MLM, CMLM, MLMM, FarmCPU, BLINK, gBLUP,
    and cBLUP. Top-level sBLUP and SUPER dispatch are not yet available.

    Examples
    --------
    # Basic GWAS with BLINK (default):
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df, model="BLINK")

    # Multiple models:
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df,
                   model=["FarmCPU", "BLINK", "MLM"])

    # Genomic prediction:
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df,
                   model="gBLUP")

    # With user kinship:
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df,
                   KI=my_kinship, model="MLM")

    Returns
    -------
    GAPITResult with GWAS table, significant SNPs, Pred table,
    h2, vg, ve, kinship, pca, QTNs
    """
    normalized_kinship_algorithm, normalized_prediction_model = (
        _validate_compatibility_options(
            Z=Z,
            FDRcut=FDRcut,
            prediction_model=prediction_model,
            Multiple_analysis=Multiple_analysis,
            kinship_algorithm=kinship_algorithm,
        )
    )
    if Z is not None and KI is None:
        raise ValueError("Z requires a corresponding KI random-effect matrix")
    models = _normalize_models(model)
    _validate_analysis_options(
        PCA_total=PCA_total,
        maf_threshold=maf_threshold,
        cutOff=cutOff,
        p_threshold=p_threshold,
        LD=LD,
        group_from=group_from,
        group_to=group_to,
        bin_size=bin_size,
        maxLoop=maxLoop,
        h2=h2,
        NQTN=NQTN,
    )
    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    pheno, geno = _load_data(Y, G, GD, GM, SNP_impute)

    # ── Simulation mode ──────────────────────────────────────────────────
    if h2 is not None and NQTN is not None:
        if NQTN > geno.GD.shape[1]:
            raise ValueError(
                f"NQTN ({NQTN}) cannot exceed the marker count ({geno.GD.shape[1]})"
            )
        print(f"[pyGAPIT] Simulation mode: h^2={h2}, NQTN={NQTN}")
        pheno = _simulate_phenotype(pheno, geno, h2=h2, n_qtn=NQTN)

    # ── Select trait ─────────────────────────────────────────────────────
    traits_to_run = _select_traits(pheno.trait_names, trait)

    # Taxa alignment is independent of the selected trait and model.
    if Z is not None:
        if KI is None:  # narrowed by the pre-load contract above
            raise ValueError("Z requires a corresponding KI random-effect matrix")
        ki_df = _incidence_kinship_to_df(Z, KI, pheno.taxa)
    else:
        ki_df = _ki_to_df(KI, pheno.taxa) if KI is not None else None
    cv_df = _cv_to_df(CV, pheno.taxa) if CV is not None else None
    aligned = align_inputs(pheno, geno, cv_df=cv_df, ki_df=ki_df)

    all_results: dict[str, GAPITResult] = {}

    for trait_name in traits_to_run:
        print(f"\n[pyGAPIT] ---- Trait: {trait_name} ----")
        prepared = _prepare_trait(
            aligned,
            trait_name,
            PCA_total,
            maf_threshold,
            normalized_kinship_algorithm,
        )
        if prepared is None:
            continue

        # ── Run requested models ─────────────────────────────────────────
        for model_name in models:
            print(f"[pyGAPIT] Running {model_name}...")
            t_model = time.time()

            result = _run_model(
                model_name=model_name,
                y=prepared.y,
                X0=prepared.design,
                GD=prepared.genotypes,
                K=prepared.kinship,
                chromosomes=prepared.chromosomes,
                positions=prepared.positions,
                p_threshold=p_threshold,
                group_from=group_from,
                group_to=group_to,
                bin_size=bin_size,
                maxLoop=maxLoop,
                LD_threshold=LD,
                fdr_cut=FDRcut,
                fdr_alpha=cutOff or 0.05,
            )

            elapsed = time.time() - t_model
            print(f"[pyGAPIT] {model_name} done in {elapsed:.1f}s | h^2={result.h2:.3f}")
            all_results[f"{trait_name}_{model_name}"] = _assemble_result(
                prepared=prepared,
                model_result=result,
                model_name=model_name,
                cut_off=cutOff,
                buspred=buspred,
                prediction_model=normalized_prediction_model,
                file_output=file_output,
                output_dir=output_dir,
                started_at=t_start,
            )

    if Multiple_analysis and file_output:
        _attach_multiple_analysis_outputs(all_results, output_dir)

    if not all_results:
        raise ValueError(
            "No analyses completed; each selected trait had fewer than 10 values"
        )

    # Return single result if only one, else dict
    if len(all_results) == 1:
        return next(iter(all_results.values()))
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper functions
# ─────────────────────────────────────────────────────────────────────────────


def _validate_compatibility_options(
    *,
    Z: Matrix | pd.DataFrame | None,
    FDRcut: object,
    prediction_model: object,
    Multiple_analysis: object,
    kinship_algorithm: object,
) -> tuple[str, str | None]:
    """Validate and normalize GAPIT compatibility options."""
    if not isinstance(FDRcut, bool):
        raise TypeError("FDRcut must be a boolean, matching GAPIT 3.5")
    if not isinstance(Multiple_analysis, bool):
        raise TypeError("Multiple_analysis must be a boolean")
    if Z is not None and not isinstance(Z, pd.DataFrame):
        as_float_matrix(Z, name="incidence matrix Z")

    if not isinstance(kinship_algorithm, str):
        raise TypeError("kinship_algorithm must be a string")
    algorithm_key = kinship_algorithm.strip().casefold()
    algorithms = {"vanraden": "VanRaden", "zhang": "Zhang"}
    if algorithm_key not in algorithms:
        raise ValueError("kinship_algorithm must be 'VanRaden' or 'Zhang'")

    normalized_prediction = None
    if prediction_model is not None:
        if not isinstance(prediction_model, str) or not prediction_model.strip():
            raise TypeError("prediction_model must be a non-empty string")
        prediction_key = prediction_model.strip().upper()
        if prediction_key not in {"GBLUP", "CBLUP", "SBLUP"}:
            raise ValueError("prediction_model must be gBLUP, cBLUP, or sBLUP")
        normalized_prediction = prediction_key

    return algorithms[algorithm_key], normalized_prediction


def _normalize_models(model: str | Sequence[object]) -> tuple[str, ...]:
    """Normalize and validate the requested top-level models."""
    requested: list[object] = [model] if isinstance(model, str) else list(model)
    if not requested:
        raise ValueError("model must contain at least one analysis model")
    normalized_names: list[str] = []
    for name in requested:
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Every model name must be a non-empty string")
        normalized_names.append(name.strip().upper())
    normalized = tuple(normalized_names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("model must not contain duplicate analysis models")
    unknown = [name for name in normalized if name not in _SUPPORTED_MODELS]
    if unknown:
        suffix = (
            " Standalone sBLUP is available as pygapit.sblup(...)."
            if "SBLUP" in unknown
            else ""
        )
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Choose from: "
            f"{', '.join(sorted(_SUPPORTED_MODELS))}.{suffix}"
        )
    return normalized


def _validate_analysis_options(
    *,
    PCA_total: int,
    maf_threshold: float,
    cutOff: float | None,
    p_threshold: float | None,
    LD: float,
    group_from: int,
    group_to: int | None,
    bin_size: int,
    maxLoop: int,
    h2: float | None,
    NQTN: int | None,
) -> None:
    """Reject invalid numerical options before loading or analyzing data."""
    if PCA_total < 0:
        raise ValueError("PCA_total must be non-negative")
    if not 0.0 <= maf_threshold <= 0.5:
        raise ValueError("maf_threshold must be between 0 and 0.5")
    for name, value in (("cutOff", cutOff), ("p_threshold", p_threshold)):
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 <= LD <= 1.0:
        raise ValueError("LD must be between 0 and 1")
    if group_from < 1:
        raise ValueError("group_from must be at least 1")
    if group_to is not None and group_to < group_from:
        raise ValueError("group_to must be greater than or equal to group_from")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if maxLoop < 1:
        raise ValueError("maxLoop must be at least 1")
    if (h2 is None) != (NQTN is None):
        raise ValueError("h2 and NQTN must be provided together for simulation")
    if h2 is not None and not 0.0 < h2 <= 1.0:
        raise ValueError("h2 must be greater than 0 and at most 1")
    if NQTN is not None and NQTN < 1:
        raise ValueError("NQTN must be at least 1")


def _select_traits(trait_names: list[str], trait: str | int | None) -> tuple[str, ...]:
    """Resolve a trait selector with explicit bounds and membership checks."""
    if not trait_names:
        raise ValueError("Phenotype data must contain at least one trait")
    if trait is None:
        return tuple(trait_names)
    if isinstance(trait, bool):
        raise TypeError("trait must be a column name or integer index, not bool")
    if isinstance(trait, int):
        if not -len(trait_names) <= trait < len(trait_names):
            raise ValueError(
                f"Trait index {trait} is out of range for {len(trait_names)} traits"
            )
        return (trait_names[trait],)
    if trait not in trait_names:
        raise ValueError(
            f"Unknown trait {trait!r}; choose from: {', '.join(trait_names)}"
        )
    return (trait,)


def _load_data(
    Y: pd.DataFrame | str | Path | None,
    G: pd.DataFrame | str | Path | None,
    GD: pd.DataFrame | Matrix | str | Path | None,
    GM: pd.DataFrame | str | Path | None,
    snp_impute: str,
) -> tuple[PhenotypeData, GenotypeData]:
    """Load and parse all input data."""
    # Phenotype
    if isinstance(Y, (str, Path)):
        pheno = read_phenotype(Y)
    elif isinstance(Y, pd.DataFrame):
        pheno = PhenotypeData.from_frame(Y)
    else:
        raise TypeError("Y must be a file path or pandas DataFrame")

    # Genotype
    if G is not None and (GD is not None or GM is not None):
        raise ValueError("Provide either G or GD with GM, not both input formats")
    if G is not None:
        geno = read_hapmap(G, impute_method=snp_impute)
    else:
        if (GD is None) != (GM is None):
            raise ValueError("GD and GM must be provided together")
        if GD is None or GM is None:
            raise ValueError("Provide either G (HapMap) or both GD and GM")

        if isinstance(GD, (str, Path)):
            geno = read_numeric(GD, GM, impute_method=snp_impute)
        else:
            if isinstance(GM, pd.DataFrame):
                gm_df = GM
            else:
                marker_path = Path(GM)
                if not marker_path.exists():
                    raise FileNotFoundError(f"Marker map file not found: {marker_path}")
                gm_df = pd.read_csv(marker_path, sep="\t")

            if isinstance(GD, pd.DataFrame):
                geno = GenotypeData.from_numeric_frame(GD, gm_df, snp_impute)
            else:
                geno = GenotypeData.from_array(
                    GD,
                    gm_df,
                    pheno.taxa,
                    impute_method=snp_impute,
                )

    return pheno, geno


def _simulate_phenotype(
    pheno: PhenotypeData,
    geno: GenotypeData,
    h2: float = 0.7,
    n_qtn: int = 20,
) -> PhenotypeData:
    """
    Simulate phenotype from genotype with given heritability.
    Translates GAPIT.Phenotype.Simulation.R
    """
    np.random.seed(198521)  # GAPIT's default seed
    n, m = geno.GD.shape

    # Random QTN indices
    qtn_idx = np.random.choice(m, size=min(n_qtn, m), replace=False)
    GD_qtn = geno.GD[:, qtn_idx]

    # Standardize QTN genotypes
    GD_std = (GD_qtn - GD_qtn.mean(axis=0)) / (GD_qtn.std(axis=0) + 1e-8)

    # Random effects
    effects = np.random.normal(0, 1, size=n_qtn)
    g = GD_std @ effects
    g_var = np.var(g)

    if g_var > 0:
        e_var = g_var * (1 - h2) / h2
    else:
        e_var = 1.0

    e = np.random.normal(0, np.sqrt(e_var), size=n)
    y_sim = g + e

    # Build new Y DataFrame using the genotype taxa (n individuals with geno data)
    geno_taxa = np.array([str(t) for t in geno.taxa])
    col_name = "SimTrait"
    Y_new = pd.DataFrame({"Taxa": geno_taxa, col_name: y_sim})

    return PhenotypeData(
        Y=Y_new,
        taxa=geno_taxa,
        trait_names=[col_name],
    )


def _ki_to_df(KI: pd.DataFrame | Matrix, taxa: StrVector) -> pd.DataFrame:
    """Convert kinship numpy array to DataFrame with taxa column."""
    if isinstance(KI, pd.DataFrame):
        return KI
    values = np.asarray(KI, dtype=float)
    expected_shape = (len(taxa), len(taxa))
    if values.ndim != 2 or values.shape != expected_shape:
        raise ValueError(
            f"KI must be a square {expected_shape} matrix matching phenotype taxa; "
            f"got {values.shape}"
        )
    df = pd.DataFrame(values, columns=taxa)
    df.insert(0, "Taxa", taxa)
    return df


def _incidence_kinship_to_df(
    Z: Matrix | pd.DataFrame,
    KI: pd.DataFrame | Matrix,
    taxa: StrVector,
) -> pd.DataFrame:
    """Expand a random-effect kinship matrix to observations via ``Z K Z'``."""
    effect_labels: list[str] | None = None
    if isinstance(Z, pd.DataFrame):
        incidence_frame = Z.copy()
        first_column: object | None = (
            incidence_frame.columns[0] if len(incidence_frame.columns) else None
        )
        if isinstance(first_column, str) and first_column.casefold() in {
            "taxa",
            "taxon",
        }:
            incidence_frame = incidence_frame.set_index(first_column)
        elif isinstance(incidence_frame.index, pd.RangeIndex):
            raise ValueError(
                "DataFrame Z must have a Taxa column or a labeled taxon index"
            )

        incidence_frame.index = incidence_frame.index.map(str)
        phenotype_taxa = [str(taxon) for taxon in taxa]
        if incidence_frame.index.has_duplicates:
            raise ValueError("DataFrame Z taxon labels must be unique")
        if set(incidence_frame.index) != set(phenotype_taxa):
            raise ValueError("DataFrame Z taxa must exactly match phenotype taxa")
        incidence_frame = incidence_frame.rename(columns=str)
        effect_labels = list(incidence_frame.columns)
        if len(set(effect_labels)) != len(effect_labels):
            raise ValueError("DataFrame Z random-effect labels must be unique")
        incidence_frame.columns = effect_labels
        incidence = as_float_matrix(
            incidence_frame.loc[phenotype_taxa], name="incidence matrix Z"
        )
    else:
        incidence = as_float_matrix(Z, name="incidence matrix Z")
    if incidence.shape[0] != len(taxa):
        raise ValueError(
            "incidence matrix Z must have one row per phenotype taxon; "
            f"expected {len(taxa)}, got {incidence.shape[0]}"
        )
    if incidence.shape[1] == 0 or not np.isfinite(incidence).all():
        raise ValueError("incidence matrix Z must be finite and contain columns")
    if np.any(np.sum(np.abs(incidence), axis=1) == 0):
        raise ValueError("every row of incidence matrix Z must map to a random effect")

    if isinstance(KI, pd.DataFrame):
        kinship_frame = KI.copy()
        first_column = (
            kinship_frame.columns[0] if len(kinship_frame.columns) > 0 else None
        )
        if isinstance(first_column, str) and first_column.casefold() in {
            "taxa",
            "taxon",
        }:
            kinship_frame = kinship_frame.set_index(first_column)
        elif isinstance(kinship_frame.index, pd.RangeIndex):
            raise ValueError(
                "DataFrame KI used with Z must have labeled rows and columns"
            )
        kinship_frame.index = kinship_frame.index.map(str)
        kinship_frame = kinship_frame.rename(columns=str)
        if kinship_frame.index.has_duplicates or kinship_frame.columns.has_duplicates:
            raise ValueError("DataFrame KI random-effect labels must be unique")
        if set(kinship_frame.index) != set(kinship_frame.columns):
            raise ValueError("DataFrame KI row and column labels must match")
        ordered_labels = (
            effect_labels if effect_labels is not None else list(kinship_frame.index)
        )
        if set(ordered_labels) != set(kinship_frame.index):
            raise ValueError("Z columns must exactly match KI random-effect labels")
        kinship_frame = kinship_frame.loc[ordered_labels, ordered_labels]
        try:
            random_kinship = as_float_matrix(
                kinship_frame, name="KI random-effect matrix"
            )
        except ValueError as exc:
            raise ValueError(
                "KI random-effect matrix must contain numeric values"
            ) from exc
    else:
        random_kinship = as_float_matrix(KI, name="KI random-effect matrix")

    require_square(
        random_kinship,
        name="KI random-effect matrix",
        size=incidence.shape[1],
    )
    effective_kinship = incidence @ random_kinship @ incidence.T
    result = pd.DataFrame(effective_kinship, columns=taxa)
    result.insert(0, "Taxa", taxa)
    return result


def _cv_to_df(CV: pd.DataFrame | Array, taxa: StrVector) -> pd.DataFrame:
    """Convert CV numpy array to DataFrame with taxa column."""
    if isinstance(CV, pd.DataFrame):
        return CV
    values = np.asarray(CV, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError(f"CV must be one- or two-dimensional; got {values.ndim}D")
    if values.shape[0] != len(taxa):
        raise ValueError(
            f"CV must have one row per phenotype taxon ({len(taxa)}); "
            f"got {values.shape[0]}"
        )
    cv_df = pd.DataFrame(values, columns=[f"CV{i + 1}" for i in range(values.shape[1])])
    cv_df.insert(0, "Taxa", taxa)
    return cv_df


def _prepare_trait(
    aligned: AlignedData,
    trait_name: str,
    pca_total: int,
    maf_threshold: float,
    kinship_algorithm: str,
) -> PreparedTrait | None:
    """Prepare the taxa, markers, kinship, PCA, and design for one trait."""
    y_column = np.asarray(aligned.phenotypes[trait_name], dtype=float)
    valid_mask = ~np.isnan(y_column)
    valid_indices = np.flatnonzero(valid_mask)
    y = y_column[valid_indices]
    if len(y) < 10:
        warnings.warn(
            f"Only {len(y)} individuals with phenotype for {trait_name}. Skipping."
        )
        return None

    genotypes = aligned.genotypes[valid_indices, :]
    taxa = aligned.taxa[valid_indices]
    print(f"[pyGAPIT] n={len(y)} individuals, m={genotypes.shape[1]} SNPs")

    filtered_genotypes, kept_marker_indices = maf_filter(
        genotypes, threshold=maf_threshold
    )
    marker_map = aligned.markers.iloc[kept_marker_indices].reset_index(drop=True)
    marker_count = filtered_genotypes.shape[1]
    print(f"[pyGAPIT] After MAF filter (≥{maf_threshold}): {marker_count} SNPs")
    if marker_count == 0:
        raise ValueError(
            f"No SNPs remain for trait {trait_name!r} after MAF filtering "
            f"at threshold {maf_threshold}"
        )

    if aligned.kinship is not None:
        kinship = aligned.kinship[np.ix_(valid_indices, valid_indices)]
        print("[pyGAPIT] Using provided kinship matrix")
    else:
        print(f"[pyGAPIT] Computing {kinship_algorithm} kinship...")
        kinship_function = (
            vanraden_kinship if kinship_algorithm == "VanRaden" else zhang_kinship
        )
        kinship = kinship_function(filtered_genotypes)

    print(f"[pyGAPIT] Computing PCA (k={pca_total})...")
    pca_result = compute_pca(filtered_genotypes, n_components=pca_total)
    extra_covariates = (
        aligned.covariates[valid_indices] if aligned.covariates is not None else None
    )
    design = build_covariate_matrix(pca_result, pca_total, extra_covariates)

    return PreparedTrait(
        name=trait_name,
        y=y,
        genotypes=filtered_genotypes,
        kinship=kinship,
        design=design,
        taxa=taxa,
        pca=pca_result,
        snp_names=np.asarray(marker_map["SNP"], dtype=str),
        chromosomes=np.asarray(marker_map["Chromosome"], dtype=str),
        positions=np.asarray(marker_map["Position"], dtype=np.float64),
        maf=_compute_maf(filtered_genotypes),
    )


def _assemble_result(
    *,
    prepared: PreparedTrait,
    model_result: ModelRunResult,
    model_name: str,
    cut_off: float | None,
    buspred: bool,
    prediction_model: str | None,
    file_output: bool,
    output_dir: str | Path,
    started_at: float,
) -> GAPITResult:
    """Build tables, optional predictions/files, and the public result object."""
    gwas = _build_gwas_table(
        snp_names=prepared.snp_names,
        chromosomes=prepared.chromosomes,
        positions=prepared.positions,
        p_values=model_result.p_values,
        effects=model_result.effects,
        se=model_result.se,
        maf=prepared.maf,
        n_obs=prepared.n_obs,
        adj_pvalues=benjamini_hochberg(model_result.p_values),
    )
    threshold = (
        cut_off if cut_off is not None else bonferroni_threshold(prepared.marker_count)
    )
    lambda_gc = genomic_inflation_factor(model_result.p_values)
    significant_mask = gwas["P.value"] <= threshold
    significant = gwas[significant_mask].copy()
    significant_count = significant_mask.sum()
    print(
        f"[pyGAPIT] lambda={lambda_gc:.3f} | {significant_count} significant SNPs "
        f"(threshold={threshold:.2e})"
    )

    prediction = None
    if buspred or prediction_model is not None or model_name in ("GBLUP", "CBLUP"):
        prediction = _run_gs_and_build_pred(
            y=prepared.y,
            X0=prepared.design,
            GD=prepared.genotypes,
            K=prepared.kinship,
            taxa=prepared.taxa,
            model_name=model_name,
            qtn_indices=model_result.selected_qtns,
            prediction_model=prediction_model,
        )

    output_files = None
    if file_output:
        output_files = _save_outputs(
            gwas_df=gwas,
            pred_df=prediction,
            pca_result=prepared.pca,
            K=prepared.kinship,
            taxa=prepared.taxa,
            trait_name=prepared.name,
            model_name=model_name,
            output_dir=output_dir,
        )

    return GAPITResult(
        GWAS=gwas,
        significant=significant if not significant.empty else None,
        lambda_gc=lambda_gc,
        Pred=prediction,
        h2=model_result.h2,
        vg=model_result.vg,
        ve=model_result.ve,
        QTNs=model_result.selected_qtns,
        kinship=prepared.kinship,
        pca=prepared.pca,
        taxa=prepared.taxa,
        output_files=output_files,
        model=model_name,
        trait=prepared.name,
        runtime_seconds=time.time() - started_at,
    )


def _run_model(
    model_name: str,
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    chromosomes: LabelVector,
    positions: FloatVector,
    p_threshold: float | None,
    group_from: int,
    group_to: int | None,
    bin_size: int,
    maxLoop: int,
    LD_threshold: float,
    fdr_cut: bool,
    fdr_alpha: float,
) -> ModelRunResult:
    """Dispatch to the correct GWAS/GS model."""
    m = GD.shape[1]
    p_thresh = p_threshold or (1.0 / m)

    if model_name == "GLM":
        r = glm_gwas(y, X0, GD)
        return ModelRunResult(r.p_values, r.effects, r.se)

    elif model_name == "MLM":
        r = mlm_gwas(y, X0, GD, K)
        return ModelRunResult(r.p_values, r.effects, r.se, r.h2, r.vg, r.ve)

    elif model_name == "CMLM":
        n = len(y)
        r = cmlm_gwas(
            y, X0, GD, K, group_from=group_from, group_to=min(group_to or n, n)
        )
        return ModelRunResult(r.p_values, r.effects, r.se, r.h2, r.vg, r.ve)

    elif model_name == "MLMM":
        r = mlmm_gwas(y, X0, GD, K)
        return ModelRunResult(
            r.p_values, r.effects, r.se, r.h2, r.vg, r.ve, r.selected_qtns
        )

    elif model_name == "BLINK":
        r = blink_gwas(
            y,
            X0,
            GD,
            max_iterations=maxLoop,
            ld_threshold=LD_threshold,
            p_threshold=None if fdr_cut and p_threshold is None else p_thresh,
            fdr_alpha=fdr_alpha if fdr_cut and p_threshold is None else None,
        )
        return ModelRunResult(
            r.p_values, r.effects, r.se, selected_qtns=r.selected_qtns
        )

    elif model_name == "FARMCPU":
        r = farmcpu_gwas(
            y,
            X0,
            GD,
            chromosomes=chromosomes,
            positions=positions,
            max_iterations=maxLoop,
            bin_size=bin_size,
            p_threshold=p_thresh,
        )
        return ModelRunResult(
            r.p_values, r.effects, r.se, r.h2, r.vg, r.ve, r.selected_qtns
        )

    elif model_name == "GBLUP":
        r = gblup(y, X0, K)
        p_vals = np.ones(GD.shape[1])
        return ModelRunResult(
            p_vals, np.zeros(GD.shape[1]), np.ones(GD.shape[1]), r.h2, r.vg, r.ve
        )

    elif model_name == "CBLUP":
        r = cblup(y, X0, GD)
        p_vals = np.ones(GD.shape[1])
        return ModelRunResult(
            p_vals, np.zeros(GD.shape[1]), np.ones(GD.shape[1]), r.h2, r.vg, r.ve
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Choose from: GLM, MLM, CMLM, MLMM, BLINK, FarmCPU, gBLUP, cBLUP. "
            "Standalone sBLUP is available as pygapit.sblup(...)."
        )


def _compute_maf(GD: FloatMatrix) -> FloatVector:
    """Compute MAF for each SNP."""
    n = GD.shape[0]
    freq = np.nansum(GD, axis=0) / (2.0 * n)
    return np.minimum(freq, 1.0 - freq)


def _build_gwas_table(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: FloatVector,
    p_values: FloatVector,
    effects: FloatVector,
    se: FloatVector,
    maf: FloatVector,
    n_obs: int,
    adj_pvalues: FloatVector,
) -> pd.DataFrame:
    """Build standardized GWAS result DataFrame matching GAPIT's CSV output."""
    return (
        pd.DataFrame(
            {
                "SNP": snp_names,
                "Chr": chromosomes,
                "Pos": positions.astype(int),
                "P.value": p_values,
                "maf": np.round(maf, 4),
                "nobs": n_obs,
                "effect": np.round(effects, 6),
                "se": np.round(se, 6),
                "FDR.Adjusted.P.values": np.round(adj_pvalues, 6),
            }
        )
        .sort_values(["Chr", "Pos"])
        .reset_index(drop=True)
    )


def _run_gs_and_build_pred(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    taxa: StrVector,
    model_name: str,
    qtn_indices: IntVector | None = None,
    prediction_model: str | None = None,
) -> pd.DataFrame | None:
    """Run genomic prediction and build prediction DataFrame."""
    selected_model = prediction_model
    if selected_model is None:
        selected_model = (
            "SBLUP"
            if qtn_indices is not None
            and len(qtn_indices) > 0
            and model_name == "FARMCPU"
            else "GBLUP"
        )
    if selected_model == "SBLUP" and (qtn_indices is None or len(qtn_indices) == 0):
        raise ValueError(
            "prediction_model='sBLUP' requires selected QTNs from the GWAS model"
        )

    try:
        if selected_model == "SBLUP" and qtn_indices is not None:
            gs_result = sblup(y, X0, GD, qtn_indices=qtn_indices, taxa=taxa)
        elif selected_model == "CBLUP":
            gs_result = cblup(y, X0, GD, taxa=taxa)
        else:
            gs_result = gblup(y, X0, K, taxa=taxa)

        return pd.DataFrame(
            {
                "Taxa": taxa,
                "BLUE": np.round(gs_result.blue, 4),
                "BLUP": np.round(gs_result.blup, 4),
                "PEV": np.round(gs_result.pev, 6),
                "gBreedingValue": np.round(gs_result.gebv, 4),
                "Prediction": np.round(gs_result.prediction, 4),
            }
        )
    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        warnings.warn(f"GS prediction failed: {e}")
        return None


def _align_multiple_gwas(
    results: Sequence[GAPITResult],
) -> tuple[pd.DataFrame, list[tuple[str, FloatVector]]]:
    """Align model p-values by marker identity and genomic coordinates."""
    keys = ["SNP", "Chr", "Pos"]
    marker_frames: list[pd.DataFrame] = []
    for result in results:
        if result.GWAS is None:
            continue
        missing = {*keys, "P.value"} - set(result.GWAS.columns)
        if missing:
            raise ValueError(
                f"{result.model} GWAS table is missing columns: {sorted(missing)}"
            )
        frame = result.GWAS.loc[:, [*keys, "P.value"]].copy()
        if frame.duplicated(keys).any():
            raise ValueError(f"{result.model} GWAS table contains duplicate markers")
        frame["SNP"] = frame["SNP"].astype(str)
        frame["Chr"] = frame["Chr"].astype(str)
        frame["Pos"] = pd.to_numeric(frame["Pos"], errors="raise")
        marker_frames.append(frame)

    if not marker_frames:
        raise ValueError("at least one GWAS table is required")

    all_markers = pd.concat(
        [frame.loc[:, keys] for frame in marker_frames], ignore_index=True
    )
    coordinate_counts = all_markers.groupby("SNP", sort=False)[["Chr", "Pos"]].nunique()
    conflicting = coordinate_counts.index[(coordinate_counts > 1).any(axis=1)]
    if len(conflicting):
        raise ValueError(
            "SNP identifiers have conflicting genomic coordinates: "
            + ", ".join(map(str, conflicting[:5]))
        )

    chromosome_order = {
        chromosome: index
        for index, chromosome in enumerate(dict.fromkeys(all_markers["Chr"]))
    }
    reference = all_markers.drop_duplicates(keys).copy()
    reference["_chromosome_order"] = reference["Chr"].map(chromosome_order)
    reference = (
        reference.sort_values(["_chromosome_order", "Pos", "SNP"], kind="stable")
        .drop(columns="_chromosome_order")
        .reset_index(drop=True)
    )

    aligned: list[tuple[str, FloatVector]] = []
    for result, frame in zip(
        (result for result in results if result.GWAS is not None),
        marker_frames,
        strict=True,
    ):
        merged = reference.merge(frame, on=keys, how="left", validate="one_to_one")
        aligned.append((result.model, np.asarray(merged["P.value"], dtype=np.float64)))
    return reference, aligned


def _attach_multiple_analysis_outputs(
    results: dict[str, GAPITResult], output_dir: str | Path
) -> None:
    """Create GAPIT-style cross-model Manhattan and QQ plots per trait."""
    import matplotlib.pyplot as plt

    from .visualization.plots import _axes, _genomic_axis, _savefig

    grouped: dict[str, list[GAPITResult]] = {}
    for result in results.values():
        if result.GWAS is not None and result.model not in {"GBLUP", "CBLUP"}:
            grouped.setdefault(result.trait, []).append(result)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for trait_name, trait_results in grouped.items():
        if not trait_results:
            continue
        component = _safe_filename_component(trait_name)
        manhattan_path = out / f"GAPIT.Multiple.Manhattan.{component}.pdf"
        qq_path = out / f"GAPIT.Multiple.QQ.{component}.pdf"

        marker_reference, aligned_p_values = _align_multiple_gwas(trait_results)
        x_values, chromosome_labels, chromosome_centers = _genomic_axis(
            as_str_vector(marker_reference["Chr"].to_numpy()),
            as_float_vector(marker_reference["Pos"].to_numpy()),
        )
        fig_man, raw_ax_man = plt.subplots(figsize=(12, 5))
        ax_man = _axes(raw_ax_man)
        for model, p_values in aligned_p_values:
            valid = np.isfinite(p_values) & (p_values > 0.0) & (p_values <= 1.0)
            ax_man.scatter(
                x_values[valid],
                -np.log10(np.maximum(p_values[valid], 1e-300)),
                s=12,
                alpha=0.65,
                label=model,
            )
        ax_man.set_xticks(chromosome_centers)
        ax_man.set_xticklabels(chromosome_labels)
        ax_man.set_xlabel("Chromosome")
        ax_man.set_ylabel(r"$-\log_{10}(p)$")
        ax_man.set_title(f"Multiple Manhattan: {trait_name}")
        ax_man.legend()
        fig_man.tight_layout()
        _savefig(fig_man, manhattan_path, bbox_inches="tight")
        plt.close(fig_man)

        fig_qq, raw_ax_qq = plt.subplots(figsize=(6, 6))
        ax_qq = _axes(raw_ax_qq)
        qq_upper = 1.0
        for result in trait_results:
            gwas = result.GWAS
            if gwas is None:
                continue
            observed = np.sort(
                np.clip(np.asarray(gwas["P.value"], dtype=np.float64), 1e-300, 1.0)
            )
            expected = (np.arange(1, len(observed) + 1) - 0.5) / len(observed)
            qq_upper = max(
                qq_upper,
                np.max(-np.log10(expected)),
                np.max(-np.log10(observed)),
            )
            ax_qq.plot(
                -np.log10(expected),
                -np.log10(observed),
                marker="o",
                markersize=3,
                linewidth=1,
                label=result.model,
            )
        ax_qq.plot([0.0, qq_upper], [0.0, qq_upper], linestyle="--", color="grey")
        ax_qq.set_xlabel(r"Expected $-\log_{10}(p)$")
        ax_qq.set_ylabel(r"Observed $-\log_{10}(p)$")
        ax_qq.set_title(f"Multiple QQ: {trait_name}")
        ax_qq.legend()
        fig_qq.tight_layout()
        _savefig(fig_qq, qq_path, bbox_inches="tight")
        plt.close(fig_qq)

        paths = (manhattan_path, qq_path)
        for result in trait_results:
            result.multiple_output_files = paths


def _save_outputs(
    gwas_df: pd.DataFrame,
    pred_df: pd.DataFrame | None,
    pca_result: PCAResult,
    K: FloatMatrix,
    taxa: StrVector,
    trait_name: str,
    model_name: str,
    output_dir: str | Path,
) -> GAPITOutputFiles:
    """Save all result files and plots. Translates GAPIT.ID.R output logic."""
    import matplotlib.pyplot as plt

    from .visualization.plots import (
        kinship_heatmap,
        manhattan_plot,
        pca_plot_2d,
        qq_plot,
    )

    prefix = _output_prefix(model_name, trait_name)
    trait_component = _safe_filename_component(trait_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── CSV outputs ────────────────────────────────────────────────────
    gwas_path = out / f"{prefix}.GWAS.Results.csv"
    gwas_df.to_csv(gwas_path, index=False)
    print(f"[pyGAPIT] Saved: {prefix}.GWAS.Results.csv")

    prediction_path = None
    if pred_df is not None:
        prediction_path = out / f"{prefix}.Prediction.csv"
        pred_df.to_csv(prediction_path, index=False)

    # Kinship
    kinship_path = out / f"GAPIT.{trait_component}.Kinship.csv"
    ki_df = pd.DataFrame(K, columns=taxa)
    ki_df.insert(0, "Taxa", taxa)
    ki_df.to_csv(kinship_path, index=False)

    # PCA scores
    pca_path = out / f"GAPIT.{trait_component}.PCA.csv"
    pca_df = pd.DataFrame(
        pca_result.scores,
        columns=[f"PC{i + 1}" for i in range(pca_result.scores.shape[1])],
    )
    pca_df.insert(0, "Taxa", taxa)
    pca_df.to_csv(pca_path, index=False)

    # ── Plots ──────────────────────────────────────────────────────────
    manhattan_path = out / f"{prefix}.Manhattan.pdf"
    qq_path = out / f"{prefix}.QQ.pdf"
    kinship_plot_path = out / f"GAPIT.{trait_component}.Kinship.pdf"
    pca_plot_path = out / f"GAPIT.{trait_component}.PCA.pdf"
    try:
        # Manhattan
        sig_mask = gwas_df["P.value"] <= bonferroni_threshold(len(gwas_df))
        sig_indices = np.where(np.asarray(sig_mask.to_numpy(), dtype=bool))[0]
        plot_snp_names = as_str_vector(gwas_df["SNP"].to_numpy())
        plot_chromosomes = as_str_vector(gwas_df["Chr"].to_numpy())
        plot_positions = as_float_vector(gwas_df["Pos"].to_numpy())
        plot_p_values = as_float_vector(gwas_df["P.value"].to_numpy())

        fig_man = manhattan_plot(
            snp_names=plot_snp_names,
            chromosomes=plot_chromosomes,
            positions=plot_positions,
            p_values=plot_p_values,
            title=f"Manhattan: {trait_name} ({model_name})",
            highlight_snps=sig_indices if len(sig_indices) > 0 else None,
            save_path=str(manhattan_path),
        )
        plt.close(fig_man)

        # QQ
        fig_qq = qq_plot(
            p_values=plot_p_values,
            title=f"QQ: {trait_name} ({model_name})",
            save_path=str(qq_path),
        )
        plt.close(fig_qq)

        # Kinship heatmap
        fig_k = kinship_heatmap(
            K=K,
            taxa=taxa,
            save_path=str(kinship_plot_path),
        )
        plt.close(fig_k)

        # PCA 2D
        if pca_result.scores.shape[1] >= 2:
            fig_pca = pca_plot_2d(
                scores=pca_result.scores,
                var_explained=pca_result.var_explained,
                title=f"PCA: {trait_name}",
                save_path=str(pca_plot_path),
            )
            plt.close(fig_pca)

    except (ValueError, TypeError, OSError, np.linalg.LinAlgError) as e:
        warnings.warn(f"Plot generation failed: {e}")

    return GAPITOutputFiles(
        gwas=gwas_path,
        prediction=prediction_path,
        kinship=kinship_path,
        pca=pca_path,
        manhattan=manhattan_path if manhattan_path.exists() else None,
        qq=qq_path if qq_path.exists() else None,
        kinship_plot=kinship_plot_path if kinship_plot_path.exists() else None,
        pca_plot=pca_plot_path if pca_plot_path.exists() else None,
    )


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename_component(value: str) -> str:
    """Make a user-controlled model or trait name safe for one filename part."""
    component = _INVALID_FILENAME_CHARS.sub("_", value).strip(" .")
    return component or "unnamed"


def _output_prefix(model_name: str, trait_name: str) -> str:
    """Build the stable filename prefix for one analysis."""
    return (
        f"GAPIT.{_safe_filename_component(model_name)}."
        f"{_safe_filename_component(trait_name)}"
    )
