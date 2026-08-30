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
from .stats.kinship import vanraden_kinship
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

    p_values: np.ndarray
    effects: np.ndarray
    se: np.ndarray
    h2: float = 0.0
    vg: float = 0.0
    ve: float = 0.0
    selected_qtns: np.ndarray | None = None

    def __post_init__(self) -> None:
        marker_count = len(self.p_values)
        if self.p_values.ndim != 1:
            raise ValueError("Model p-values must be one-dimensional")
        if self.effects.shape != (marker_count,) or self.se.shape != (marker_count,):
            raise ValueError(
                "Model p-values, effects, and standard errors must have equal length"
            )


@dataclass(frozen=True, slots=True)
class PreparedTrait:
    """Arrays and annotations shared by every model for one trait."""

    name: str
    y: np.ndarray
    genotypes: np.ndarray
    kinship: np.ndarray
    design: np.ndarray
    taxa: np.ndarray
    pca: PCAResult
    snp_names: np.ndarray
    chromosomes: np.ndarray
    positions: np.ndarray
    maf: np.ndarray

    @property
    def n_obs(self) -> int:
        return len(self.y)

    @property
    def marker_count(self) -> int:
        return int(self.genotypes.shape[1])


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
    QTNs: np.ndarray | None = None

    # Intermediate objects (useful for custom downstream analysis)
    kinship: np.ndarray | None = None
    pca: PCAResult | None = None
    taxa: np.ndarray | None = None
    output_files: GAPITOutputFiles | None = None

    # Method used
    model: str = ""
    trait: str = ""
    runtime_seconds: float = 0.0


def GAPIT(
    # ── Input data ──────────────────────────────────────────────────────
    Y: pd.DataFrame | str | Path | None = None,  # phenotype
    G: pd.DataFrame | str | Path | None = None,  # HapMap genotype
    GD: pd.DataFrame | np.ndarray | str | Path | None = None,  # numeric genotype
    GM: pd.DataFrame | str | Path | None = None,  # SNP map
    KI: np.ndarray | pd.DataFrame | None = None,  # kinship
    CV: pd.DataFrame | np.ndarray | None = None,  # covariates
    Z: np.ndarray | None = None,  # incidence matrix
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
    _validate_compatibility_options(
        Z=Z,
        FDRcut=FDRcut,
        prediction_model=prediction_model,
        Multiple_analysis=Multiple_analysis,
        kinship_algorithm=kinship_algorithm,
    )
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
        print(f"[pyGAPIT] Simulation mode: h²={h2}, NQTN={NQTN}")
        pheno = _simulate_phenotype(pheno, geno, h2=h2, n_qtn=NQTN)

    # ── Select trait ─────────────────────────────────────────────────────
    traits_to_run = _select_traits(pheno.trait_names, trait)

    # Taxa alignment is independent of the selected trait and model.
    ki_df = _ki_to_df(KI, pheno.taxa) if KI is not None else None
    cv_df = _cv_to_df(CV, pheno.taxa) if CV is not None else None
    aligned = align_inputs(pheno, geno, cv_df=cv_df, ki_df=ki_df)

    all_results: dict[str, GAPITResult] = {}

    for trait_name in traits_to_run:
        print(f"\n[pyGAPIT] ──── Trait: {trait_name} ────")
        prepared = _prepare_trait(aligned, trait_name, PCA_total, maf_threshold)
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
            )

            elapsed = time.time() - t_model
            print(f"[pyGAPIT] {model_name} done in {elapsed:.1f}s | h²={result.h2:.3f}")
            all_results[f"{trait_name}_{model_name}"] = _assemble_result(
                prepared=prepared,
                model_result=result,
                model_name=model_name,
                cut_off=cutOff,
                buspred=buspred,
                file_output=file_output,
                output_dir=output_dir,
                started_at=t_start,
            )

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
    Z: np.ndarray | None,
    FDRcut: object,
    prediction_model: str | None,
    Multiple_analysis: bool,
    kinship_algorithm: str,
) -> None:
    """Reject accepted GAPIT-style options that are not implemented yet."""
    unsupported: list[str] = []
    if Z is not None:
        unsupported.append("Z incidence matrices")
    if not isinstance(FDRcut, bool):
        raise TypeError("FDRcut must be a boolean, matching GAPIT 3.5")
    if FDRcut is True:
        unsupported.append("FDR-based BLINK pseudo-QTN filtering")
    if prediction_model is not None:
        unsupported.append("prediction_model overrides")
    if Multiple_analysis:
        unsupported.append("Multiple_analysis plots")
    if kinship_algorithm.casefold() != "vanraden":
        unsupported.append(f"kinship_algorithm={kinship_algorithm!r}")
    if unsupported:
        raise NotImplementedError(
            "Unsupported GAPIT option(s): " + ", ".join(unsupported)
        )


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
    GD: pd.DataFrame | np.ndarray | str | Path | None,
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
        elif isinstance(GD, pd.DataFrame):
            if isinstance(GM, pd.DataFrame):
                gm_df = GM
            else:
                marker_path = Path(GM)
                if not marker_path.exists():
                    raise FileNotFoundError(f"Marker map file not found: {marker_path}")
                gm_df = pd.read_csv(marker_path, sep="\t")
            geno = GenotypeData.from_numeric_frame(GD, gm_df, snp_impute)
        else:
            if isinstance(GM, pd.DataFrame):
                gm_df = GM
            else:
                marker_path = Path(GM)
                if not marker_path.exists():
                    raise FileNotFoundError(f"Marker map file not found: {marker_path}")
                gm_df = pd.read_csv(marker_path, sep="\t")
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


def _ki_to_df(KI: pd.DataFrame | np.ndarray, taxa: np.ndarray) -> pd.DataFrame:
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


def _cv_to_df(CV: pd.DataFrame | np.ndarray, taxa: np.ndarray) -> pd.DataFrame:
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
        print("[pyGAPIT] Computing VanRaden kinship...")
        kinship = vanraden_kinship(filtered_genotypes)

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
        chromosomes=np.asarray(marker_map["Chromosome"]),
        positions=np.asarray(marker_map["Position"], dtype=float),
        maf=_compute_maf(filtered_genotypes),
    )


def _assemble_result(
    *,
    prepared: PreparedTrait,
    model_result: ModelRunResult,
    model_name: str,
    cut_off: float | None,
    buspred: bool,
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
        f"[pyGAPIT] λ={lambda_gc:.3f} | {significant_count} significant SNPs "
        f"(threshold={threshold:.2e})"
    )

    prediction = None
    if buspred or model_name in ("GBLUP", "CBLUP"):
        prediction = _run_gs_and_build_pred(
            y=prepared.y,
            X0=prepared.design,
            GD=prepared.genotypes,
            K=prepared.kinship,
            taxa=prepared.taxa,
            model_name=model_name,
            qtn_indices=model_result.selected_qtns,
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
    y: np.ndarray,
    X0: np.ndarray,
    GD: np.ndarray,
    K: np.ndarray,
    chromosomes: np.ndarray,
    positions: np.ndarray,
    p_threshold: float | None,
    group_from: int,
    group_to: int | None,
    bin_size: int,
    maxLoop: int,
    LD_threshold: float,
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
        r = mlmm_gwas(y, X0, GD, K, p_threshold=p_thresh)
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
            p_threshold=p_thresh,
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


def _compute_maf(GD: np.ndarray) -> np.ndarray:
    """Compute MAF for each SNP."""
    n = GD.shape[0]
    freq = np.nansum(GD, axis=0) / (2.0 * n)
    return np.asarray(np.minimum(freq, 1.0 - freq), dtype=float)


def _build_gwas_table(
    snp_names: np.ndarray,
    chromosomes: np.ndarray,
    positions: np.ndarray,
    p_values: np.ndarray,
    effects: np.ndarray,
    se: np.ndarray,
    maf: np.ndarray,
    n_obs: int,
    adj_pvalues: np.ndarray,
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
    y: np.ndarray,
    X0: np.ndarray,
    GD: np.ndarray,
    K: np.ndarray,
    taxa: np.ndarray,
    model_name: str,
    qtn_indices: np.ndarray | None = None,
) -> pd.DataFrame | None:
    """Run genomic prediction and build prediction DataFrame."""
    try:
        if qtn_indices is not None and len(qtn_indices) > 0 and model_name == "FARMCPU":
            gs_result = sblup(y, X0, GD, qtn_indices=qtn_indices, taxa=taxa)
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


def _save_outputs(
    gwas_df: pd.DataFrame,
    pred_df: pd.DataFrame | None,
    pca_result: PCAResult,
    K: np.ndarray,
    taxa: np.ndarray,
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
    kinship_path = out / "GAPIT.Kinship.csv"
    ki_df = pd.DataFrame(K, columns=taxa)
    ki_df.insert(0, "Taxa", taxa)
    ki_df.to_csv(kinship_path, index=False)

    # PCA scores
    pca_path = out / "GAPIT.PCA.csv"
    pca_df = pd.DataFrame(
        pca_result.scores,
        columns=[f"PC{i + 1}" for i in range(pca_result.scores.shape[1])],
    )
    pca_df.insert(0, "Taxa", taxa)
    pca_df.to_csv(pca_path, index=False)

    # ── Plots ──────────────────────────────────────────────────────────
    manhattan_path = out / f"{prefix}.Manhattan.pdf"
    qq_path = out / f"{prefix}.QQ.pdf"
    kinship_plot_path = out / "GAPIT.Kinship.pdf"
    pca_plot_path = out / "GAPIT.PCA.pdf"
    try:
        # Manhattan
        sig_mask = gwas_df["P.value"] <= bonferroni_threshold(len(gwas_df))
        sig_indices = np.where(np.asarray(sig_mask.to_numpy(), dtype=bool))[0]

        fig_man = manhattan_plot(
            snp_names=np.asarray(gwas_df["SNP"].to_numpy()),
            chromosomes=np.asarray(gwas_df["Chr"].to_numpy()),
            positions=np.asarray(gwas_df["Pos"].to_numpy()),
            p_values=np.asarray(gwas_df["P.value"].to_numpy(), dtype=float),
            title=f"Manhattan: {trait_name} ({model_name})",
            highlight_snps=sig_indices if len(sig_indices) > 0 else None,
            save_path=str(manhattan_path),
        )
        plt.close(fig_man)

        # QQ
        fig_qq = qq_plot(
            p_values=np.asarray(gwas_df["P.value"].to_numpy(), dtype=float),
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
