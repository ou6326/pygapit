"""Shared inputs and R-side design construction for workflow parity tests."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
from numpy.typing import NDArray

from pygapit.gapit import GAPITResult
from pygapit.io.formats import GenotypeData
from pygapit.io.storage import (
    NumpyGenotypeStore,
    open_genotype_store,
    write_numpy_genotype,
)
from pygapit.stats.kinship import vanraden_kinship
from tests.cross_language.r_bridge import RBridge, RList, RMatrix, RObject, RVector

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class WorkflowInputs:
    """Labeled top-level inputs plus their canonical post-filter arrays."""

    phenotype: pd.DataFrame
    genotype: pd.DataFrame
    marker_map: pd.DataFrame
    covariate: pd.DataFrame
    kinship: pd.DataFrame
    taxa: StringArray
    phenotype_values: FloatArray
    genotype_values: FloatArray
    covariate_values: FloatArray
    kinship_values: FloatArray


@dataclass(frozen=True, slots=True)
class PreprocessingReference:
    """GAPIT preprocessing reference for complete labeled workflows."""

    phenotype: pd.DataFrame
    genotype: pd.DataFrame
    marker_map: pd.DataFrame
    filtered_map: pd.DataFrame
    taxa: StringArray
    phenotype_values: FloatArray
    genotype_values: FloatArray
    maf: FloatArray
    pca_scores: FloatArray
    design: FloatArray
    kinship: FloatArray
    output_order: NDArray[np.intp]


@contextmanager
def open_numpy_workflow_store(
    directory: Path,
    genotype: pd.DataFrame,
    marker_map: pd.DataFrame,
) -> Generator[NumpyGenotypeStore]:
    """Open a dependency-free mmap store for direct R workflow comparisons."""
    path = directory / "genotype-store"
    write_numpy_genotype(
        path,
        GenotypeData.from_numeric_frame(genotype, marker_map),
        marker_chunk_size=2,
    )
    with open_genotype_store(path, backend="numpy") as store:
        yield store


def r_scalar(r_bridge: RBridge, result: RObject, name: str):
    """Extract one numeric scalar from a named R list component."""
    values = r_bridge.float_array(r_bridge.component(result, name)).reshape(-1)
    value: np.float64 = values[0]
    return value


def prepare_complete_workflow_reference(
    r_bridge: RBridge,
    r_root: Path,
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> PreprocessingReference:
    """Run GAPIT's imputation, phenotype subset, MAF, PCA, and kinship steps."""
    phenotype, genotype, marker_map = (frame.copy() for frame in fixed_gapit_inputs)
    phenotype.loc[3, "Trait"] = np.nan
    genotype.iloc[0, 1] = np.nan
    genotype.iloc[:, -2] = 2.0
    genotype.iloc[:, -1] = 0.0
    genotype.iloc[0, -1] = 1.0

    raw_genotypes = genotype.iloc[:, 1:].to_numpy(dtype=np.float64)
    r_bridge.source(r_root, "GAPIT.Imputation.R")
    r_impute = r_bridge.function(
        "function(x) apply(x, 2, function(column) "
        "GAPIT.Imputation(column, impute='Middle', byRow=TRUE))",
        returns=RMatrix,
    )
    imputed = r_bridge.float_array(r_impute(r_bridge.matrix(raw_genotypes)))

    valid_samples = phenotype["Trait"].notna().to_numpy()
    analysis_genotypes = imputed[valid_samples]
    analysis_phenotype = phenotype.loc[valid_samples, "Trait"].to_numpy(
        dtype=np.float64
    )
    allele_frequency = np.mean(analysis_genotypes, axis=0) / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    keep = maf >= 0.1
    filtered_genotypes = analysis_genotypes[:, keep]
    filtered_map = marker_map.loc[keep].reset_index(drop=True)
    retained_maf = maf[keep]

    r_pca = r_bridge.source_function(
        r_root,
        "GAPIT.PCA.R",
        "GAPIT.PCA",
        returns=RList,
    )
    r_pca_result = r_pca(
        r_bridge.matrix(filtered_genotypes),
        r_bridge.float_vector(np.arange(len(analysis_phenotype), dtype=np.float64)),
        PC_number=2,
        file_output=False,
        PCA_total=2,
    )
    r_scores_with_taxa = r_bridge.float_array(r_bridge.component(r_pca_result, "PCs"))
    if r_scores_with_taxa.shape[0] != len(analysis_phenotype):
        r_scores_with_taxa = r_scores_with_taxa.T
    r_scores = r_scores_with_taxa[:, 1:3].copy()
    design = np.column_stack([np.ones(len(analysis_phenotype)), r_scores])

    r_kinship = r_bridge.source_function(
        r_root,
        "GAPIT.kinship.VanRaden.R",
        "GAPIT.kinship.VanRaden",
        returns=RMatrix,
    )
    expected_kinship = r_bridge.float_array(
        r_kinship(r_bridge.matrix(filtered_genotypes))
    ).copy()

    chromosomes = filtered_map["Chromosome"].astype(str).to_numpy()
    positions = filtered_map["Position"].to_numpy(dtype=np.float64)
    output_order = np.lexsort((np.arange(len(filtered_map)), positions, chromosomes))
    return PreprocessingReference(
        phenotype=phenotype,
        genotype=genotype,
        marker_map=marker_map,
        filtered_map=filtered_map,
        taxa=phenotype.loc[valid_samples, "Taxa"].to_numpy(dtype=np.str_),
        phenotype_values=analysis_phenotype,
        genotype_values=filtered_genotypes,
        maf=retained_maf,
        pca_scores=r_scores,
        design=design,
        kinship=expected_kinship,
        output_order=output_order,
    )


def assert_complete_workflow_preparation(
    result: GAPITResult,
    reference: PreprocessingReference,
) -> pd.DataFrame:
    """Check preprocessing and return the non-optional final GWAS table."""
    assert result.GWAS is not None
    assert result.pca is not None
    assert result.kinship is not None
    nt.assert_array_equal(result.taxa, reference.taxa)
    nt.assert_allclose(result.kinship, reference.kinship, rtol=1e-12, atol=1e-12)
    for component in range(reference.pca_scores.shape[1]):
        py_scores = result.pca.scores[:, component]
        r_scores = reference.pca_scores[:, component]
        sign = np.sign(np.dot(py_scores, r_scores)) or 1.0
        nt.assert_allclose(py_scores, sign * r_scores, rtol=1e-12, atol=1e-12)

    output_order = reference.output_order
    marker_map = reference.filtered_map
    gwas = result.GWAS
    nt.assert_array_equal(gwas["SNP"], marker_map.loc[output_order, "SNP"])
    nt.assert_array_equal(
        gwas["Chr"].astype(str),
        marker_map["Chromosome"].astype(str).to_numpy()[output_order],
    )
    nt.assert_array_equal(
        gwas["Pos"], marker_map["Position"].to_numpy(dtype=np.float64)[output_order]
    )
    nt.assert_array_equal(
        np.asarray(gwas["maf"], dtype=np.float64),
        np.round(reference.maf[output_order], 4),
    )
    nt.assert_array_equal(gwas["nobs"], len(reference.phenotype_values))
    return gwas


def r_adjusted_p_values(r_bridge: RBridge, p_values: FloatArray) -> FloatArray:
    """Return GAPIT's R-side BH adjustment reference."""
    r_adjust = r_bridge.function("stats::p.adjust", returns=RVector)
    return r_bridge.float_array(
        r_adjust(r_bridge.float_vector(p_values), method="BH")
    ).reshape(-1)


def assert_top_level_preparation(
    result: GAPITResult,
    inputs: WorkflowInputs,
    r_scores: FloatArray,
) -> None:
    """Check taxa, marker, kinship, and PCA preparation at the public boundary."""
    assert result.GWAS is not None
    assert result.pca is not None
    assert result.kinship is not None
    nt.assert_array_equal(result.taxa, inputs.taxa)
    nt.assert_allclose(result.kinship, inputs.kinship_values, rtol=0.0, atol=0.0)
    nt.assert_array_equal(result.GWAS["SNP"], inputs.marker_map["SNP"])
    nt.assert_array_equal(
        result.GWAS["Chr"].astype(str),
        inputs.marker_map["Chromosome"].astype(str),
    )
    nt.assert_array_equal(result.GWAS["Pos"], inputs.marker_map["Position"])
    for component in range(r_scores.shape[1]):
        py_scores = result.pca.scores[:, component]
        sign = np.sign(np.dot(py_scores, r_scores[:, component])) or 1.0
        nt.assert_allclose(
            py_scores, sign * r_scores[:, component], rtol=1e-12, atol=1e-12
        )


def make_workflow_inputs(
    fixed_genotypes: FloatArray,
    fixed_phenotype: FloatArray,
    fixed_covariate: FloatArray,
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> WorkflowInputs:
    """Create shuffled CV/KI inputs with one missing phenotype observation."""
    phenotype, genotype, marker_map = fixed_gapit_inputs
    phenotype = phenotype.copy()
    phenotype.loc[3, "Trait"] = np.nan

    taxa = np.asarray(phenotype["Taxa"].astype(str), dtype=np.str_)
    valid = phenotype["Trait"].notna().to_numpy()
    canonical_kinship = vanraden_kinship(fixed_genotypes)
    reverse = np.arange(len(taxa) - 1, -1, -1)

    covariate = pd.DataFrame({
        "Taxa": taxa[reverse],
        "Covariate": fixed_covariate[reverse],
    })
    kinship = pd.DataFrame(
        canonical_kinship[np.ix_(reverse, reverse)],
        columns=taxa[reverse],
    )
    kinship.insert(0, "Taxa", taxa[reverse])

    return WorkflowInputs(
        phenotype=phenotype,
        genotype=genotype.copy(),
        marker_map=marker_map.copy(),
        covariate=covariate,
        kinship=kinship,
        taxa=taxa[valid],
        phenotype_values=fixed_phenotype[valid],
        genotype_values=fixed_genotypes[valid],
        covariate_values=fixed_covariate[valid, np.newaxis],
        kinship_values=canonical_kinship[np.ix_(valid, valid)],
    )


def r_design_with_pca(
    r_bridge: RBridge,
    r_root: Path,
    inputs: WorkflowInputs,
    pca_total: int,
) -> tuple[FloatArray, FloatArray]:
    """Build GAPIT's R PCA plus CV design for the canonical filtered rows."""
    r_pca = r_bridge.source_function(
        r_root,
        "GAPIT.PCA.R",
        "GAPIT.PCA",
        returns=RList,
    )
    r_result = r_pca(
        r_bridge.matrix(inputs.genotype_values),
        r_bridge.float_vector(np.arange(len(inputs.taxa), dtype=np.float64)),
        PC_number=pca_total,
        file_output=False,
        PCA_total=pca_total,
    )
    scores_with_taxa = r_bridge.float_array(r_bridge.component(r_result, "PCs"))
    if scores_with_taxa.shape[0] != len(inputs.taxa):
        scores_with_taxa = scores_with_taxa.T
    scores = scores_with_taxa[:, 1 : pca_total + 1]
    design = np.column_stack([
        np.ones(len(inputs.taxa)),
        scores,
        inputs.covariate_values,
    ])
    return design, scores
