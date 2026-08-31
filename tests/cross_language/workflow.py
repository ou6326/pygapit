"""Shared inputs and R-side design construction for workflow parity tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pygapit.stats.kinship import vanraden_kinship
from tests.cross_language.r_bridge import RBridge

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

    covariate = pd.DataFrame(
        {
            "Taxa": taxa[reverse],
            "Covariate": fixed_covariate[reverse],
        }
    )
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
    r_pca = r_bridge.source_function(r_root, "GAPIT.PCA.R", "GAPIT.PCA")
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
    design = np.column_stack(
        [np.ones(len(inputs.taxa)), scores, inputs.covariate_values]
    )
    return design, scores
