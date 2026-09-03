"""Boundary-case alignment tests for GAPIT's EMMAX/P3D scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.stats.emma import emmax_p3d
from pygapit.stats.kinship import vanraden_kinship
from tests.cross_language.r_bridge import RBridge, RList


def _run_r_p3d(
    r_bridge: RBridge,
    r_root: Path,
    phenotype: NDArray[np.float64],
    covariate: NDArray[np.float64],
    genotypes: NDArray[np.float64],
    kinship: NDArray[np.float64],
    snp_impute: str = "Middle",
) -> RList:
    design = np.column_stack([np.ones(len(phenotype)), covariate])
    covariates_with_taxa = np.column_stack([
        np.arange(len(phenotype), dtype=float),
        covariate,
    ])
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, filename)
    r_emmax_p3d = r_bridge.source_function(
        r_root,
        "GAPIT.EMMAxP3D.R",
        "GAPIT.EMMAxP3D",
        returns=RList,
    )
    r_null = r_bridge.evaluate("NULL")
    return r_emmax_p3d(
        ys=r_bridge.matrix(phenotype[np.newaxis, :]),
        xs=r_bridge.matrix(genotypes),
        K=r_bridge.matrix(kinship),
        X0=r_bridge.matrix(design),
        CVI=r_bridge.matrix(covariates_with_taxa),
        file_from=1,
        file_to=1,
        file_fragment=genotypes.shape[1],
        fullGD=True,
        SNP_P3D=True,
        Timmer=r_null,
        Memory=r_null,
        optOnly=False,
        SNP_impute=snp_impute,
    )


def test_p3d_monomorphic_marker_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
) -> None:
    """A monomorphic marker has p=1 and undefined effect statistics."""
    genotypes = np.column_stack([
        fixed_genotypes,
        np.full(len(fixed_phenotype), 2.0, dtype=float),
    ])
    kinship = vanraden_kinship(genotypes)
    design = np.column_stack([np.ones(len(fixed_phenotype)), fixed_covariate])
    r_result = _run_r_p3d(
        r_bridge,
        r_root,
        fixed_phenotype,
        fixed_covariate,
        genotypes,
        kinship,
    )
    py_result = emmax_p3d(fixed_phenotype, design, genotypes, kinship)

    for py_values, r_name in (
        (py_result.p_values, "ps"),
        (py_result.effects, "effect.est"),
        (py_result.se, "stderr"),
        (py_result.stats, "tvalue"),
    ):
        r_values = r_bridge.float_array(r_bridge.component(r_result, r_name)).reshape(
            -1
        )
        nt.assert_allclose(py_values, r_values, rtol=2e-6, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    ("r_impute", "py_impute"),
    [("Middle", "middle"), ("Major", "major"), ("Minor", "minor")],
)
def test_p3d_missing_genotype_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    r_impute: str,
    py_impute: str,
) -> None:
    """Numeric-genotype imputation follows GAPIT's three supported policies."""
    genotypes = fixed_genotypes.copy()
    genotypes[0, 0] = np.nan
    kinship = vanraden_kinship(fixed_genotypes)
    design = np.column_stack([np.ones(len(fixed_phenotype)), fixed_covariate])
    r_result = _run_r_p3d(
        r_bridge,
        r_root,
        fixed_phenotype,
        fixed_covariate,
        genotypes,
        kinship,
        snp_impute=r_impute,
    )
    py_result = emmax_p3d(
        fixed_phenotype,
        design,
        genotypes,
        kinship,
        snp_impute=py_impute,
    )
    for py_values, r_name in (
        (py_result.p_values, "ps"),
        (py_result.effects, "effect.est"),
        (py_result.se, "stderr"),
        (py_result.stats, "tvalue"),
    ):
        r_values = r_bridge.float_array(r_bridge.component(r_result, r_name)).reshape(
            -1
        )
        nt.assert_allclose(py_values, r_values, rtol=2e-6, atol=1e-12, equal_nan=True)
