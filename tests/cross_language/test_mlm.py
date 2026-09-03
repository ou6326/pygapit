"""Cross-language mixed-model alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.stats.emma import emma_remle, emmax_p3d
from pygapit.stats.kinship import vanraden_kinship
from tests.cross_language.r_bridge import RBridge, RObject


def _r_scalar(r_bridge: RBridge, result: RObject, name: str):
    values = r_bridge.float_array(r_bridge.component(result, name)).reshape(-1)
    value: np.float64 = values[0]
    return value


@pytest.mark.parametrize(
    "with_covariate", [False, True], ids=["intercept-only", "covariate"]
)
def test_emma_remle_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    with_covariate: bool,
) -> None:
    """Compare null-model REML likelihood and variance components."""
    if with_covariate:
        design = np.column_stack([np.ones(len(fixed_phenotype)), fixed_covariate])
    else:
        design = np.ones((len(fixed_phenotype), 1), dtype=float)
    kinship = vanraden_kinship(fixed_genotypes)

    r_emma_remle = r_bridge.source_function(r_root, "GAPIT.emma.R", "emma.REMLE")
    r_result = r_emma_remle(
        r_bridge.float_vector(fixed_phenotype),
        r_bridge.matrix(design),
        r_bridge.matrix(kinship),
        ngrids=100,
        llim=-10.0,
        ulim=10.0,
        esp=1e-10,
    )
    py_result = emma_remle(fixed_phenotype, design, kinship)

    r_reml = _r_scalar(r_bridge, r_result, "REML")
    r_delta = _r_scalar(r_bridge, r_result, "delta")
    r_vg = _r_scalar(r_bridge, r_result, "vg")
    r_ve = _r_scalar(r_bridge, r_result, "ve")
    r_h2 = r_vg / (r_vg + r_ve)

    nt.assert_allclose(py_result.reml, r_reml, rtol=1e-12, atol=1e-12)
    # R's uniroot and SciPy's brentq stop at slightly different roots while
    # producing the same optimum likelihood, so variance terms use 1e-6.
    nt.assert_allclose(py_result.delta, r_delta, rtol=1e-6, atol=1e-12)
    nt.assert_allclose(py_result.vg, r_vg, rtol=1e-6, atol=1e-12)
    nt.assert_allclose(py_result.ve, r_ve, rtol=1e-6, atol=1e-12)
    nt.assert_allclose(py_result.h2, r_h2, rtol=1e-6, atol=1e-12)


def test_emmax_p3d_scan_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
) -> None:
    """Compare SNP-level P3D p-values, effects, errors, and statistics."""
    design = np.column_stack([np.ones(len(fixed_phenotype)), fixed_covariate])
    covariates_with_taxa = np.column_stack([
        np.arange(len(fixed_phenotype), dtype=float),
        fixed_covariate,
    ])
    kinship = vanraden_kinship(fixed_genotypes)

    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, filename)
    r_emmax_p3d = r_bridge.source_function(r_root, "GAPIT.EMMAxP3D.R", "GAPIT.EMMAxP3D")
    r_null = r_bridge.evaluate("NULL")
    r_result = r_emmax_p3d(
        ys=r_bridge.matrix(fixed_phenotype[np.newaxis, :]),
        xs=r_bridge.matrix(fixed_genotypes),
        K=r_bridge.matrix(kinship),
        X0=r_bridge.matrix(design),
        CVI=r_bridge.matrix(covariates_with_taxa),
        file_from=1,
        file_to=1,
        file_fragment=fixed_genotypes.shape[1],
        fullGD=True,
        SNP_P3D=True,
        Timmer=r_null,
        Memory=r_null,
        optOnly=False,
    )
    py_result = emmax_p3d(fixed_phenotype, design, fixed_genotypes, kinship)

    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "ps")).reshape(-1)
    r_effects = r_bridge.float_array(
        r_bridge.component(r_result, "effect.est")
    ).reshape(-1)
    r_standard_errors = r_bridge.float_array(
        r_bridge.component(r_result, "stderr")
    ).reshape(-1)
    r_statistics = r_bridge.float_array(r_bridge.component(r_result, "tvalue")).reshape(
        -1
    )

    nt.assert_allclose(py_result.p_values, r_p_values, rtol=2e-6, atol=1e-12)
    nt.assert_allclose(py_result.effects, r_effects, rtol=2e-6, atol=1e-12)
    nt.assert_allclose(py_result.se, r_standard_errors, rtol=2e-6, atol=1e-12)
    nt.assert_allclose(py_result.stats, r_statistics, rtol=2e-6, atol=1e-12)
