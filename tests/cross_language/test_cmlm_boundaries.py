"""Degenerate native-incidence REML behavior against GAPIT 3.5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.gwas.mlm import compress_kinship
from pygapit.stats.emma import EMMAResult, emma_remle
from pygapit.stats.kinship import vanraden_kinship
from tests.cross_language.r_bridge import RBridge

FloatArray = NDArray[np.float64]


def _incidence_inputs(
    fixed_genotypes: FloatArray,
    fixed_phenotype: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    full_kinship = vanraden_kinship(fixed_genotypes)
    kinship, incidence = compress_kinship(full_kinship, 5)
    covariate = np.linspace(-1.0, 1.0, len(fixed_phenotype), dtype=np.float64)
    design = np.column_stack([np.ones(len(fixed_phenotype)), covariate])
    return fixed_phenotype, design, kinship, incidence


def _r_remle(
    r_bridge: RBridge,
    r_root: Path,
    y: FloatArray,
    design: FloatArray,
    kinship: FloatArray,
    incidence: FloatArray,
) -> tuple[bool, FloatArray]:
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
    ):
        r_bridge.source(r_root, filename)
    r_fit = r_bridge.function(
        "function(y, X, K, Z) {"
        " tryCatch({"
        "   fit <- GAPIT.emma.REMLE(y, X, K, Z=Z);"
        "   list(ok=1, values=c(fit$REML, fit$delta, fit$vg, fit$ve))"
        " }, error=function(e) list(ok=0, values=rep(NA_real_, 4)))"
        "}"
    )
    result = r_fit(
        r_bridge.float_vector(y),
        r_bridge.matrix(design),
        r_bridge.matrix(kinship),
        r_bridge.matrix(incidence),
    )
    ok = bool(r_bridge.float_array(r_bridge.component(result, "ok"))[0])
    values = r_bridge.float_array(r_bridge.component(result, "values"))
    return ok, values


def _result_values(result: EMMAResult) -> FloatArray:
    return np.array([result.reml, result.delta, result.vg, result.ve])


@pytest.mark.parametrize("redundancy", ["duplicate", "empty"])
def test_redundant_native_incidence_levels_match_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: FloatArray,
    fixed_phenotype: FloatArray,
    redundancy: str,
) -> None:
    """Redundant random-effect columns remain a defined marginal model."""
    y, design, kinship, incidence = _incidence_inputs(fixed_genotypes, fixed_phenotype)
    extra_column = incidence[:, 0] if redundancy == "duplicate" else np.zeros(len(y))
    incidence = np.column_stack([incidence, extra_column])
    expanded_kinship = np.zeros((6, 6), dtype=np.float64)
    expanded_kinship[:5, :5] = kinship
    expanded_kinship[5, 5] = np.mean(np.diag(kinship))

    r_ok, r_values = _r_remle(r_bridge, r_root, y, design, expanded_kinship, incidence)
    py_values = _result_values(emma_remle(y, design, expanded_kinship, Z=incidence))

    assert r_ok
    nt.assert_allclose(py_values, r_values, rtol=1e-5, atol=1e-12)


def test_near_collinear_fixed_effects_have_stable_native_incidence_fit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: FloatArray,
    fixed_phenotype: FloatArray,
) -> None:
    """Least-squares residualization avoids squaring the design condition number."""
    y, _design, kinship, incidence = _incidence_inputs(fixed_genotypes, fixed_phenotype)
    covariate = np.linspace(-1.0, 1.0, len(y), dtype=np.float64)
    perturbation = np.where(np.arange(len(y)) % 2 == 0, 1.0, -1.0)
    design = np.column_stack([
        np.ones(len(y)),
        covariate,
        covariate + 1e-6 * perturbation,
    ])

    r_ok, r_values = _r_remle(r_bridge, r_root, y, design, kinship, incidence)
    py_values = _result_values(emma_remle(y, design, kinship, Z=incidence))

    assert r_ok
    assert np.isfinite(py_values).all()
    nt.assert_allclose(py_values[0], r_values[0], rtol=1e-6, atol=1e-12)
    nt.assert_allclose(py_values[1:], r_values[1:], rtol=1e-4, atol=1e-12)


def test_singular_fixed_effects_are_rejected_instead_of_returning_gapit_sentinel(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: FloatArray,
    fixed_phenotype: FloatArray,
) -> None:
    """Reject GAPIT's all-zero REML sentinel as an invalid model fit."""
    y, _design, kinship, incidence = _incidence_inputs(fixed_genotypes, fixed_phenotype)
    covariate = np.linspace(-1.0, 1.0, len(y), dtype=np.float64)
    design = np.column_stack([np.ones(len(y)), covariate, covariate])

    r_ok, r_values = _r_remle(r_bridge, r_root, y, design, kinship, incidence)

    assert r_ok
    nt.assert_array_equal(r_values, np.zeros(4))
    with pytest.raises(ValueError, match="linearly independent"):
        emma_remle(y, design, kinship, Z=incidence)
