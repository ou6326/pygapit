"""Cross-language kinship alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.io.formats import impute_missing
from pygapit.stats.kinship import vanraden_kinship, zhang_kinship
from tests.cross_language.r_bridge import RBridge


def _r_vanraden(
    r_bridge: RBridge,
    r_root: Path,
    genotypes: NDArray[np.float64],
) -> NDArray[np.float64]:
    function = r_bridge.source_function(
        r_root,
        "GAPIT.kinship.VanRaden.R",
        "GAPIT.kinship.VanRaden",
    )
    return r_bridge.float_array(function(r_bridge.matrix(genotypes)))


@pytest.mark.parametrize(
    "include_monomorphic", [False, True], ids=["variable", "monomorphic"]
)
def test_vanraden_kinship_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    include_monomorphic: bool,
) -> None:
    """Compare VanRaden matrices, including invariant-marker removal."""
    genotypes = fixed_genotypes
    if include_monomorphic:
        invariant = np.full((fixed_genotypes.shape[0], 1), 2.0, dtype=float)
        genotypes = np.column_stack([fixed_genotypes, invariant])

    r_matrix = _r_vanraden(r_bridge, r_root, genotypes)
    py_matrix = vanraden_kinship(genotypes)

    assert r_matrix.shape == py_matrix.shape == (genotypes.shape[0],) * 2
    nt.assert_allclose(py_matrix, r_matrix, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["extreme_frequency", "rank_deficient"])
def test_vanraden_pathological_spectra_match_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    case: str,
) -> None:
    """Preserve GAPIT's relationship matrix at frequency/rank boundaries."""
    if case == "extreme_frequency":
        rare = np.zeros(len(fixed_genotypes), dtype=np.float64)
        rare[0] = 1.0
        near_fixed = np.full(len(fixed_genotypes), 2.0, dtype=np.float64)
        near_fixed[-1] = 1.0
        genotypes = np.column_stack((rare, near_fixed, fixed_genotypes[:, :2]))
    else:
        base = fixed_genotypes[:, :3]
        genotypes = np.column_stack((base, base[:, 0], 2.0 - base[:, 1]))

    r_matrix = _r_vanraden(r_bridge, r_root, genotypes)
    py_matrix = vanraden_kinship(genotypes, marker_workspace_mib=0.001)

    nt.assert_allclose(py_matrix, r_matrix, rtol=1e-12, atol=1e-12)


def test_middle_imputation_then_vanraden_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
) -> None:
    """Match GAPIT when both implementations first apply middle imputation."""
    missing = fixed_genotypes.copy()
    missing[[0, 3, 7], [0, 2, 5]] = np.nan
    python_imputed = impute_missing(missing, method="middle")

    r_bridge.source(r_root, "GAPIT.Imputation.R")
    r_impute = r_bridge.function(
        "function(x) apply(x, 2, function(column) "
        "GAPIT.Imputation(column, impute='Middle', byRow=TRUE))"
    )
    r_imputed = r_bridge.float_array(r_impute(r_bridge.matrix(missing)))
    nt.assert_array_equal(python_imputed, r_imputed)

    r_matrix = _r_vanraden(r_bridge, r_root, r_imputed)
    py_matrix = vanraden_kinship(python_imputed, marker_workspace_mib=0.001)
    nt.assert_allclose(py_matrix, r_matrix, rtol=1e-12, atol=1e-12)


def test_all_monomorphic_vanraden_has_safe_documented_divergence(
    r_bridge: RBridge,
    r_root: Path,
) -> None:
    """Return identity where GAPIT 3.5 divides its zero matrix by zero."""
    genotypes = np.zeros((8, 4), dtype=np.float64)

    r_matrix = _r_vanraden(r_bridge, r_root, genotypes)
    with pytest.warns(UserWarning, match="All SNPs are monomorphic"):
        py_matrix = vanraden_kinship(genotypes)

    assert np.all(~np.isfinite(r_matrix))
    nt.assert_array_equal(py_matrix, np.eye(len(genotypes)))


def test_zhang_kinship_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
) -> None:
    """Compare the scaled Zhang relationship matrix with GAPIT 3.5."""
    r_zhang = r_bridge.source_function(
        r_root,
        "GAPIT.kinship.Zhang.R",
        "GAPIT.kinship.Zhang",
    )
    r_matrix = r_bridge.float_array(r_zhang(r_bridge.matrix(fixed_genotypes)))
    py_matrix = zhang_kinship(fixed_genotypes)

    assert r_matrix.shape == py_matrix.shape == (fixed_genotypes.shape[0],) * 2
    nt.assert_allclose(py_matrix, r_matrix, rtol=1e-12, atol=1e-12)
