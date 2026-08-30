"""Cross-language kinship alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.stats.kinship import vanraden_kinship, zhang_kinship
from tests.cross_language.r_bridge import RBridge


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

    r_vanraden = r_bridge.source_function(
        r_root,
        "GAPIT.kinship.VanRaden.R",
        "GAPIT.kinship.VanRaden",
    )
    r_matrix = r_bridge.float_array(r_vanraden(r_bridge.matrix(genotypes)))
    py_matrix = vanraden_kinship(genotypes)

    assert r_matrix.shape == py_matrix.shape == (genotypes.shape[0],) * 2
    nt.assert_allclose(py_matrix, r_matrix, rtol=1e-12, atol=1e-12)


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
