"""Cross-language compressed-kinship alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.gwas.mlm import compress_kinship
from pygapit.stats.kinship import vanraden_kinship
from tests.cross_language.r_bridge import RBridge


@pytest.mark.parametrize("n_groups", [3, 6], ids=["coarse", "fine"])
def test_compressed_kinship_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    n_groups: int,
) -> None:
    """Compare GAPIT average-linkage memberships and group kinship."""
    kinship = vanraden_kinship(fixed_genotypes)
    kinship_with_taxa = np.column_stack([
        np.arange(kinship.shape[0], dtype=float),
        kinship,
    ])
    r_bridge.source(r_root, "GAPIT.Timmer.R")
    r_bridge.source(r_root, "GAPIT.Memory.R")
    r_bridge.source(r_root, "GAPIT.Compress.R")
    r_compress = r_bridge.function(
        "function(KI, groups) { "
        "out <- GAPIT.Compress(KI, GN=groups, Timmer=NULL, Memory=NULL); "
        "list(labels=as.numeric(out$GA[,2]), kinship=out$KG) }"
    )
    r_result = r_compress(r_bridge.matrix(kinship_with_taxa), groups=n_groups)
    r_labels = r_bridge.float_array(r_bridge.component(r_result, "labels")).astype(int)
    r_kinship = r_bridge.float_array(r_bridge.component(r_result, "kinship"))

    py_kinship, py_incidence = compress_kinship(kinship, n_groups)
    py_labels = np.argmax(py_incidence, axis=1) + 1

    nt.assert_array_equal(
        py_labels[:, None] == py_labels[None, :],
        r_labels[:, None] == r_labels[None, :],
    )
    r_group_order = [
        r_labels[np.argmax(py_labels == label)] - 1 for label in range(1, n_groups + 1)
    ]
    nt.assert_allclose(
        py_kinship,
        r_kinship[np.ix_(r_group_order, r_group_order)],
        rtol=1e-12,
        atol=1e-12,
    )
