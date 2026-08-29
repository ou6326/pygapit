"""Cross-language principal-component alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
from numpy.typing import NDArray

from pygapit.stats.pca import compute_pca
from tests.cross_language.r_bridge import RBridge


def test_pca_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
) -> None:
    """Compare PCA eigenvalues and scores, allowing arbitrary component signs."""
    r_pca = r_bridge.source_function(r_root, "GAPIT.PCA.R", "GAPIT.PCA")
    r_result = r_pca(
        r_bridge.matrix(fixed_genotypes),
        r_bridge.float_vector(np.arange(fixed_genotypes.shape[0], dtype=float)),
        PC_number=3,
        file_output=False,
        PCA_total=3,
    )
    r_scores_with_taxa = r_bridge.float_array(r_bridge.component(r_result, "PCs"))
    if r_scores_with_taxa.shape[0] != fixed_genotypes.shape[0]:
        r_scores_with_taxa = r_scores_with_taxa.T
    r_scores = r_scores_with_taxa[:, 1:4]
    r_eigenvalues = r_bridge.float_array(r_bridge.component(r_result, "EV")).reshape(-1)
    py_result = compute_pca(fixed_genotypes, n_components=3, maf_filter=0.0)

    nt.assert_allclose(py_result.eigenvalues, r_eigenvalues[:3], rtol=1e-12, atol=1e-12)
    nt.assert_allclose(
        py_result.var_explained,
        r_eigenvalues[:3] / np.sum(r_eigenvalues),
        rtol=1e-12,
        atol=1e-12,
    )
    for component in range(3):
        sign = np.sign(np.dot(py_result.scores[:, component], r_scores[:, component]))
        if sign == 0:
            sign = 1.0
        nt.assert_allclose(
            py_result.scores[:, component],
            sign * r_scores[:, component],
            rtol=1e-12,
            atol=1e-12,
        )
