"""Correctness checks for the manual PCA crossover benchmark."""

from __future__ import annotations

import numpy as np

from benchmarks.benchmark_pca_crossover import (
    _sample_batched_tall_pca,
    _verify_equivalence,
)
from pygapit.stats.pca import compute_pca


def test_sample_batched_tall_candidate_matches_production_pca() -> None:
    rng = np.random.default_rng(20260912)
    genotype = rng.binomial(2, 0.35, size=(80, 30)).astype(np.float64)

    production = compute_pca(
        genotype,
        n_components=4,
        maf_filter=0.0,
        marker_workspace_mib=0.01,
    )
    sample_batched = _sample_batched_tall_pca(
        genotype,
        n_components=4,
        maf_filter=0.0,
        marker_workspace_mib=0.01,
    )

    _verify_equivalence(production, sample_batched)
