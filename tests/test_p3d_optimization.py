"""Regression tests for the batched EMMAX/P3D marker scan."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from pygapit.stats import emma


def test_emmax_complete_markers_avoid_per_marker_linear_solves() -> None:
    rng = np.random.default_rng(20260904)
    genotype = rng.binomial(2, 0.35, size=(40, 80)).astype(np.float64)
    covariate = rng.normal(size=40)
    genotype[:, 0] = 1.0
    genotype[:, 1] = covariate
    design = np.column_stack([np.ones(40), covariate])
    phenotype = genotype[:, 2:5] @ np.array([0.7, -0.4, 0.2])
    phenotype += 0.3 * covariate + rng.normal(0.0, 0.5, size=40)
    kinship = np.eye(40, dtype=np.float64)

    with patch.object(np.linalg, "lstsq", wraps=np.linalg.lstsq) as tracked_lstsq:
        result = emma.emmax_p3d(phenotype, design, genotype, kinship)

    assert tracked_lstsq.call_count == 0
    assert np.all(np.isfinite(result.p_values))
    np.testing.assert_array_equal(result.p_values[:2], 1.0)
    assert np.all(np.isnan(result.effects[:2]))
    assert np.all(np.isnan(result.se[:2]))
    assert np.all(np.isnan(result.stats[:2]))


def test_emmax_marker_batches_respect_memory_target() -> None:
    n_individuals = 1_000_000
    batch_size = emma._emmax_marker_batch_size(n_individuals)

    assert batch_size >= 1
    assert batch_size <= emma._EMMAX_MARKER_BATCH_SIZE
    assert (
        batch_size * n_individuals * emma._FLOAT64_BYTES
        <= emma._EMMAX_MARKER_BATCH_TARGET_BYTES
    )
