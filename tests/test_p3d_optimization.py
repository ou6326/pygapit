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

    with (
        patch.object(np.linalg, "lstsq", wraps=np.linalg.lstsq) as tracked_lstsq,
        patch(
            "pygapit.stats.emma._eigen_L_wo_Z",
            side_effect=AssertionError("ordinary P3D should use Cholesky whitening"),
        ),
    ):
        result = emma.emmax_p3d(phenotype, design, genotype, kinship)

    assert tracked_lstsq.call_count == 0
    assert np.all(np.isfinite(result.p_values))
    np.testing.assert_array_equal(result.p_values[:2], 1.0)
    assert np.all(np.isnan(result.effects[:2]))
    assert np.all(np.isnan(result.se[:2]))
    assert np.all(np.isnan(result.stats[:2]))


def test_emmax_cholesky_fallback_matches_spectral_whitening() -> None:
    rng = np.random.default_rng(20260904)
    genotype = rng.binomial(2, 0.35, size=(30, 20)).astype(np.float64)
    covariate = rng.normal(size=30)
    design = np.column_stack([np.ones(30), covariate])
    phenotype = genotype[:, :3] @ np.array([0.7, -0.4, 0.2])
    phenotype += 0.3 * covariate + rng.normal(0.0, 0.5, size=30)
    kinship = np.eye(30, dtype=np.float64)

    cholesky_result = emma.emmax_p3d(phenotype, design, genotype, kinship)
    with patch.object(
        np.linalg,
        "cholesky",
        side_effect=np.linalg.LinAlgError("forced fallback"),
    ):
        spectral_result = emma.emmax_p3d(phenotype, design, genotype, kinship)

    for cholesky_values, spectral_values in (
        (cholesky_result.p_values, spectral_result.p_values),
        (cholesky_result.effects, spectral_result.effects),
        (cholesky_result.se, spectral_result.se),
        (cholesky_result.stats, spectral_result.stats),
    ):
        np.testing.assert_allclose(
            cholesky_values,
            spectral_values,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


def test_emmax_marker_batches_respect_memory_target() -> None:
    n_individuals = 1_000_000
    batch_size = emma._emmax_marker_batch_size(n_individuals)

    assert batch_size >= 1
    assert batch_size <= emma._EMMAX_MARKER_BATCH_SIZE
    assert (
        batch_size * n_individuals * emma._FLOAT64_BYTES
        <= emma._EMMAX_MARKER_BATCH_TARGET_BYTES
    )
