"""Regression tests for the shared FarmCPU/BLINK reward calculation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as t_dist

from pygapit._typing import FloatMatrix, FloatVector, IntVector
from pygapit.gwas import glm as glm_module
from pygapit.gwas.glm import (
    GLMResult,
    glm_scan_with_cofactors,
    reward_substitute_cofactor_statistics,
)


def _reference_reward(
    result: GLMResult,
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    qtns: IntVector,
) -> GLMResult:
    """Evaluate the pre-batching pseudoinverse-per-marker formulation."""
    base_design: FloatMatrix = np.column_stack([X0, GD[:, qtns]])
    base_design_pinv = np.linalg.pinv(base_design)
    n = len(y)
    cofactor_count = len(qtns)
    cofactor_p = np.full((GD.shape[1], cofactor_count), np.nan, dtype=np.float64)
    standard_errors: FloatVector
    for marker in range(GD.shape[1]):
        marker_values: FloatVector = GD[:, marker]
        residualized = marker_values - base_design @ (base_design_pinv @ marker_values)
        if residualized @ residualized < 1e-8:
            continue
        design: FloatMatrix = np.column_stack([base_design, marker_values])
        degrees_of_freedom = n - design.shape[1]
        design_pinv = np.linalg.pinv(design)
        beta = design_pinv @ y
        residual = y - design @ beta
        mse = (residual @ residual) / degrees_of_freedom
        covariance = design_pinv @ design_pinv.T * mse
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        statistics = beta / standard_errors
        p_values = np.asarray(
            2.0 * t_dist.sf(np.abs(statistics), degrees_of_freedom),
            dtype=np.float64,
        )
        start = X0.shape[1]
        cofactor_p[marker] = p_values[start : start + cofactor_count]

    reward_p = np.asarray(
        [
            np.min(column[np.isfinite(column)]) if np.isfinite(column).any() else 1.0
            for column in cofactor_p.T
        ],
        dtype=np.float64,
    )
    degrees_of_freedom = n - base_design.shape[1]
    beta = base_design_pinv @ y
    residual = y - base_design @ beta
    mse = (residual @ residual) / degrees_of_freedom
    covariance = base_design_pinv @ base_design_pinv.T * mse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistics = beta / standard_errors
    start = X0.shape[1]

    p_values = result.p_values.copy()
    effects = result.effects.copy()
    se = result.se.copy()
    t_stats = result.t_stats.copy()
    p_values[qtns] = reward_p
    effects[qtns] = beta[start : start + cofactor_count]
    se[qtns] = standard_errors[start : start + cofactor_count]
    t_stats[qtns] = statistics[start : start + cofactor_count]
    return GLMResult(p_values, effects, se, t_stats, result.r2_full)


@pytest.mark.parametrize("rank_case", ["full", "near", "deficient"])
def test_reward_optimization_matches_original_formulation(
    rank_case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260902)
    n, m = 40, 24
    GD = rng.binomial(2, 0.35, size=(n, m)).astype(np.float64)
    if rank_case == "near":
        GD[:, 5] = GD[:, 1] + rng.normal(scale=1e-7, size=n)
    elif rank_case == "deficient":
        GD[:, 5] = GD[:, 1]
    y = rng.normal(size=n)
    X0: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])

    qtns: IntVector = np.array([1, 5], dtype=int)
    initial = glm_scan_with_cofactors(y, X0, GD, qtns)

    expected = _reference_reward(initial, y, X0, GD, qtns)
    monkeypatch.setattr(glm_module, "_REWARD_MARKER_BATCH_SIZE", 7)
    actual = reward_substitute_cofactor_statistics(initial, y, X0, GD, qtns)

    np.testing.assert_allclose(actual.p_values, expected.p_values, rtol=1e-10)
    np.testing.assert_allclose(actual.effects, expected.effects, rtol=1e-10)
    np.testing.assert_allclose(actual.se, expected.se, rtol=1e-10)
    np.testing.assert_allclose(actual.t_stats, expected.t_stats, rtol=1e-10)


def test_reward_full_rank_path_computes_one_pseudoinverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260903)
    n, m = 40, 24
    GD = rng.binomial(2, 0.35, size=(n, m)).astype(np.float64)
    y = rng.normal(size=n)
    X0: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    qtns: IntVector = np.array([1, 5], dtype=int)
    initial = glm_scan_with_cofactors(y, X0, GD, qtns)

    original_pinv = np.linalg.pinv
    pinv_calls = 0

    def counted_pinv(matrix: FloatMatrix) -> FloatMatrix:
        nonlocal pinv_calls
        pinv_calls += 1
        return original_pinv(matrix)

    monkeypatch.setattr(np.linalg, "pinv", counted_pinv)
    reward_substitute_cofactor_statistics(initial, y, X0, GD, qtns)

    assert pinv_calls == 1


def test_reward_batch_size_respects_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(glm_module, "_REWARD_BATCH_TARGET_BYTES", 560)
    monkeypatch.setattr(glm_module, "_REWARD_MARKER_BATCH_SIZE", 4096)

    assert glm_module._reward_marker_batch_size(10) == 7
    assert glm_module._reward_marker_batch_size(1000) == 1
