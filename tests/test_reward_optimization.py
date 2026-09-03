"""Regression tests for the shared FarmCPU/BLINK reward calculation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as t_dist

from pygapit._typing import FloatMatrix, FloatVector, IntVector
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
    """Evaluate the original three-pseudoinverse-per-marker formulation."""
    base_design: FloatMatrix = np.column_stack([X0, GD[:, qtns]])
    n = len(y)
    cofactor_count = len(qtns)
    cofactor_p = np.full((GD.shape[1], cofactor_count), np.nan, dtype=np.float64)
    standard_errors: FloatVector
    for marker in range(GD.shape[1]):
        marker_values: FloatVector = GD[:, marker]
        residualized = marker_values - base_design @ (
            np.linalg.pinv(base_design) @ marker_values
        )
        if residualized @ residualized < 1e-8:
            continue
        design: FloatMatrix = np.column_stack([base_design, marker_values])
        degrees_of_freedom = n - design.shape[1]
        beta = np.linalg.pinv(design) @ y
        residual = y - design @ beta
        mse = (residual @ residual) / degrees_of_freedom
        covariance = np.linalg.pinv(design.T @ design) * mse
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
    beta = np.linalg.pinv(base_design) @ y
    residual = y - base_design @ beta
    mse = (residual @ residual) / degrees_of_freedom
    covariance = np.linalg.pinv(base_design.T @ base_design) * mse
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


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_reward_optimization_matches_original_formulation(rank_deficient: bool) -> None:
    rng = np.random.default_rng(20260902)
    n, m = 40, 24
    GD = rng.binomial(2, 0.35, size=(n, m)).astype(np.float64)
    if rank_deficient:
        GD[:, 5] = GD[:, 1]
    y = rng.normal(size=n)
    X0: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])

    qtns: IntVector = np.array([1, 5], dtype=int)
    initial = glm_scan_with_cofactors(y, X0, GD, qtns)

    expected = _reference_reward(initial, y, X0, GD, qtns)
    actual = reward_substitute_cofactor_statistics(initial, y, X0, GD, qtns)

    np.testing.assert_allclose(actual.p_values, expected.p_values, rtol=1e-10)
    np.testing.assert_allclose(actual.effects, expected.effects, rtol=1e-10)
    np.testing.assert_allclose(actual.se, expected.se, rtol=1e-10)
    np.testing.assert_allclose(actual.t_stats, expected.t_stats, rtol=1e-10)
