"""Regression tests for cBLUP matrix-work reuse."""

from __future__ import annotations

import numpy as np
import pytest

from pygapit._typing import FloatMatrix, FloatVector
from pygapit.gs.blup import _emma_blup_with_incidence, cblup
from pygapit.gwas import mlm


def _reference_incidence_blup(
    y: FloatVector,
    X: FloatMatrix,
    K: FloatMatrix,
    Z: FloatMatrix,
    vg: float,
    ve: float,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """Evaluate the original explicit-precision formulation."""
    genetic_covariance = vg * K
    covariance = Z @ genetic_covariance @ Z.T + ve * np.eye(len(y))
    precision = np.linalg.inv(covariance)
    information = X.T @ precision @ X
    try:
        information_inverse = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        information_inverse = np.linalg.pinv(information)
    beta = information_inverse @ X.T @ precision @ y
    random_effect = genetic_covariance @ Z.T @ precision @ (y - X @ beta)

    kinship_inverse = np.linalg.pinv(K)
    random_information = Z.T @ Z / ve + kinship_inverse / vg
    try:
        conditional_covariance = np.linalg.inv(random_information)
    except np.linalg.LinAlgError:
        conditional_covariance = np.linalg.pinv(random_information)
    fixed_effect_correction = (
        genetic_covariance
        @ Z.T
        @ precision
        @ X
        @ information_inverse
        @ X.T
        @ precision
        @ Z
        @ genetic_covariance
    )
    pev: FloatVector = np.diag(conditional_covariance + fixed_effect_correction)
    return beta, random_effect, pev


@pytest.mark.parametrize("singular_kinship", [False, True])
def test_incidence_blup_solve_matches_explicit_precision(
    singular_kinship: bool,
) -> None:
    rng = np.random.default_rng(20260903)
    n, groups = 18, 6
    y: FloatVector = rng.normal(size=n)
    X: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    Z = np.zeros((n, groups), dtype=np.float64)
    Z[np.arange(n), np.arange(n) % groups] = 1.0
    factors = rng.normal(size=(groups, groups - singular_kinship))
    K: FloatMatrix = factors @ factors.T
    if not singular_kinship:
        K += np.eye(groups)

    expected = _reference_incidence_blup(y, X, K, Z, vg=1.3, ve=0.7)
    actual = _emma_blup_with_incidence(y, X, K, Z, vg=1.3, ve=0.7)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-11, atol=1e-12)


def test_cblup_builds_kinship_cluster_tree_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260903)
    n = 24
    genotype: FloatMatrix = rng.binomial(2, 0.35, size=(n, 40)).astype(np.float64)
    phenotype: FloatVector = rng.normal(size=n)
    design: FloatMatrix = np.ones((n, 1), dtype=np.float64)
    original = mlm._kinship_cluster_tree
    call_count = 0

    def counted_cluster_tree(K: FloatMatrix) -> FloatMatrix:
        nonlocal call_count
        call_count += 1
        return original(K)

    monkeypatch.setattr(mlm, "_kinship_cluster_tree", counted_cluster_tree)
    cblup(phenotype, design, genotype, group_to=12, ngrids=10)

    assert call_count == 1
