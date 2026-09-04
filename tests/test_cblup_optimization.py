"""Regression tests for cBLUP matrix-work reuse."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from pygapit._typing import FloatMatrix, FloatVector
from pygapit.gs import blup as blup_module
from pygapit.gs.blup import _emma_blup_with_incidence, cblup
from pygapit.gwas import mlm
from pygapit.stats import emma as emma_module
from pygapit.stats.emma import EMMAResult, _eigen_R_w_Z, emma_remle


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
):
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


def test_incidence_blup_uses_cholesky_for_observation_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260904)
    n, groups = 18, 6
    y: FloatVector = rng.normal(size=n)
    X: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    Z = np.zeros((n, groups), dtype=np.float64)
    Z[np.arange(n), np.arange(n) % groups] = 1.0
    factors = rng.normal(size=(groups, groups))
    K: FloatMatrix = factors @ factors.T

    def unexpected_general_solve(
        matrix: FloatMatrix,
        right_hand_side: FloatMatrix,
    ) -> None:
        pytest.fail("positive-definite observation covariance should use Cholesky")

    monkeypatch.setattr(np.linalg, "solve", unexpected_general_solve)
    _emma_blup_with_incidence(y, X, K, Z, vg=1.3, ve=0.7)


def test_cblup_builds_kinship_cluster_tree_once(
    monkeypatch: pytest.MonkeyPatch,
):
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


@pytest.mark.parametrize(("ngrids", "expected_final_refits"), [(100, 0), (10, 1)])
def test_cblup_reuses_only_matching_candidate_reml_fit(
    monkeypatch: pytest.MonkeyPatch,
    ngrids: int,
    expected_final_refits: int,
) -> None:
    rng = np.random.default_rng(20260904)
    n = 24
    genotype: FloatMatrix = rng.binomial(2, 0.35, size=(n, 40)).astype(np.float64)
    phenotype: FloatVector = rng.normal(size=n)
    design: FloatMatrix = np.ones((n, 1), dtype=np.float64)
    final_refit_count = 0

    def counted_final_refit(
        y: FloatVector,
        X: FloatMatrix,
        K: FloatMatrix,
        ngrids: int = 100,
        llim: float = -10.0,
        ulim: float = 10.0,
        esp: float = 1e-10,
        Z: FloatMatrix | None = None,
    ) -> EMMAResult:
        nonlocal final_refit_count
        final_refit_count += 1
        return emma_remle(y, X, K, ngrids, llim, ulim, esp, Z)

    monkeypatch.setattr(blup_module, "emma_remle", counted_final_refit)

    cblup(phenotype, design, genotype, group_to=12, ngrids=ngrids)

    # The default candidate fit is reusable as-is.  A custom final grid retains
    # the established refit so callers receive the requested resolution.
    assert final_refit_count == expected_final_refits


def test_incidence_partial_eigendecomposition_matches_full_reml_space():
    rng = np.random.default_rng(20260903)
    n, groups = 30, 8
    X: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    Z = np.zeros((n, groups), dtype=np.float64)
    Z[np.arange(n), np.arange(n) % groups] = 1.0
    factors = rng.normal(size=(groups, groups))
    K: FloatMatrix = factors @ factors.T + np.eye(groups)

    projection_coefficients, *_ = np.linalg.lstsq(X, Z, rcond=None)
    residualized_Z = Z - X @ projection_coefficients
    residual_covariance = residualized_Z @ K @ residualized_Z.T
    residual_covariance = (residual_covariance + residual_covariance.T) / 2.0
    expected_values, expected_random_basis = np.linalg.eigh(residual_covariance)
    expected_values = expected_values[::-1]
    expected_random_basis = expected_random_basis[:, ::-1]
    random_rank = groups - X.shape[1]
    fixed_basis, _ = np.linalg.qr(X, mode="reduced")
    combined: FloatMatrix = np.column_stack([
        expected_random_basis[:, :random_rank],
        fixed_basis,
    ])
    expected_complete_basis, _ = np.linalg.qr(combined, mode="complete")
    selected = [*range(random_rank), *range(groups, n)]
    expected_reml_basis = expected_complete_basis[:, selected]

    actual_values, actual_model_basis = _eigen_R_w_Z(Z, K, X)

    np.testing.assert_allclose(actual_values, expected_values[:random_rank])
    np.testing.assert_allclose(
        actual_model_basis.T @ actual_model_basis,
        np.eye(groups),
        rtol=1e-10,
        atol=1e-11,
    )
    for column in range(random_rank):
        assert abs(
            actual_model_basis[:, column] @ expected_reml_basis[:, column]
        ) == pytest.approx(1.0)
    expected_model_basis = expected_complete_basis[:, :groups]
    np.testing.assert_allclose(
        actual_model_basis @ actual_model_basis.T,
        expected_model_basis @ expected_model_basis.T,
        rtol=1e-10,
        atol=1e-11,
    )

    phenotype: FloatVector = rng.normal(size=n)
    expected_coordinates = expected_reml_basis.T @ phenotype
    actual_model_coordinates = actual_model_basis.T @ phenotype
    actual_residual_sum_squares = (
        phenotype @ phenotype - actual_model_coordinates @ actual_model_coordinates
    )
    np.testing.assert_allclose(
        actual_residual_sum_squares,
        expected_coordinates[random_rank:] @ expected_coordinates[random_rank:],
        rtol=1e-10,
        atol=1e-11,
    )


def test_incidence_group_space_falls_back_for_low_rank_kinship():
    rng = np.random.default_rng(20260903)
    n, groups = 30, 8
    X: FloatMatrix = np.ones((n, 1), dtype=np.float64)
    Z = np.zeros((n, groups), dtype=np.float64)
    Z[np.arange(n), np.arange(n) % groups] = 1.0
    factors = rng.normal(size=(groups, 3))
    K: FloatMatrix = factors @ factors.T

    values, model_basis = _eigen_R_w_Z(Z, K, X)

    assert values.shape == (groups - X.shape[1],)
    assert model_basis.shape == (n, groups)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(model_basis))
    np.testing.assert_allclose(
        model_basis.T @ model_basis,
        np.eye(groups),
        rtol=1e-10,
        atol=1e-11,
    )


def test_incidence_centered_kinship_uses_one_eigendecomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260904)
    n, groups = 30, 8
    X: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    Z = np.zeros((n, groups), dtype=np.float64)
    Z[np.arange(n), np.arange(n) % groups] = 1.0
    group_sizes = Z.sum(axis=0)
    factors = rng.normal(size=(groups, groups - 1))
    factors -= np.outer(group_sizes, group_sizes @ factors) / (
        group_sizes @ group_sizes
    )
    K: FloatMatrix = factors @ factors.T
    original_eigh = np.linalg.eigh
    eigh_calls = 0

    def counted_eigh(matrix: FloatMatrix) -> tuple[FloatVector, FloatMatrix]:
        nonlocal eigh_calls
        eigh_calls += 1
        return original_eigh(matrix)

    monkeypatch.setattr(np.linalg, "eigh", counted_eigh)
    values, model_basis = _eigen_R_w_Z(Z, K, X)

    assert eigh_calls == 1
    assert values.shape == (groups - X.shape[1],)
    np.testing.assert_allclose(
        model_basis.T @ model_basis,
        np.eye(groups),
        rtol=1e-10,
        atol=1e-11,
    )


@pytest.mark.parametrize("groups", [15, 24])
def test_incidence_benchmark_supported_ratios_use_group_space(
    monkeypatch: pytest.MonkeyPatch,
    groups: int,
) -> None:
    rng = np.random.default_rng(20260904)
    n = 30
    X: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    Z = np.zeros((n, groups), dtype=np.float64)
    Z[np.arange(n), np.arange(n) % groups] = 1.0
    factors = rng.normal(size=(groups, groups))
    K: FloatMatrix = factors @ factors.T + np.eye(groups)

    def unexpected_observation_eigh(
        matrix: FloatMatrix,
        *,
        subset_by_index: tuple[int, int],
    ) -> None:
        pytest.fail(
            "benchmark-supported incidence ratios should not use observation-space eigh"
        )

    monkeypatch.setattr(emma_module, "scipy_eigh", unexpected_observation_eigh)

    values, model_basis = _eigen_R_w_Z(Z, K, X)

    assert values.shape == (groups - X.shape[1],)
    assert model_basis.shape == (n, groups)


def test_incidence_full_rank_spectrum_avoids_combined_qr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260904)
    n, groups = 30, 8
    X: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    Z = np.zeros((n, groups), dtype=np.float64)
    Z[np.arange(n), np.arange(n) % groups] = 1.0
    factors = rng.normal(size=(groups, groups))
    K: FloatMatrix = factors @ factors.T + np.eye(groups)
    original_qr = np.linalg.qr
    qr_calls = 0

    def counted_qr(
        matrix: FloatMatrix,
        mode: Literal["reduced"] = "reduced",
    ) -> tuple[FloatMatrix, FloatMatrix]:
        nonlocal qr_calls
        qr_calls += 1
        return original_qr(matrix, mode=mode)

    monkeypatch.setattr(np.linalg, "qr", counted_qr)
    _eigen_R_w_Z(Z, K, X)

    assert qr_calls == 1
