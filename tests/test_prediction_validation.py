"""Statistical and split contracts for standalone genomic prediction."""

import numpy as np
import pytest

from pygapit.gs.validation import cross_validate_gblup, cross_validate_rrblup
from pygapit.models.genomic_prediction import GBLUP, RR_BLUP


@pytest.mark.parametrize("markers", [4, 40])
def test_rrblup_matches_augmented_ridge_equations(markers: int) -> None:
    rng = np.random.default_rng(41)
    z = rng.normal(size=(17, markers)) + 2
    y = 20 + z[:, 0] + rng.normal(size=17)
    result = cross_validate_rrblup(y, z, n_folds=5, lambda_=2.5)
    expected = np.empty(len(y))
    for fold in range(5):
        test = result.fold_ids == fold
        design = np.column_stack((np.ones(sum(~test)), z[~test]))
        penalty = np.diag(np.r_[0.0, np.full(markers, 2.5)])
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y[~test])
        expected[test] = np.column_stack((np.ones(sum(test)), z[test])) @ beta
    np.testing.assert_allclose(result.predictions, expected, rtol=1e-11, atol=1e-11)
    assert np.bincount(result.fold_ids).tolist() == [4, 4, 3, 3, 3]
    np.testing.assert_allclose(result.rmse, np.sqrt(np.mean((y - expected) ** 2)))


@pytest.mark.parametrize("method", ["rr", "gb"])
def test_legacy_wrapper_uses_complete_fold_predictions(method: str) -> None:
    rng = np.random.default_rng(33)
    z = rng.normal(size=(17, 6))
    y = 10 + z[:, 0] + rng.normal(size=17)
    if method == "rr":
        result = cross_validate_rrblup(y, z, lambda_=3)
        gebv, accuracy = RR_BLUP(y, z, lambda_=3)
        centered = z - z.mean(axis=0)
        expected = centered @ np.linalg.solve(
            centered.T @ centered + 3 * np.eye(6), centered.T @ (y - y.mean())
        )
    else:
        from pygapit.gs.blup import gblup

        k = z @ z.T / 6
        result = cross_validate_gblup(y, k)
        gebv, accuracy = GBLUP(y, k)
        expected = gblup(y, np.ones((len(y), 1)), k).gebv
    np.testing.assert_allclose(accuracy, result.pearson_r, atol=1e-12)
    np.testing.assert_allclose(gebv, expected, atol=1e-10)


@pytest.mark.parametrize("method", ["rr", "gb"])
def test_heldout_phenotype_cannot_change_its_predictions(method: str) -> None:
    rng = np.random.default_rng(123)
    z = rng.normal(size=(18, 7))
    y = rng.normal(size=18)
    changed = y.copy()
    changed[:6] += 1000
    if method == "rr":
        first = cross_validate_rrblup(y, z, n_folds=3)
        second = cross_validate_rrblup(changed, z, n_folds=3)
    else:
        k = z @ z.T / 7
        first = cross_validate_gblup(y, k, n_folds=3)
        second = cross_validate_gblup(changed, k, n_folds=3)
    np.testing.assert_array_equal(first.predictions[:6], second.predictions[:6])
    assert first.regularization[0] == second.regularization[0]


def test_grouped_seeded_and_explicit_folds() -> None:
    rng = np.random.default_rng(16)
    z = rng.normal(size=(24, 5))
    y = rng.normal(size=24)
    groups = np.repeat(np.arange(8), 3)
    first = cross_validate_rrblup(y, z, n_folds=4, groups=groups, seed=2, lambda_=1)
    for group in np.unique(groups):
        assert len(np.unique(first.fold_ids[groups == group])) == 1
    second = cross_validate_rrblup(y, z, n_folds=4, groups=groups, seed=2, lambda_=1)
    explicit = cross_validate_rrblup(y, z, fold_ids=first.fold_ids + 10, lambda_=1)
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_array_equal(first.predictions, explicit.predictions)
    assert not first.predictions.flags.writeable


def test_missing_markers_use_training_means_and_preserve_input() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=(12, 4))
    z[::2, 0] = np.nan
    z[4:, 1] = np.nan
    y = rng.normal(size=12)
    original = z.copy()
    first = cross_validate_rrblup(y, z, n_folds=3, lambda_=1)
    changed = z.copy()
    changed[:4, 1] = 1e8
    second = cross_validate_rrblup(y, changed, n_folds=3, lambda_=1)
    np.testing.assert_allclose(first.predictions[:4], second.predictions[:4])
    np.testing.assert_array_equal(original, z)
    assert np.all(np.isfinite(first.predictions))


@pytest.mark.parametrize("folds", [0, 1, 13, True])
def test_invalid_fold_count_is_rejected(folds: int) -> None:
    with pytest.raises(ValueError, match="n_folds"):
        cross_validate_rrblup(np.arange(12.0), np.ones((12, 2)), n_folds=folds)


def test_constant_phenotype_has_undefined_correlation() -> None:
    result = cross_validate_rrblup(np.ones(12), np.ones((12, 4)), lambda_=1, n_folds=3)
    assert np.isnan(result.pearson_r)
    assert result.rmse == 0


@pytest.mark.parametrize("penalty", [0.0, -1.0, np.nan, np.inf])
def test_invalid_penalty_is_rejected(penalty: float) -> None:
    with pytest.raises(ValueError, match="lambda_"):
        cross_validate_rrblup(np.arange(12.0), np.ones((12, 2)), lambda_=penalty)


def test_invalid_custom_splits_are_rejected() -> None:
    y, z = np.arange(12.0), np.ones((12, 2))
    for ids in (
        np.zeros(12, dtype=int),
        np.r_[np.zeros(11, dtype=int), 1],
        np.zeros(12),
    ):
        with pytest.raises(ValueError):
            cross_validate_rrblup(y, z, fold_ids=ids)
    with pytest.raises(ValueError, match="cannot be combined"):
        cross_validate_rrblup(y, z, fold_ids=np.arange(12), seed=1)
    with pytest.raises(ValueError, match="distinct groups"):
        cross_validate_rrblup(y, z, groups=np.zeros(12, dtype=int))


def test_missing_phenotypes_and_invalid_kinship_are_rejected() -> None:
    y = np.arange(12.0)
    y[0] = np.nan
    with pytest.raises(ValueError, match="finite observations"):
        cross_validate_gblup(y, np.eye(12))
    k = np.eye(12)
    k[0, 1] = 3
    with pytest.raises(ValueError, match="symmetric"):
        cross_validate_gblup(np.arange(12.0), k)


def test_seeded_folds_ignore_global_rng_and_singleton_scores_are_nan() -> None:
    rng = np.random.default_rng(20)
    y, z = rng.normal(size=9), rng.normal(size=(9, 4))
    first = cross_validate_rrblup(y, z, n_folds=9, seed=77, lambda_=1)
    np.random.seed(200)
    second = cross_validate_rrblup(y, z, n_folds=9, seed=77, lambda_=1)
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    np.testing.assert_array_equal(first.predictions, second.predictions)
    assert np.all(np.isnan(first.fold_pearson_r))
