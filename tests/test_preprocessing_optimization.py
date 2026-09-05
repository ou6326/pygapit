"""Regression tests for genotype preprocessing optimizations."""

from __future__ import annotations

import numpy as np
import pytest

from pygapit.io.formats import impute_missing
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import compute_pca


@pytest.mark.parametrize("shape", [(40, 120), (120, 40)])
def test_gram_pca_matches_full_svd(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(20260905)
    genotype = rng.binomial(2, 0.35, size=shape).astype(np.float64)
    centered = genotype - genotype.mean(axis=0)
    left, singular_values, right_transpose = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    result = compute_pca(genotype, n_components=3, maf_filter=0.0)

    np.testing.assert_allclose(
        result.eigenvalues,
        singular_values[:3] ** 2 / (shape[0] - 1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.var_explained,
        result.eigenvalues / np.sum(singular_values**2 / (shape[0] - 1)),
        rtol=1e-12,
        atol=1e-12,
    )
    for component in range(3):
        sign = np.sign(result.scores[:, component] @ left[:, component]) or 1.0
        np.testing.assert_allclose(
            result.scores[:, component],
            sign * left[:, component] * singular_values[component],
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            result.loadings[:, component],
            sign * right_transpose[component],
            rtol=1e-11,
            atol=1e-11,
        )


def test_zero_component_pca_skips_decomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genotype = np.arange(60, dtype=np.float64).reshape(10, 6) % 3

    def unexpected_eigh(*args: object, **kwargs: object) -> None:
        raise AssertionError("zero-component PCA must not decompose a Gram matrix")

    monkeypatch.setattr("pygapit.stats.pca.scipy_eigh", unexpected_eigh)
    result = compute_pca(genotype, n_components=0, maf_filter=0.0)

    assert result.scores.shape == (10, 0)
    assert result.loadings.shape == (6, 0)
    assert result.eigenvalues.shape == (0,)


def test_mean_imputation_matches_column_means_and_all_missing_fallback() -> None:
    genotype = np.asarray([
        [0.0, np.nan, np.nan],
        [2.0, 1.0, np.nan],
        [np.nan, 2.0, np.nan],
    ])

    result = impute_missing(genotype, method="mean")

    expected = np.asarray([[0.0, 1.5, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    np.testing.assert_allclose(result, expected)
    assert np.isnan(genotype).any()


def test_batched_vanraden_matches_full_centered_crossproduct() -> None:
    rng = np.random.default_rng(20260905)
    genotype = rng.binomial(2, 0.35, size=(40, 257)).astype(np.float64)
    frequencies = genotype.mean(axis=0) / 2.0
    valid = (frequencies > 0.0) & (frequencies < 1.0)
    centered = genotype[:, valid] - 2.0 * frequencies[valid]
    adjustment = 2.0 * np.sum(frequencies[valid] * (1.0 - frequencies[valid]))
    expected = centered @ centered.T / adjustment

    actual = vanraden_kinship(genotype, marker_workspace_mib=0.001)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
