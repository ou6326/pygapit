"""Regression tests for the batched EMMAX/P3D marker scan."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from pygapit._typing import FloatMatrix
from pygapit.gwas.mlm import MLMResult, mlm_gwas
from pygapit.stats import emma


class _RecordingGenotypeStore:
    """Store double that rejects whole-matrix conversion and records reads."""

    def __init__(self, genotype: npt.NDArray[np.float64]) -> None:
        self._genotype: npt.NDArray[np.float64] = genotype
        self.marker_slices: list[slice] = []

    @property
    def shape(self) -> tuple[int, int]:
        return self._genotype.shape

    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[np.float64]:
        raise AssertionError("EMMAX must not materialize a genotype store")

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: npt.NDArray[np.int_] | slice | None = None,
    ) -> npt.NDArray[np.float64]:
        self.marker_slices.append(marker_slice)
        block = self._genotype[:, marker_slice]
        if sample_indices is not None:
            block = block[sample_indices]
        result = np.asarray(block, dtype=np.float64).copy()
        result.setflags(write=False)
        return result


def _assert_gwas_results_equal(
    actual: emma.GWASResult | MLMResult,
    expected: emma.GWASResult | MLMResult,
) -> None:
    for field in ("p_values", "effects", "se", "stats"):
        np.testing.assert_allclose(
            getattr(actual, field),
            getattr(expected, field),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


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
    n_individuals = 1_000
    workspace_mib = 8.0
    batch_size = emma._emmax_marker_batch_size(n_individuals, workspace_mib)

    assert batch_size >= 1
    assert batch_size <= emma._EMMAX_MARKER_BATCH_SIZE
    assert batch_size * n_individuals * 8 <= workspace_mib * 1024**2


def test_emmax_marker_workspace_preserves_results() -> None:
    rng = np.random.default_rng(20260905)
    genotype = rng.binomial(2, 0.35, size=(30, 50)).astype(np.float64)
    design = np.ones((30, 1), dtype=np.float64)
    phenotype = genotype[:, :3] @ np.array([0.7, -0.4, 0.2])
    phenotype += rng.normal(0.0, 0.5, size=30)
    kinship = np.eye(30, dtype=np.float64)

    expected = emma.emmax_p3d(phenotype, design, genotype, kinship)
    actual = emma.emmax_p3d(
        phenotype,
        design,
        genotype,
        kinship,
        marker_workspace_mib=0.001,
    )

    for actual_values, expected_values in (
        (actual.p_values, expected.p_values),
        (actual.effects, expected.effects),
        (actual.se, expected.se),
        (actual.stats, expected.stats),
    ):
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize("snp_impute", ["middle", "major", "minor", "mean", "none"])
def test_emmax_reads_store_batches_with_missing_value_semantics(
    snp_impute: str,
) -> None:
    rng = np.random.default_rng(20260907)
    n, m = 30, 20
    genotype = rng.binomial(2, 0.35, size=(n, m)).astype(np.float64)
    genotype[::5, 1] = np.nan
    genotype[1::6, 7] = np.nan
    phenotype = rng.normal(size=n)
    design: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    kinship = np.eye(n, dtype=np.float64)
    workspace_mib = 0.001
    store = _RecordingGenotypeStore(genotype)
    batch_size = emma._emmax_marker_batch_size(n, workspace_mib)

    assert batch_size == 4
    expected = emma.emmax_p3d(
        phenotype,
        design,
        genotype,
        kinship,
        snp_impute=snp_impute,
        marker_workspace_mib=workspace_mib,
    )
    actual = emma.emmax_p3d(
        phenotype,
        design,
        store,
        kinship,
        snp_impute=snp_impute,
        marker_workspace_mib=workspace_mib,
    )

    _assert_gwas_results_equal(actual, expected)
    assert store.marker_slices == [
        slice(start, min(start + batch_size, m)) for start in range(0, m, batch_size)
    ]


def test_mlm_reads_store_batches_with_spectral_whitening() -> None:
    rng = np.random.default_rng(20260907)
    n, m = 30, 20
    genotype = rng.binomial(2, 0.35, size=(n, m)).astype(np.float64)
    phenotype = rng.normal(size=n)
    design: FloatMatrix = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    kinship = np.eye(n, dtype=np.float64)
    workspace_mib = 0.001
    store = _RecordingGenotypeStore(genotype)
    batch_size = emma._emmax_marker_batch_size(n, workspace_mib)

    with patch.object(
        np.linalg,
        "cholesky",
        side_effect=np.linalg.LinAlgError("forced fallback"),
    ):
        expected = mlm_gwas(
            phenotype,
            design,
            genotype,
            kinship,
            marker_workspace_mib=workspace_mib,
        )
    with patch.object(
        np.linalg,
        "cholesky",
        side_effect=np.linalg.LinAlgError("forced fallback"),
    ):
        actual = mlm_gwas(
            phenotype,
            design,
            store,
            kinship,
            marker_workspace_mib=workspace_mib,
        )

    _assert_gwas_results_equal(actual, expected)
    assert store.marker_slices == [
        slice(start, min(start + batch_size, m)) for start in range(0, m, batch_size)
    ]


def test_emmax_validates_genotype_store_row_count() -> None:
    phenotype = np.arange(30.0)
    design = np.ones((30, 1), dtype=np.float64)
    store = _RecordingGenotypeStore(np.ones((29, 2), dtype=np.float64))
    kinship = np.eye(30, dtype=np.float64)

    with pytest.raises(ValueError, match="genotype matrix must have 30 rows"):
        emma.emmax_p3d(phenotype, design, store, kinship)


def test_mlm_validates_genotype_store_row_count() -> None:
    phenotype = np.arange(30.0)
    design = np.ones((30, 1), dtype=np.float64)
    store = _RecordingGenotypeStore(np.ones((29, 2), dtype=np.float64))
    kinship = np.eye(30, dtype=np.float64)

    with pytest.raises(ValueError, match="genotype matrix must have 30 rows"):
        mlm_gwas(phenotype, design, store, kinship)
