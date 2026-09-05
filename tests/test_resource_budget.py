"""Contracts for shared numerical workspace budgets."""

import typing as t

import numpy as np
import pytest

from pygapit._resources import (
    iter_marker_slices,
    marker_batch_size,
    validate_marker_workspace_mib,
)
from pygapit.gwas.glm import glm_gwas
from pygapit.gwas.mlm import mlm_gwas
from pygapit.stats.emma import emmax_p3d
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import compute_pca


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_marker_workspace_requires_finite_positive_value(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        validate_marker_workspace_mib(value)


@pytest.mark.parametrize("value", [True, "32"])
def test_marker_workspace_rejects_non_real_values(value: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        validate_marker_workspace_mib(t.cast(float, value))


def test_marker_batch_size_respects_budget_and_cap() -> None:
    assert marker_batch_size(100, 1.0) == 1_310
    assert marker_batch_size(10, 32.0) == 4_096
    assert marker_batch_size(1_000_000, 1.0) == 1


def test_marker_slices_cover_each_marker_with_shared_batch_sizing() -> None:
    slices = list(iter_marker_slices(100, 3_000, 1.0))

    assert slices == [slice(0, 1_310), slice(1_310, 2_620), slice(2_620, 3_000)]
    assert list(iter_marker_slices(100, 0, 1.0)) == []


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan, True])
def test_direct_scan_boundaries_always_validate_workspace(value: object) -> None:
    phenotype = np.arange(8.0)
    design = np.ones((8, 1), dtype=np.float64)
    genotype = np.ones((8, 3), dtype=np.float64)
    kinship = np.eye(8, dtype=np.float64)
    invalid = t.cast(float, value)

    error = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(error, match="marker_workspace_mib"):
        glm_gwas(
            phenotype,
            design,
            genotype,
            marker_workspace_mib=invalid,
        )
    with pytest.raises(error, match="marker_workspace_mib"):
        mlm_gwas(
            phenotype,
            design,
            genotype,
            kinship,
            marker_workspace_mib=invalid,
        )
    with pytest.raises(error, match="marker_workspace_mib"):
        emmax_p3d(
            phenotype,
            design,
            genotype,
            kinship,
            marker_workspace_mib=invalid,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan, True])
def test_vanraden_always_validates_workspace(value: object) -> None:
    invalid = t.cast(float, value)
    error = TypeError if isinstance(value, bool) else ValueError

    with pytest.raises(error, match="marker_workspace_mib"):
        vanraden_kinship(
            np.ones((8, 3), dtype=np.float64),
            marker_workspace_mib=invalid,
        )
    with pytest.raises(error, match="marker_workspace_mib"):
        compute_pca(
            np.ones((8, 3), dtype=np.float64),
            marker_workspace_mib=invalid,
        )
