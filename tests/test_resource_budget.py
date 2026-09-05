"""Contracts for shared numerical workspace budgets."""

import typing as t

import numpy as np
import pytest

from pygapit._resources import marker_batch_size, validate_marker_workspace_mib
from pygapit.gwas.glm import glm_gwas
from pygapit.gwas.mlm import mlm_gwas
from pygapit.stats.emma import emmax_p3d


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
