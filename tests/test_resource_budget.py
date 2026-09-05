"""Contracts for shared numerical workspace budgets."""

import typing as t

import numpy as np
import pytest

from pygapit._resources import marker_batch_size, validate_marker_workspace_mib


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
