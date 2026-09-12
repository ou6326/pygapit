"""Shared resource-budget validation for bounded numerical workspaces."""

from __future__ import annotations

import typing as t
from collections.abc import Iterator
from numbers import Real

import numpy as np

if t.TYPE_CHECKING:
    from ._typing import Slice

DEFAULT_MARKER_WORKSPACE_MIB = 32.0
MAX_MARKERS_PER_BATCH = 4096
_MIB = 1024**2
_FLOAT64_BYTES = np.dtype(np.float64).itemsize


def _float64_batch_size(
    fixed_axis_size: int,
    workspace_mib: float,
    *,
    max_items: int,
) -> int:
    budget = validate_marker_workspace_mib(workspace_mib)
    target_bytes = int(budget * _MIB)
    return min(
        max_items,
        max(1, target_bytes // (fixed_axis_size * _FLOAT64_BYTES)),
    )


def validate_marker_workspace_mib(value: float) -> float:
    """Return a finite positive marker-workspace budget in MiB."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("marker_workspace_mib must be a real number, not bool")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("marker_workspace_mib must be finite and positive")
    return normalized


def marker_batch_size(
    n_individuals: int,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
    *,
    max_markers: int = MAX_MARKERS_PER_BATCH,
) -> int:
    """Size one float64 sample-by-marker workspace within a target budget."""
    if isinstance(n_individuals, bool) or n_individuals < 1:
        raise ValueError("n_individuals must be positive")
    if isinstance(max_markers, bool) or max_markers < 1:
        raise ValueError("max_markers must be positive")
    return _float64_batch_size(
        n_individuals,
        marker_workspace_mib,
        max_items=max_markers,
    )


def sample_batch_size(
    n_markers: int,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
    *,
    max_samples: int = MAX_MARKERS_PER_BATCH,
) -> int:
    """Size one float64 sample-by-marker workspace within a target budget."""
    if isinstance(n_markers, bool) or n_markers < 1:
        raise ValueError("n_markers must be positive")
    if isinstance(max_samples, bool) or max_samples < 1:
        raise ValueError("max_samples must be positive")
    return _float64_batch_size(
        n_markers,
        marker_workspace_mib,
        max_items=max_samples,
    )


def iter_marker_slices(
    n_individuals: int,
    n_markers: int,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
    *,
    max_markers: int = MAX_MARKERS_PER_BATCH,
) -> Iterator[Slice]:
    """Yield contiguous marker slices sized for one float64 workspace."""
    if isinstance(n_markers, bool) or n_markers < 0:
        raise ValueError("n_markers must be non-negative")
    batch_size = marker_batch_size(
        n_individuals,
        marker_workspace_mib,
        max_markers=max_markers,
    )
    for start in range(0, n_markers, batch_size):
        yield slice(start, min(start + batch_size, n_markers))
