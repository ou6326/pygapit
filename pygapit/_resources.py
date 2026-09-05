"""Shared resource-budget validation for bounded numerical workspaces."""

from __future__ import annotations

from numbers import Real

import numpy as np

DEFAULT_MARKER_WORKSPACE_MIB = 32.0
MAX_MARKERS_PER_BATCH = 4096
_MIB = 1024**2
_FLOAT64_BYTES = np.dtype(np.float64).itemsize


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
    budget = validate_marker_workspace_mib(marker_workspace_mib)
    target_bytes = int(budget * _MIB)
    memory_limited_size = max(
        1,
        target_bytes // (n_individuals * _FLOAT64_BYTES),
    )
    return min(max_markers, memory_limited_size)
