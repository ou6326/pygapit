"""Shared genotype quality-control primitives."""

from __future__ import annotations

import numpy as np

from .._resources import (
    iter_marker_slices,
    validate_marker_workspace_mib,
)
from .._typing import FloatMatrix, FloatVector, IntVector
from ._genotype_store import GenotypeStore, GenotypeView


def filter_markers_by_maf(
    genotype: FloatMatrix | GenotypeStore,
    threshold: float,
    *,
    marker_workspace_mib: float,
    require_finite: bool = False,
) -> tuple[FloatMatrix | GenotypeView, IntVector, FloatVector]:
    """Filter markers and return retained MAF from the frequency pass."""
    marker_workspace_mib = validate_marker_workspace_mib(marker_workspace_mib)
    n, marker_count = genotype.shape
    if isinstance(genotype, GenotypeStore):
        frequencies = np.empty(marker_count, dtype=np.float64)
        for marker_slice in iter_marker_slices(
            n,
            marker_count,
            marker_workspace_mib,
        ):
            block = genotype.read_markers(marker_slice)
            if require_finite and not np.isfinite(block).all():
                raise ValueError(
                    "Disk-backed GAPIT requires finite, pre-imputed genotype values"
                )
            frequencies[marker_slice] = np.nansum(block, axis=0) / (2.0 * n)
    else:
        if require_finite and not np.isfinite(genotype).all():
            raise ValueError("Genotype values must be finite")
        frequencies = np.nansum(genotype, axis=0) / (2.0 * n)

    maf = np.minimum(frequencies, 1.0 - frequencies)
    keep = maf >= threshold
    kept_indices: IntVector = np.flatnonzero(keep)
    if isinstance(genotype, GenotypeStore):
        filtered: FloatMatrix | GenotypeView = GenotypeView(
            genotype,
            marker_indices=kept_indices,
        )
    else:
        filtered = genotype[:, keep]
    return filtered, kept_indices, maf[keep]
