"""Backend-independent preparation for GWAS visualizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import (
    FloatVector,
    LabelVector,
    NumericVector,
    StrVector,
    as_float_vector,
    as_str_vector,
    require_length,
)


@dataclass(frozen=True, slots=True)
class ManhattanPlotData:
    """Validated coordinates shared by static and interactive renderers.

    Input vectors are reused when their dtype already matches the plotting
    contract. Renderers must treat every array as read-only.
    """

    snp_names: StrVector
    chromosomes: StrVector
    positions: FloatVector
    p_values: FloatVector
    x_values: FloatVector
    log_p_values: FloatVector
    chromosome_labels: tuple[str, ...]
    chromosome_centers: FloatVector
    significance_threshold: float
    suggestive_threshold: float


def prepare_genomic_axis(
    chromosomes: LabelVector,
    positions: NumericVector,
    chromosome_gap: float = 5_000_000.0,
) -> tuple[FloatVector, tuple[str, ...], FloatVector]:
    """Map chromosome-local positions onto one cumulative genomic axis."""
    chroms = as_str_vector(chromosomes, name="chromosomes")
    pos = as_float_vector(positions, name="positions")
    require_length(pos, len(chroms), name="positions")
    if len(pos) == 0:
        raise ValueError("chromosomes and positions must not be empty")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must contain only finite values")
    if not np.isfinite(chromosome_gap) or chromosome_gap < 0.0:
        raise ValueError("chromosome_gap must be finite and non-negative")

    chromosome_labels = tuple(dict.fromkeys(chroms.tolist()))
    x_values = np.empty(len(pos), dtype=np.float64)
    chromosome_centers = np.empty(len(chromosome_labels), dtype=np.float64)
    cumulative = 0.0
    for index, chromosome in enumerate(chromosome_labels):
        mask = chroms == chromosome
        chromosome_positions = pos[mask]
        minimum = np.min(chromosome_positions)
        maximum = np.max(chromosome_positions)
        span = maximum - minimum
        x_values[mask] = cumulative + chromosome_positions - minimum
        chromosome_centers[index] = cumulative + span / 2.0
        cumulative += span + chromosome_gap
    return x_values, chromosome_labels, chromosome_centers


def prepare_manhattan_data(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    *,
    significance_threshold: float | None = None,
    suggestive_threshold: float | None = None,
    chromosome_gap: float = 5_000_000.0,
) -> ManhattanPlotData:
    """Validate and transform Manhattan inputs once for every renderer."""
    names = as_str_vector(snp_names, name="SNP names")
    chroms = as_str_vector(chromosomes, name="chromosomes")
    pos = as_float_vector(positions, name="positions")
    probabilities = as_float_vector(p_values, name="p-values")
    marker_count = len(probabilities)
    if marker_count == 0:
        raise ValueError("Manhattan data must contain at least one marker")
    for values, name in (
        (names, "SNP names"),
        (chroms, "chromosomes"),
        (pos, "positions"),
    ):
        require_length(values, marker_count, name=name)

    significance = (
        0.05 / marker_count
        if significance_threshold is None
        else significance_threshold
    )
    suggestive = (
        1.0 / marker_count if suggestive_threshold is None else suggestive_threshold
    )
    for threshold, name in (
        (significance, "significance_threshold"),
        (suggestive, "suggestive_threshold"),
    ):
        if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")

    valid = np.isfinite(probabilities) & (probabilities > 0.0) & (probabilities <= 1.0)
    log_p_values = np.zeros(marker_count, dtype=np.float64)
    np.log10(probabilities, out=log_p_values, where=valid)
    log_p_values *= -1.0
    x_values, chromosome_labels, chromosome_centers = prepare_genomic_axis(
        chroms,
        pos,
        chromosome_gap,
    )
    return ManhattanPlotData(
        snp_names=names,
        chromosomes=chroms,
        positions=pos,
        p_values=probabilities,
        x_values=x_values,
        log_p_values=log_p_values,
        chromosome_labels=chromosome_labels,
        chromosome_centers=chromosome_centers,
        significance_threshold=significance,
        suggestive_threshold=suggestive,
    )
