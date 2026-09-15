"""Benchmark backend-independent Manhattan plot preparation."""

from __future__ import annotations

import argparse
import json
import typing as t

import numpy as np

from benchmarks.run_baseline import _benchmark
from pygapit._typing import FloatVector, StrVector
from pygapit.visualization.data import (
    prepare_genomic_axis,
    prepare_manhattan_data,
)


class PlotPreparationWorkload(t.TypedDict):
    markers: int
    chromosomes: int
    warmups: int
    repeats: int


class PlotPreparationMeasurement(t.TypedDict):
    name: str
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    traced_peak_mib: float


class PlotPreparationReport(t.TypedDict):
    workload: PlotPreparationWorkload
    measurements: list[PlotPreparationMeasurement]
    memory_note: str


def _make_plot_inputs(
    n_markers: int,
    n_chromosomes: int,
) -> tuple[StrVector, StrVector, FloatVector, FloatVector]:
    if n_markers < 1:
        raise ValueError("markers must be at least one")
    if not 1 <= n_chromosomes <= n_markers:
        raise ValueError("chromosomes must be between one and markers")

    counts = np.full(n_chromosomes, n_markers // n_chromosomes, dtype=np.int_)
    counts[: n_markers % n_chromosomes] += 1
    chromosome_codes = np.repeat(np.arange(n_chromosomes), counts)
    chromosomes = np.char.add("Chr", chromosome_codes.astype(str))
    chromosome_starts = np.repeat(np.cumsum(counts) - counts, counts)
    positions = np.arange(n_markers, dtype=np.float64) - chromosome_starts
    snp_names = np.full(n_markers, "SNP", dtype=str)
    p_values = np.linspace(1.0 / n_markers, 1.0, n_markers, dtype=np.float64)
    return snp_names, chromosomes, positions, p_values


def run_plot_preparation_benchmark(
    *,
    n_markers: int,
    n_chromosomes: int,
    warmups: int,
    repeats: int,
) -> PlotPreparationReport:
    """Measure genomic-axis and complete Manhattan data preparation separately."""
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    snp_names, chromosomes, positions, p_values = _make_plot_inputs(
        n_markers,
        n_chromosomes,
    )
    measurements = (
        _benchmark(
            "genomic_axis",
            lambda: prepare_genomic_axis(chromosomes, positions),
            warmups=warmups,
            repeats=repeats,
        ),
        _benchmark(
            "manhattan_data",
            lambda: prepare_manhattan_data(
                snp_names,
                chromosomes,
                positions,
                p_values,
            ),
            warmups=warmups,
            repeats=repeats,
        ),
    )
    return {
        "workload": {
            "markers": n_markers,
            "chromosomes": n_chromosomes,
            "warmups": warmups,
            "repeats": repeats,
        },
        "measurements": [
            {
                "name": measurement.name,
                "median_seconds": measurement.median_seconds,
                "minimum_seconds": measurement.minimum_seconds,
                "maximum_seconds": measurement.maximum_seconds,
                "traced_peak_mib": measurement.traced_peak_mib,
            }
            for measurement in measurements
        ],
        "memory_note": (
            "traced_peak_mib measures Python and NumPy allocations during each "
            "operation; input arrays are allocated before tracing"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=int, default=1_000_000)
    parser.add_argument("--chromosomes", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    report = run_plot_preparation_benchmark(
        n_markers=args.markers,
        n_chromosomes=args.chromosomes,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
