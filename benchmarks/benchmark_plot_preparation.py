"""Benchmark Manhattan data preparation, object construction, and rendering."""

from __future__ import annotations

import argparse
import json
import typing as t

import holoviews as hv
import numpy as np
from holoviews.core import Dimensioned

from benchmarks.run_baseline import _benchmark
from pygapit._typing import FloatVector, StrVector
from pygapit.visualization.data import (
    ManhattanPlotData,
    prepare_genomic_axis,
    prepare_manhattan_data,
)
from pygapit.visualization.plots import (
    _build_manhattan_plot,
    _register_holoviews_backends,
)

RendererBackend = t.Literal["matplotlib", "bokeh", "plotly"]


class PlotPreparationWorkload(t.TypedDict):
    markers: int
    chromosomes: int
    warmups: int
    repeats: int
    render_backends: list[RendererBackend]


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
    p_values[0] = 0.01 / n_markers
    return snp_names, chromosomes, positions, p_values


def _build_plot(data: ManhattanPlotData, *, aggregate: bool) -> Dimensioned:
    return t.cast(
        Dimensioned,
        _build_manhattan_plot(
            data,
            aggregate=aggregate,
            title="Manhattan benchmark",
            highlight_snps=None,
            effects=None,
            maf=None,
            figsize=(14.0, 5.0),
            point_size=1.5,
        ),
    )


def _render_clone(plot: Dimensioned, backend: RendererBackend) -> None:
    clone = t.cast(Dimensioned, plot.clone(shared_data=True, link=False))
    rendered = t.cast(object, hv.render(clone, backend=backend))
    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        plt.close(t.cast(Figure, rendered))


def run_plot_preparation_benchmark(
    *,
    n_markers: int,
    n_chromosomes: int,
    warmups: int,
    repeats: int,
    render_backends: tuple[RendererBackend, ...] = (),
) -> PlotPreparationReport:
    """Measure preparation and construction, plus requested renderer stages."""
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    snp_names, chromosomes, positions, p_values = _make_plot_inputs(
        n_markers,
        n_chromosomes,
    )
    prepared = prepare_manhattan_data(
        snp_names,
        chromosomes,
        positions,
        p_values,
    )
    _register_holoviews_backends()
    measurements = [
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
        _benchmark(
            "manhattan_points_object",
            lambda: _build_plot(prepared, aggregate=False),
            warmups=warmups,
            repeats=repeats,
        ),
        _benchmark(
            "manhattan_aggregate_object",
            lambda: _build_plot(prepared, aggregate=True),
            warmups=warmups,
            repeats=repeats,
        ),
    ]
    if render_backends:
        point_plot = _build_plot(prepared, aggregate=False)
        aggregate_plot = _build_plot(prepared, aggregate=True)
        for backend in render_backends:
            measurements.extend([
                _benchmark(
                    f"manhattan_points_render_{backend}",
                    lambda backend=backend: _render_clone(point_plot, backend),
                    warmups=warmups,
                    repeats=repeats,
                ),
                _benchmark(
                    f"manhattan_aggregate_render_{backend}",
                    lambda backend=backend: _render_clone(aggregate_plot, backend),
                    warmups=warmups,
                    repeats=repeats,
                ),
            ])
    return {
        "workload": {
            "markers": n_markers,
            "chromosomes": n_chromosomes,
            "warmups": warmups,
            "repeats": repeats,
            "render_backends": list(render_backends),
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
            "operation; input arrays and renderer templates are allocated before "
            "tracing, and renderer measurements clone their template to avoid "
            "reusing a populated DynamicMap cache"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=int, default=1_000_000)
    parser.add_argument("--chromosomes", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--render-backend",
        action="append",
        choices=("matplotlib", "bokeh", "plotly"),
        default=[],
        help=(
            "Renderer to benchmark after construction; repeat the option to test "
            "multiple backends. Rendering is skipped by default."
        ),
    )
    args = parser.parse_args()

    report = run_plot_preparation_benchmark(
        n_markers=args.markers,
        n_chromosomes=args.chromosomes,
        warmups=args.warmups,
        repeats=args.repeats,
        render_backends=tuple(args.render_backend),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
