"""Manual large-marker benchmark for streaming storage and visualization paths.

The default 100k-marker workload is deliberately sized for a developer
machine.  One- and ten-million-marker runs are opt-in comparisons, not CI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.benchmark_plot_preparation import run_plot_preparation_benchmark
from benchmarks.run_baseline import _benchmark, _measure_scenario_peak_rss
from pygapit._resources import DEFAULT_MARKER_WORKSPACE_MIB
from pygapit.gwas.glm import glm_gwas
from pygapit.io.formats import import_numeric_genotype_store
from pygapit.io.storage import open_genotype_store
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import compute_pca

_MARKER_SCALES = (100_000, 1_000_000, 10_000_000)


def _write_numeric_source(
    directory: Path,
    *,
    n_individuals: int,
    n_markers: int,
    seed: int,
) -> tuple[Path, Path, np.ndarray]:
    """Write deterministic GD/GM files in bounded blocks before import timing."""
    genotype_path = directory / "genotype.tsv"
    marker_path = directory / "markers.tsv"
    chunk_size = 4_096
    with genotype_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("Taxa")
        for start in range(0, n_markers, chunk_size):
            stop = min(start + chunk_size, n_markers)
            stream.write("\t")
            stream.write("\t".join(f"SNP{i:08d}" for i in range(start, stop)))
        stream.write("\n")
        for sample in range(n_individuals):
            rng = np.random.default_rng(np.random.SeedSequence((seed, sample)))
            stream.write(f"T{sample:05d}")
            for start in range(0, n_markers, chunk_size):
                stop = min(start + chunk_size, n_markers)
                calls = rng.binomial(2, 0.3, size=stop - start)
                stream.write("\t")
                stream.write("\t".join(map(str, calls.tolist())))
            stream.write("\n")
    with marker_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("SNP\tChromosome\tPosition\n")
        for start in range(0, n_markers, chunk_size):
            stop = min(start + chunk_size, n_markers)
            stream.writelines(
                f"SNP{i:08d}\t{i % 20 + 1}\t{(i // 20 + 1) * 10000}\n"
                for i in range(start, stop)
            )
    phenotype = np.random.default_rng(seed).normal(size=n_individuals)
    return genotype_path, marker_path, phenotype


def run_large_scale_benchmark(
    *,
    n_individuals: int,
    n_markers: int,
    seed: int,
    warmups: int,
    repeats: int,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
    render_backend: str = "bokeh",
) -> dict[str, object]:
    """Run the representative store, scan, and Manhattan stages."""
    if n_markers not in _MARKER_SCALES:
        raise ValueError(f"markers must be one of {_MARKER_SCALES}")
    if n_individuals < 10:
        raise ValueError("individuals must be at least 10")

    def run_scenario() -> tuple[list[object], dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="pygapit-large-scale-") as temp_dir:
            directory = Path(temp_dir)
            genotype_path, marker_path, phenotype = _write_numeric_source(
                directory,
                n_individuals=n_individuals,
                n_markers=n_markers,
                seed=seed,
            )
            benchmark_store_path = directory / "import-benchmark"

            def import_store() -> None:
                import_numeric_genotype_store(
                    benchmark_store_path,
                    genotype_path,
                    marker_path,
                    backend="numpy",
                    marker_chunk_size=4_096,
                    marker_workspace_mib=marker_workspace_mib,
                )

            def cleanup_import_store() -> None:
                shutil.rmtree(benchmark_store_path, ignore_errors=True)

            primary_store_path = directory / "primary-store"
            import_numeric_genotype_store(
                primary_store_path,
                genotype_path,
                marker_path,
                backend="numpy",
                marker_chunk_size=4_096,
                marker_workspace_mib=marker_workspace_mib,
            )
            with open_genotype_store(primary_store_path, backend="numpy") as store:
                design = np.ones((n_individuals, 1), dtype=np.float64)
                numerical_measurements = [
                    _benchmark(
                        "streaming_numeric_import",
                        import_store,
                        warmups=warmups,
                        repeats=repeats,
                        cleanup=cleanup_import_store,
                    ),
                    _benchmark(
                        "vanraden_kinship_store",
                        lambda: vanraden_kinship(
                            store,
                            marker_workspace_mib=marker_workspace_mib,
                        ),
                        warmups=warmups,
                        repeats=repeats,
                    ),
                    _benchmark(
                        "pca_store",
                        lambda: compute_pca(
                            store,
                            n_components=3,
                            marker_workspace_mib=marker_workspace_mib,
                        ),
                        warmups=warmups,
                        repeats=repeats,
                    ),
                    _benchmark(
                        "glm_marker_scan_store",
                        lambda: glm_gwas(
                            phenotype,
                            design,
                            store,
                            marker_workspace_mib=marker_workspace_mib,
                        ),
                        warmups=warmups,
                        repeats=repeats,
                    ),
                ]

            plot_report = run_plot_preparation_benchmark(
                n_markers=n_markers,
                n_chromosomes=20,
                warmups=warmups,
                repeats=repeats,
                render_backends=(render_backend,),
                include_points=n_markers == 100_000,
            )
            return numerical_measurements, plot_report

    scenario_result, scenario_peak_rss_mib, scenario_rss_source = (
        _measure_scenario_peak_rss(run_scenario)
    )
    numerical_measurements, plot_report = scenario_result

    return {
        "workload": {
            "individuals": n_individuals,
            "markers": n_markers,
            "seed": seed,
            "warmups": warmups,
            "repeats": repeats,
            "marker_workspace_mib": marker_workspace_mib,
            "store_backend": "numpy",
            "plot_render_backend": render_backend,
            "include_exact_manhattan_points": n_markers == 100_000,
        },
        "numerical_measurements": [asdict(item) for item in numerical_measurements],
        "manhattan": plot_report,
        "scenario_process_peak_rss_mib": scenario_peak_rss_mib,
        "scenario_process_rss_source": scenario_rss_source,
        "memory_note": (
            "traced_peak_mib is per-stage tracemalloc. scenario_process_peak_rss_mib "
            "covers the complete import, numerical, and rendering scenario; it is "
            "not attributed to an individual stage. On Windows it samples the "
            "working set, while POSIX uses resource.ru_maxrss."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", type=int, default=64)
    parser.add_argument("--markers", type=int, choices=_MARKER_SCALES, default=100_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--marker-workspace-mib",
        type=float,
        default=DEFAULT_MARKER_WORKSPACE_MIB,
    )
    parser.add_argument(
        "--render-backend",
        choices=("matplotlib", "bokeh", "plotly"),
        default="bokeh",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_large_scale_benchmark(
        n_individuals=args.individuals,
        n_markers=args.markers,
        seed=args.seed,
        warmups=args.warmups,
        repeats=args.repeats,
        marker_workspace_mib=args.marker_workspace_mib,
        render_backend=args.render_backend,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
