"""Manual large-marker benchmark for streaming storage and visualization paths.

The default 100k-marker workload is deliberately sized for a developer
machine.  One- and ten-million-marker runs are opt-in comparisons, not CI.
The NumPy store backend runs by default; select ``--store-backend hdf5`` or
``--store-backend zarr`` to spot-check the optional backends at the same scale.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Protocol, cast, final

import numpy as np

from benchmarks.benchmark_plot_preparation import (
    PlotPreparationReport,
    RendererBackend,
    run_plot_preparation_benchmark,
)
from benchmarks.run_baseline import BenchmarkResult, _benchmark
from pygapit._resources import DEFAULT_MARKER_WORKSPACE_MIB
from pygapit.gwas.glm import glm_gwas
from pygapit.io.formats import import_numeric_genotype_store
from pygapit.io.storage import open_genotype_store
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import compute_pca

_MARKER_SCALES = (100_000, 1_000_000, 10_000_000)

type _StoreBackend = Literal["numpy", "hdf5", "zarr"]
_STORE_BACKENDS: tuple[_StoreBackend, ...] = ("numpy", "hdf5", "zarr")
_STORE_SUFFIXES: dict[_StoreBackend, str] = {
    "numpy": "",
    "hdf5": ".h5",
    "zarr": ".zarr",
}

if sys.platform == "win32":

    @final
    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _KERNEL32 = ctypes.WinDLL("kernel32")
    _KERNEL32.GetCurrentProcess.restype = ctypes.c_void_p
    _CURRENT_PROCESS = _KERNEL32.GetCurrentProcess()
    _GET_PROCESS_MEMORY_INFO = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    _GET_PROCESS_MEMORY_INFO.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    _GET_PROCESS_MEMORY_INFO.restype = ctypes.c_int


class _ResourceUsage(Protocol):
    ru_maxrss: int


class _ResourceModule(Protocol):
    RUSAGE_SELF: int

    def getrusage(self, who: int) -> _ResourceUsage: ...


def _runtime_object(value: object) -> object:
    """Cross a platform-stub boundary without weakening the result type."""
    return value


def _process_rss_bytes() -> tuple[int, str]:
    """Return process RSS, without adding a benchmark-only dependency.

    ``resource.ru_maxrss`` is the OS high-water mark on POSIX.  Windows has no
    ``resource`` module, so query the current working set through the native
    process API; the caller samples it while an operation is active.
    """
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _GET_PROCESS_MEMORY_INFO(
            _CURRENT_PROCESS,
            ctypes.byref(counters),
            counters.cb,
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize), "windows_working_set_sample"

    import resource

    resource_module = cast(_ResourceModule, _runtime_object(resource))
    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
    # Linux reports KiB, while macOS reports bytes.
    rss = usage.ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return rss, "resource_ru_maxrss"


def _measure_scenario_peak_rss[ResultT](
    operation: Callable[[], ResultT],
) -> tuple[ResultT, float, str]:
    """Measure a whole-scenario process RSS peak separately from Python allocations."""
    gc.collect()
    initial_rss, source = _process_rss_bytes()
    peak_rss = initial_rss
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak_rss
        while not stop.wait(0.005):
            current_rss, _ = _process_rss_bytes()
            peak_rss = max(peak_rss, current_rss)

    sampler = threading.Thread(target=sample, name="pygapit-rss-sampler", daemon=True)
    sampler.start()
    try:
        result = operation()
    finally:
        stop.set()
        sampler.join()
    final_rss, _ = _process_rss_bytes()
    return result, max(peak_rss, final_rss) / (1024.0**2), source


def _remove_store(path: Path) -> None:
    """Remove a written store whether it is a directory or a single file."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


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
    store_backend: _StoreBackend = "numpy",
    render_backend: RendererBackend = "bokeh",
) -> dict[str, object]:
    """Run the representative store, scan, and Manhattan stages."""
    if n_markers not in _MARKER_SCALES:
        raise ValueError(f"markers must be one of {_MARKER_SCALES}")
    if n_individuals < 10:
        raise ValueError("individuals must be at least 10")
    suffix = _STORE_SUFFIXES[store_backend]

    def run_scenario() -> tuple[list[BenchmarkResult], PlotPreparationReport]:
        with tempfile.TemporaryDirectory(prefix="pygapit-large-scale-") as temp_dir:
            directory = Path(temp_dir)
            genotype_path, marker_path, phenotype = _write_numeric_source(
                directory,
                n_individuals=n_individuals,
                n_markers=n_markers,
                seed=seed,
            )
            benchmark_store_path = directory / f"import-benchmark{suffix}"

            def import_store() -> None:
                import_numeric_genotype_store(
                    benchmark_store_path,
                    genotype_path,
                    marker_path,
                    backend=store_backend,
                    marker_chunk_size=4_096,
                    marker_workspace_mib=marker_workspace_mib,
                )

            def cleanup_import_store() -> None:
                _remove_store(benchmark_store_path)

            primary_store_path = directory / f"primary-store{suffix}"
            import_numeric_genotype_store(
                primary_store_path,
                genotype_path,
                marker_path,
                backend=store_backend,
                marker_chunk_size=4_096,
                marker_workspace_mib=marker_workspace_mib,
            )
            with open_genotype_store(
                primary_store_path, backend=store_backend
            ) as store:
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
            "store_backend": store_backend,
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
        "--store-backend",
        choices=_STORE_BACKENDS,
        default="numpy",
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
        store_backend=cast(_StoreBackend, args.store_backend),
        render_backend=cast(RendererBackend, args.render_backend),
    )
    rendered = json.dumps(report, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
