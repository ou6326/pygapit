"""Stage-level profiler for streaming numeric genotype imports.

This is a manual companion to ``benchmark_large_scale.py``.  It profiles the
dependency-free NumPy backend, which exposes the direct sample-row writes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarks.benchmark_large_scale import _MARKER_SCALES, _write_numeric_source
from benchmarks.run_baseline import _benchmark
from pygapit._resources import DEFAULT_MARKER_WORKSPACE_MIB
from pygapit._typing import FloatMatrix, FloatVector, StrVector
from pygapit.io._storage_numpy import write_numpy_genotype
from pygapit.io.formats import _NumericFileWriteSource

if TYPE_CHECKING:
    from pygapit._typing import Slice


@dataclass(frozen=True, slots=True)
class _TimedSampleSource:
    """Delegate a numeric source while accounting for parser/imputer time."""

    source: _NumericFileWriteSource
    text_parse_impute_seconds: list[float]

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.shape

    @property
    def taxa(self) -> StrVector:
        return self.source.taxa

    @property
    def marker_ids(self) -> StrVector:
        return self.source.marker_ids

    @property
    def chromosomes(self) -> StrVector:
        return self.source.chromosomes

    @property
    def positions(self) -> FloatVector:
        return self.source.positions

    def iter_sample_blocks(self) -> Iterator[tuple[Slice, FloatMatrix]]:
        blocks = self.source.iter_sample_blocks()
        while True:
            started = time.perf_counter()
            try:
                item = next(blocks)
            except StopIteration:
                return
            self.text_parse_impute_seconds.append(time.perf_counter() - started)
            yield item


def _profile_one_import(
    store_path: Path,
    genotype_path: Path,
    marker_path: Path,
    *,
    marker_workspace_mib: float,
) -> dict[str, float | int]:
    """Run the actual NumPy writer and separate source work from writer work."""
    marker_map_started = time.perf_counter()
    marker_map = pd.read_csv(marker_path, sep="\t", header=0)
    marker_map_read_seconds = time.perf_counter() - marker_map_started
    source_started = time.perf_counter()
    source = _NumericFileWriteSource(
        genotype_path, marker_map, "middle", marker_workspace_mib
    )
    source_init_seconds = time.perf_counter() - source_started
    parser_seconds: list[float] = []
    timed_source = _TimedSampleSource(source, parser_seconds)
    write_started = time.perf_counter()
    write_numpy_genotype(store_path, timed_source, marker_chunk_size=4_096)
    write_total_seconds = time.perf_counter() - write_started
    text_parse_impute_seconds = sum(parser_seconds)
    return {
        "marker_map_read_seconds": marker_map_read_seconds,
        "source_init_seconds": source_init_seconds,
        "text_parse_impute_seconds": text_parse_impute_seconds,
        "store_matrix_metadata_seconds": write_total_seconds
        - text_parse_impute_seconds,
        "total_seconds": (
            marker_map_read_seconds + source_init_seconds + write_total_seconds
        ),
        "sample_block_count": len(parser_seconds),
        # SampleBlockWriteSource maps source row slices directly to destination rows.
        "sample_row_to_marker_block_reorder_seconds": 0.0,
        "temporary_reorder_bytes": 0,
    }


def run_numeric_import_profile(
    *,
    n_individuals: int,
    n_markers: int,
    seed: int,
    warmups: int,
    repeats: int,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
) -> dict[str, object]:
    """Profile setup, text conversion/imputation, and NumPy store persistence."""
    if n_markers not in _MARKER_SCALES:
        raise ValueError(f"markers must be one of {_MARKER_SCALES}")
    if n_individuals < 1:
        raise ValueError("individuals must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    with tempfile.TemporaryDirectory(
        prefix="pygapit-numeric-import-profile-"
    ) as temp_dir:
        directory = Path(temp_dir)
        genotype_path, marker_path, _ = _write_numeric_source(
            directory,
            n_individuals=n_individuals,
            n_markers=n_markers,
            seed=seed,
        )
        store_path = directory / "numeric-store"

        def profile() -> dict[str, float | int]:
            return _profile_one_import(
                store_path,
                genotype_path,
                marker_path,
                marker_workspace_mib=marker_workspace_mib,
            )

        def cleanup() -> None:
            shutil.rmtree(store_path, ignore_errors=True)

        total = _benchmark(
            "streaming_numeric_import",
            profile,
            warmups=warmups,
            repeats=repeats,
            cleanup=cleanup,
        )
        samples: list[dict[str, float | int]] = []
        for _ in range(repeats):
            samples.append(profile())
            cleanup()

    def median(name: str) -> float:
        return float(np.median([float(sample[name]) for sample in samples]))

    return {
        "workload": {
            "individuals": n_individuals,
            "markers": n_markers,
            "seed": seed,
            "warmups": warmups,
            "repeats": repeats,
            "marker_workspace_mib": marker_workspace_mib,
            "store_backend": "numpy",
        },
        "total": asdict(total),
        "stages": {
            "marker_map_read_seconds": median("marker_map_read_seconds"),
            "numeric_source_init_seconds": median("source_init_seconds"),
            "text_parse_and_imputation_seconds": median("text_parse_impute_seconds"),
            "store_matrix_and_metadata_seconds": median(
                "store_matrix_metadata_seconds"
            ),
            "sample_row_to_marker_block_reorder_seconds": 0.0,
            "temporary_reorder_bytes": 0,
            "sample_block_count": int(median("sample_block_count")),
        },
        "note": (
            "SampleBlockWriteSource writes each source row range directly into the "
            "memmapped matrix, so no transpose or temporary file is used. "
            "traced_peak_mib is an end-to-end tracemalloc high-water mark; stage "
            "figures are wall-clock attribution within the same import."
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
        "--marker-workspace-mib", type=float, default=DEFAULT_MARKER_WORKSPACE_MIB
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(
        run_numeric_import_profile(
            n_individuals=args.individuals,
            n_markers=args.markers,
            seed=args.seed,
            warmups=args.warmups,
            repeats=args.repeats,
            marker_workspace_mib=args.marker_workspace_mib,
        ),
        indent=2,
    )
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
