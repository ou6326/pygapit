"""Benchmark PCA I/O amplification for dense and sparse marker views."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from benchmarks.run_baseline import BenchmarkResult, _benchmark, _make_data
from pygapit._typing import FloatMatrix, IntVector
from pygapit.io.formats import GenotypeData
from pygapit.io.storage import (
    GenotypeStore,
    GenotypeView,
    HDF5GenotypeStore,
    MarkerChunkedGenotypeStore,
    NumpyGenotypeStore,
    write_hdf5_genotype,
    write_numpy_genotype,
)
from pygapit.stats.pca import PCAResult, compute_pca

BenchmarkBackend = Literal["numpy", "hdf5"]
RetentionPattern = Literal["dense", "interleaved"]


@dataclass(frozen=True, slots=True)
class PCAStoreIOMeasurement:
    backend: BenchmarkBackend
    pattern: RetentionPattern
    parent_markers: int
    retained_markers: int
    parent_read_calls: int
    parent_cells_read: int
    timing: BenchmarkResult


@dataclass(frozen=True, slots=True)
class PCAStoreIOReport:
    individuals: int
    parent_markers: int
    retained_markers: int
    components: int
    marker_workspace_mib: float
    seed: int
    warmups: int
    repeats: int
    measurements: tuple[PCAStoreIOMeasurement, ...]
    memory_note: str


class _CountingStore:
    def __init__(self, parent: GenotypeStore) -> None:
        self.parent = parent
        self.parent_read_calls = 0
        self.parent_cells_read = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.parent.shape

    @property
    def marker_chunk_size(self) -> int | None:
        if isinstance(self.parent, MarkerChunkedGenotypeStore):
            return self.parent.marker_chunk_size
        return None

    def reset(self) -> None:
        self.parent_read_calls = 0
        self.parent_cells_read = 0

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        rows, columns = self.shape
        selected_rows = (
            rows
            if sample_indices is None
            else len(range(*sample_indices.indices(rows)))
            if isinstance(sample_indices, slice)
            else len(sample_indices)
        )
        selected_columns = len(range(*marker_slice.indices(columns)))
        self.parent_read_calls += 1
        self.parent_cells_read += selected_rows * selected_columns
        return self.parent.read_markers(marker_slice, sample_indices)


def _marker_indices(
    pattern: RetentionPattern,
    parent_markers: int,
    retained_markers: int,
) -> IntVector:
    if pattern == "dense":
        return np.arange(retained_markers, dtype=np.int_)
    stride = parent_markers // retained_markers
    return np.arange(retained_markers, dtype=np.int_) * stride


def _assert_equivalent(reference: PCAResult, candidate: PCAResult) -> None:
    np.testing.assert_allclose(
        candidate.eigenvalues,
        reference.eigenvalues,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        candidate.var_explained,
        reference.var_explained,
        rtol=1e-10,
        atol=1e-12,
    )
    for component in range(reference.scores.shape[1]):
        alignment = candidate.scores[:, component] @ reference.scores[:, component]
        sign = np.sign(alignment) or 1.0
        np.testing.assert_allclose(
            candidate.scores[:, component],
            sign * reference.scores[:, component],
            rtol=1e-9,
            atol=1e-10,
        )


def _genotype_data(
    individuals: int,
    markers: int,
    seed: int,
) -> GenotypeData:
    genotype, _phenotype, chromosomes, positions = _make_data(
        individuals,
        markers,
        seed,
    )
    marker_ids = np.asarray([f"SNP{index:06d}" for index in range(markers)])
    marker_map = pd.DataFrame({
        "SNP": marker_ids,
        "Chromosome": chromosomes,
        "Position": positions,
    })
    taxa = np.asarray([f"T{index:05d}" for index in range(individuals)])
    return GenotypeData(genotype, marker_map, taxa)


def _measure_pattern(
    store: GenotypeStore,
    genotype: FloatMatrix,
    *,
    backend: BenchmarkBackend,
    pattern: RetentionPattern,
    retained_markers: int,
    components: int,
    marker_workspace_mib: float,
    warmups: int,
    repeats: int,
) -> PCAStoreIOMeasurement:
    marker_indices = _marker_indices(pattern, store.shape[1], retained_markers)
    counting_store = _CountingStore(store)
    view = GenotypeView(counting_store, marker_indices=marker_indices)
    reference = compute_pca(
        genotype[:, marker_indices],
        n_components=components,
        marker_workspace_mib=marker_workspace_mib,
    )

    def operation() -> PCAResult:
        counting_store.reset()
        return compute_pca(
            view,
            n_components=components,
            marker_workspace_mib=marker_workspace_mib,
        )

    _assert_equivalent(reference, operation())
    timing = _benchmark(
        f"{backend}_{pattern}",
        operation,
        warmups=warmups,
        repeats=repeats,
    )
    return PCAStoreIOMeasurement(
        backend=backend,
        pattern=pattern,
        parent_markers=store.shape[1],
        retained_markers=retained_markers,
        parent_read_calls=counting_store.parent_read_calls,
        parent_cells_read=counting_store.parent_cells_read,
        timing=timing,
    )


def run_pca_store_io_benchmark(
    *,
    individuals: int,
    parent_markers: int,
    retained_markers: int,
    components: int,
    marker_workspace_mib: float,
    seed: int,
    warmups: int,
    repeats: int,
    backends: tuple[BenchmarkBackend, ...] = ("numpy", "hdf5"),
) -> PCAStoreIOReport:
    """Measure how physical marker layout affects disk-backed PCA."""
    if individuals < 10 or parent_markers < 10:
        raise ValueError("individuals and parent markers must be at least 10")
    if not 0 < retained_markers <= parent_markers // 2:
        raise ValueError("retained markers must be at most half of parent markers")
    if components < 1 or components > retained_markers:
        raise ValueError("components must be between one and retained markers")

    genotype = _genotype_data(individuals, parent_markers, seed)
    measurements: list[PCAStoreIOMeasurement] = []
    with tempfile.TemporaryDirectory(prefix="pygapit-pca-store-io-") as temp_dir:
        root = Path(temp_dir)
        if "numpy" in backends:
            numpy_path = root / "numpy-store"
            write_numpy_genotype(numpy_path, genotype)
            with NumpyGenotypeStore(numpy_path) as store:
                for pattern in ("dense", "interleaved"):
                    measurements.append(
                        _measure_pattern(
                            store,
                            genotype.GD,
                            backend="numpy",
                            pattern=pattern,
                            retained_markers=retained_markers,
                            components=components,
                            marker_workspace_mib=marker_workspace_mib,
                            warmups=warmups,
                            repeats=repeats,
                        )
                    )
        if "hdf5" in backends and find_spec("h5py") is not None:
            hdf5_path = root / "genotype.h5"
            write_hdf5_genotype(hdf5_path, genotype)
            with HDF5GenotypeStore(hdf5_path) as store:
                for pattern in ("dense", "interleaved"):
                    measurements.append(
                        _measure_pattern(
                            store,
                            genotype.GD,
                            backend="hdf5",
                            pattern=pattern,
                            retained_markers=retained_markers,
                            components=components,
                            marker_workspace_mib=marker_workspace_mib,
                            warmups=warmups,
                            repeats=repeats,
                        )
                    )

    return PCAStoreIOReport(
        individuals=individuals,
        parent_markers=parent_markers,
        retained_markers=retained_markers,
        components=components,
        marker_workspace_mib=marker_workspace_mib,
        seed=seed,
        warmups=warmups,
        repeats=repeats,
        measurements=tuple(measurements),
        memory_note=(
            "traced_peak_mib excludes the pre-existing store and may exclude native "
            "HDF5 and BLAS allocations"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", type=int, default=2_000)
    parser.add_argument("--parent-markers", type=int, default=2_000)
    parser.add_argument("--retained-markers", type=int, default=500)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--marker-workspace-mib", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_pca_store_io_benchmark(
        individuals=args.individuals,
        parent_markers=args.parent_markers,
        retained_markers=args.retained_markers,
        components=args.components,
        marker_workspace_mib=args.marker_workspace_mib,
        seed=args.seed,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    rendered = json.dumps(asdict(report), indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
