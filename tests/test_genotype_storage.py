"""Contracts for chunk-readable in-memory and optional disk-backed genotypes."""

from __future__ import annotations

import builtins
import json
import warnings
from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd
import pytest

from pygapit._typing import FloatMatrix, FloatVector, IntVector, StrVector
from pygapit.io.formats import (
    GenotypeData,
    import_hapmap_genotype_store,
    import_numeric_genotype_store,
    maf_filter,
    read_hapmap,
    read_numeric,
)
from pygapit.io.storage import (
    ArrayGenotypeStore,
    GenotypeView,
    HDF5GenotypeStore,
    LabeledGenotypeStore,
    NumpyGenotypeStore,
    StorageBackend,
    ZarrGenotypeStore,
    open_genotype_store,
    write_genotype_store,
    write_hdf5_genotype,
    write_numpy_genotype,
    write_zarr_genotype,
)
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import compute_pca

_STORAGE_BACKENDS = [
    pytest.param("numpy", id="numpy"),
    pytest.param(
        "hdf5",
        id="hdf5",
        marks=pytest.mark.skipif(
            find_spec("h5py") is None,
            reason="h5py is not installed",
        ),
    ),
    pytest.param(
        "zarr",
        id="zarr",
        marks=pytest.mark.skipif(
            find_spec("zarr") is None,
            reason="zarr is not installed",
        ),
    ),
]


class _ReadNumericCsv(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        sep: str,
        nrows: int | None = None,
        usecols: list[str] | None = None,
        chunksize: int | None = None,
        low_memory: bool = False,
    ) -> pd.DataFrame | Iterator[pd.DataFrame]: ...


class _NoMaterializationStore:
    def __init__(self, values: FloatMatrix) -> None:
        self._values: FloatMatrix = values
        self.read_count: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> FloatMatrix:
        raise AssertionError("the complete genotype store must not be materialized")

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        self.read_count += 1
        result = self._values[:, marker_slice]
        if sample_indices is not None:
            result = result[sample_indices]
        return result.copy()


class _LabeledNoMaterializationStore(_NoMaterializationStore):
    def __init__(self, genotype: GenotypeData) -> None:
        super().__init__(genotype.GD)
        self.taxa: StrVector = genotype.taxa
        self.marker_ids: StrVector = np.asarray(genotype.GM["SNP"], dtype=str)
        self.chromosomes: StrVector = np.asarray(genotype.GM["Chromosome"], dtype=str)
        self.positions: FloatVector = np.asarray(
            genotype.GM["Position"], dtype=np.float64
        )


class _RecordingParentStore:
    """Read-only parent double that records bounded reads from GenotypeView."""

    def __init__(self, values: FloatMatrix) -> None:
        self._values: FloatMatrix = values
        self.marker_slices: list[slice] = []
        self.sample_requests: list[IntVector | slice | None] = []

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> FloatMatrix:
        raise AssertionError("GenotypeView must not materialize its parent")

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        self.marker_slices.append(marker_slice)
        self.sample_requests.append(sample_indices)
        block = self._values[:, marker_slice]
        if sample_indices is not None:
            block = block[sample_indices]
        result = np.array(block, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result


class _ChunkedRecordingStore(_RecordingParentStore):
    def __init__(self, values: FloatMatrix, marker_chunk_size: int = 8) -> None:
        super().__init__(values)
        self._marker_chunk_size: int = marker_chunk_size

    @property
    def marker_chunk_size(self) -> int:
        return self._marker_chunk_size


def _genotype_data() -> GenotypeData:
    values = np.asarray([
        [0.0, 1.0, 2.0, 0.0],
        [1.0, 1.0, 0.0, 2.0],
        [2.0, 0.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 0.0],
    ])
    marker_map = pd.DataFrame({
        "SNP": ["s1", "s2", "s3", "s4"],
        "Chromosome": [1, 1, 2, 2],
        "Position": [10.0, 20.0, 30.0, 40.0],
    })
    return GenotypeData(values, marker_map, np.asarray(["a", "b", "c", "d"]))


def _hapmap_data() -> pd.DataFrame:
    metadata = [
        "rs",
        "alleles",
        "chrom",
        "pos",
        "strand",
        "assembly",
        "center",
        "protLSID",
        "assayLSID",
        "panelLSID",
        "QCcode",
    ]
    rows = [
        ["s1", "A/T", 1, 10, "+", "NA", "NA", "NA", "NA", "NA", "NA"],
        ["s2", "C/G", 1, 20, "+", "NA", "NA", "NA", "NA", "NA", "NA"],
        ["s3", "A/G", 2, 30, "+", "NA", "NA", "NA", "NA", "NA", "NA"],
        ["s4", "C/T", 2, 40, "+", "NA", "NA", "NA", "NA", "NA", "NA"],
        ["s5", "A/C", 3, 50, "+", "NA", "NA", "NA", "NA", "NA", "NA"],
    ]
    calls = [
        ["A", "A", "T", "N"],
        ["C", "S", "G", "C"],
        ["G", "A", "R", "N"],
        ["T", "C", "Y", "T"],
        ["A", "C", "M", "A"],
    ]
    return pd.DataFrame(
        [
            metadata_row + marker_calls
            for metadata_row, marker_calls in zip(rows, calls)
        ],
        columns=[*metadata, "a", "b", "c", "d"],
    )


def test_array_store_returns_readonly_ordered_blocks() -> None:
    genotype = _genotype_data().GD
    store = ArrayGenotypeStore(genotype)

    block = store.read_markers(slice(1, 4), np.asarray([3, 0], dtype=np.int_))

    np.testing.assert_array_equal(block, genotype[[3, 0], 1:4])
    assert not block.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        block[0, 0] = -99.0


def test_vanraden_reads_store_blocks_without_whole_array_conversion() -> None:
    genotype = _genotype_data().GD
    store = _NoMaterializationStore(genotype)

    expected = vanraden_kinship(genotype, marker_workspace_mib=0.001)
    actual = vanraden_kinship(store, marker_workspace_mib=0.001)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert store.read_count >= 2


def test_maf_filter_returns_bounded_store_view_without_materialization() -> None:
    genotype = np.asarray([
        [0.0, 0.0, 2.0, 0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        [0.0, 2.0, 2.0, 1.0, np.nan, 2.0],
        [0.0, 1.0, 2.0, 1.0, 1.0, 2.0],
    ])
    store = _RecordingParentStore(genotype)

    expected, expected_indices = maf_filter(
        genotype,
        threshold=0.2,
        marker_workspace_mib=0.0001,
    )
    actual, actual_indices = maf_filter(
        store,
        threshold=0.2,
        marker_workspace_mib=0.0001,
    )

    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual.read_markers(slice(None)), expected)
    assert store.marker_slices[:2] == [slice(0, 3), slice(3, 6)]
    assert not actual.read_markers(slice(None)).flags.writeable


def test_genotype_view_reads_contiguous_parent_markers_with_selected_samples() -> None:
    values = _genotype_data().GD
    parent = _RecordingParentStore(values)
    view = GenotypeView(
        parent,
        sample_indices=np.asarray([3, 1], dtype=np.int_),
        marker_indices=np.asarray([1, 2, 3], dtype=np.int_),
    )

    actual = view.read_markers(slice(0, 2))

    assert view.shape == (2, 3)
    np.testing.assert_array_equal(actual, values[[3, 1], 1:3])
    assert parent.marker_slices == [slice(1, 3)]
    np.testing.assert_array_equal(parent.sample_requests[0], np.asarray([3, 1]))
    assert not actual.flags.writeable


def test_genotype_view_identity_forwards_relative_sample_selection() -> None:
    values = _genotype_data().GD
    parent = _RecordingParentStore(values)
    view = GenotypeView(parent)

    actual = view.read_markers(slice(1, 3), slice(3, 0, -2))

    np.testing.assert_array_equal(actual, values[3:0:-2, 1:3])
    assert parent.marker_slices == [slice(1, 3)]
    assert parent.sample_requests == [slice(3, 0, -2)]


def test_genotype_view_coalesces_ordered_runs_and_preserves_repeats() -> None:
    values = _genotype_data().GD
    parent = _RecordingParentStore(values)
    marker_indices = np.asarray([3, 1, 2, 0, 0], dtype=np.int_)
    view = GenotypeView(parent, marker_indices=marker_indices)

    actual = view.read_markers(slice(None))

    np.testing.assert_array_equal(actual, values[:, marker_indices])
    assert parent.marker_slices == [
        slice(3, 4),
        slice(1, 3),
        slice(0, 1),
        slice(0, 1),
    ]


def test_genotype_view_coalesces_dense_selections_within_storage_chunks() -> None:
    values = np.arange(64, dtype=np.float64).reshape(4, 16)
    parent = _ChunkedRecordingStore(values)
    marker_indices = np.asarray([6, 0, 2, 6, 8, 10, 12, 14], dtype=np.int_)
    view = GenotypeView(parent, marker_indices=marker_indices)

    actual = view.read_markers(slice(None))

    np.testing.assert_array_equal(actual, values[:, marker_indices])
    assert parent.marker_slices == [slice(0, 7), slice(8, 15)]


def test_genotype_view_does_not_overread_sparse_storage_chunks() -> None:
    values = np.arange(64, dtype=np.float64).reshape(4, 16)
    parent = _ChunkedRecordingStore(values, marker_chunk_size=16)
    marker_indices = np.asarray([0, 15], dtype=np.int_)
    view = GenotypeView(parent, marker_indices=marker_indices)

    actual = view.read_markers(slice(None))

    np.testing.assert_array_equal(actual, values[:, marker_indices])
    assert parent.marker_slices == [slice(0, 1), slice(15, 16)]


def test_genotype_view_propagates_chunks_through_sample_only_view() -> None:
    values = np.arange(64, dtype=np.float64).reshape(4, 16)
    parent = _ChunkedRecordingStore(values)
    samples = GenotypeView(
        parent,
        sample_indices=np.asarray([3, 1], dtype=np.int_),
    )
    markers = np.asarray([0, 2, 4, 6], dtype=np.int_)
    view = GenotypeView(samples, marker_indices=markers)

    actual = view.read_markers(slice(None))

    np.testing.assert_array_equal(actual, values[[3, 1]][:, markers])
    assert parent.marker_slices == [slice(0, 7)]


def test_genotype_view_composes_nested_samples_and_markers() -> None:
    values = _genotype_data().GD
    parent = _RecordingParentStore(values)
    outer = GenotypeView(
        parent,
        sample_indices=np.asarray([3, 1, 0], dtype=np.int_),
        marker_indices=np.asarray([3, 1, 2], dtype=np.int_),
    )
    inner = GenotypeView(
        outer,
        sample_indices=np.asarray([1, 1], dtype=np.int_),
        marker_indices=np.asarray([2, 0], dtype=np.int_),
    )

    actual = inner.read_markers(slice(None), np.asarray([1, 0], dtype=np.int_))

    np.testing.assert_array_equal(actual, values[[1, 1], :][:, [2, 3]])
    assert parent.marker_slices == [slice(2, 3), slice(3, 4)]
    np.testing.assert_array_equal(parent.sample_requests[0], np.asarray([1, 1]))


def test_genotype_view_empty_selection_returns_independent_readonly_block() -> None:
    values = _genotype_data().GD
    parent = _RecordingParentStore(values)
    view = GenotypeView(
        parent,
        sample_indices=np.asarray([], dtype=np.int_),
        marker_indices=np.asarray([], dtype=np.int_),
    )

    actual = view.read_markers(slice(None))

    assert view.shape == (0, 0)
    assert actual.shape == (0, 0)
    assert parent.marker_slices == []
    assert not actual.flags.writeable


@pytest.mark.parametrize(
    ("sample_indices", "marker_indices", "error", "message"),
    [
        (
            np.asarray([True, False]),
            None,
            TypeError,
            "integer indices",
        ),
        (
            None,
            np.asarray([1.0]),
            TypeError,
            "integer indices",
        ),
        (
            np.asarray([[0]], dtype=np.int_),
            None,
            ValueError,
            "one-dimensional",
        ),
        (
            None,
            np.asarray([4], dtype=np.int_),
            IndexError,
            "outside",
        ),
    ],
)
def test_genotype_view_rejects_invalid_selection_indices(
    sample_indices: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None,
    marker_indices: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        GenotypeView(
            _RecordingParentStore(_genotype_data().GD),
            sample_indices=sample_indices,
            marker_indices=marker_indices,
        )


def test_numpy_store_round_trip_preserves_data_and_metadata(tmp_path: Path) -> None:
    genotype = _genotype_data()
    path = tmp_path / "genotype"

    write_numpy_genotype(path, genotype, marker_chunk_size=2)

    with open_genotype_store(path) as store:
        assert isinstance(store, NumpyGenotypeStore)
        assert isinstance(store, LabeledGenotypeStore)
        assert store.shape == genotype.GD.shape
        np.testing.assert_array_equal(store.taxa, genotype.taxa)
        np.testing.assert_array_equal(
            store.read_markers(slice(1, 4), np.asarray([3, 0], dtype=np.int_)),
            genotype.GD[[3, 0], 1:4],
        )
        assert store.marker_map["SNP"].tolist() == genotype.GM["SNP"].tolist()
        assert store.marker_map["Chromosome"].tolist() == ["1", "1", "2", "2"]
        np.testing.assert_array_equal(
            store.marker_map["Position"], genotype.GM["Position"]
        )
        np.testing.assert_allclose(
            vanraden_kinship(store, marker_workspace_mib=0.001),
            vanraden_kinship(genotype.GD, marker_workspace_mib=0.001),
            rtol=1e-12,
            atol=1e-12,
        )

    assert store.closed
    with pytest.raises(ValueError, match="closed"):
        store.read_markers(slice(0, 1))


def test_hdf5_store_round_trip_preserves_data_and_metadata(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    genotype = _genotype_data()
    path = tmp_path / "genotype.h5"

    write_hdf5_genotype(path, genotype, marker_chunk_size=2)

    with HDF5GenotypeStore(path) as store:
        assert isinstance(store, LabeledGenotypeStore)
        assert store.shape == genotype.GD.shape
        assert store.marker_chunk_size == 2
        np.testing.assert_array_equal(store.taxa, genotype.taxa)
        np.testing.assert_array_equal(
            store.read_markers(slice(1, 4), np.asarray([3, 0], dtype=np.int_)),
            genotype.GD[[3, 0], 1:4],
        )
        np.testing.assert_array_equal(
            store.read_markers(
                slice(1, 4),
                np.asarray([3, 0, 3], dtype=np.int_),
            ),
            genotype.GD[[3, 0, 3], 1:4],
        )
        np.testing.assert_array_equal(
            store.read_markers(slice(1, 4), slice(3, 0, -2)),
            genotype.GD[3:0:-2, 1:4],
        )
        assert store.marker_map["SNP"].tolist() == genotype.GM["SNP"].tolist()
        assert store.marker_map["Chromosome"].tolist() == ["1", "1", "2", "2"]
        np.testing.assert_array_equal(
            store.marker_map["Position"], genotype.GM["Position"]
        )
        np.testing.assert_allclose(
            vanraden_kinship(store, marker_workspace_mib=0.001),
            vanraden_kinship(genotype.GD, marker_workspace_mib=0.001),
            rtol=1e-12,
            atol=1e-12,
        )

    assert store.closed


def test_zarr_store_round_trip_preserves_data_and_metadata(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    base = _genotype_data()
    genotype = GenotypeData(
        base.GD,
        base.GM,
        np.asarray(["样本-1", "sample-2", "sample-3", "sample-4"]),
    )
    path = tmp_path / "genotype.zarr"

    write_zarr_genotype(path, genotype, marker_chunk_size=2)

    with open_genotype_store(path) as store:
        assert isinstance(store, ZarrGenotypeStore)
        assert store.shape == genotype.GD.shape
        assert store.marker_chunk_size == 2
        np.testing.assert_array_equal(store.taxa, genotype.taxa)
        np.testing.assert_array_equal(
            store.read_markers(slice(1, 4), np.asarray([3, 0, 3], dtype=np.int_)),
            genotype.GD[[3, 0, 3], 1:4],
        )
        np.testing.assert_array_equal(
            store.read_markers(slice(1, 4), slice(3, 0, -2)),
            genotype.GD[3:0:-2, 1:4],
        )
        assert store.marker_map["SNP"].tolist() == genotype.GM["SNP"].tolist()

    assert store.closed


def test_zarr_suffix_selects_zarr_backend(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    path = tmp_path / "genotype.zarr"

    write_genotype_store(path, _genotype_data())

    with open_genotype_store(path) as store:
        assert isinstance(store, ZarrGenotypeStore)


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
def test_store_to_store_conversion_reads_bounded_marker_blocks(
    tmp_path: Path,
    backend: StorageBackend,
) -> None:
    genotype = _genotype_data()
    source = _LabeledNoMaterializationStore(genotype)
    suffix = {"numpy": "", "hdf5": ".h5", "zarr": ".zarr"}[backend]
    path = tmp_path / f"converted{suffix}"

    write_genotype_store(path, source, backend=backend, marker_chunk_size=2)

    assert source.read_count == 2
    with open_genotype_store(path, backend=backend) as converted:
        np.testing.assert_array_equal(converted.read_markers(slice(None)), genotype.GD)
        np.testing.assert_array_equal(converted.taxa, genotype.taxa)
        np.testing.assert_array_equal(
            converted.marker_ids, genotype.GM["SNP"].to_numpy(dtype=str)
        )
        np.testing.assert_array_equal(
            converted.chromosomes,
            genotype.GM["Chromosome"].to_numpy(dtype=str),
        )
        np.testing.assert_array_equal(
            converted.positions,
            genotype.GM["Position"].to_numpy(dtype=np.float64),
        )


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
@pytest.mark.parametrize("impute_method", ["middle", "mean"])
def test_numeric_store_import_streams_sample_rows_once_per_required_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: StorageBackend,
    impute_method: str,
) -> None:
    genotype_path = tmp_path / "genotype.tsv"
    marker_map = pd.DataFrame({
        "SNP": ["s1", "s2", "s3", "s4", "s5"],
        "Chromosome": [1, 1, 2, 2, 3],
        "Position": [10, 20, 30, 40, 50],
    })
    pd.DataFrame({
        "Taxa": ["a", "b", "c", "d"],
        "s1": [0.0, 1.0, 2.0, np.nan],
        "s2": [2.0, 1.0, 0.0, 1.0],
        "s3": [0.0, np.nan, 2.0, 1.0],
        "s4": [1.0, 2.0, 0.0, 1.0],
        "s5": [2.0, 2.0, np.nan, 0.0],
    }).to_csv(genotype_path, sep="\t", index=False)
    expected = read_numeric(genotype_path, marker_map, impute_method=impute_method)
    original_read_csv = cast(_ReadNumericCsv, pd.read_csv)
    requests: list[tuple[int | None, list[str] | None, int | None]] = []

    def bounded_read_csv(
        path: Path,
        *,
        sep: str,
        nrows: int | None = None,
        usecols: list[str] | None = None,
        chunksize: int | None = None,
        low_memory: bool = False,
    ) -> pd.DataFrame | Iterator[pd.DataFrame]:
        if nrows is None and usecols is None and chunksize is None:
            raise AssertionError("numeric import must not read the complete GD table")
        requests.append((nrows, usecols, chunksize))
        return original_read_csv(
            path,
            sep=sep,
            nrows=nrows,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=low_memory,
        )

    monkeypatch.setattr("pygapit.io.formats.pd.read_csv", bounded_read_csv)
    suffix = {"numpy": "", "hdf5": ".h5", "zarr": ".zarr"}[backend]
    store_path = tmp_path / f"numeric-store{suffix}"

    import_numeric_genotype_store(
        store_path,
        genotype_path,
        marker_map,
        impute_method=impute_method,
        backend=backend,
        marker_chunk_size=2,
        marker_workspace_mib=0.00008,
    )

    expected_requests: list[tuple[int | None, list[str] | None, int | None]] = [
        (0, None, None),
        (None, ["Taxa"], None),
    ]
    expected_requests.extend([(None, None, 2)] * (2 if impute_method == "mean" else 1))
    assert requests == expected_requests
    with open_genotype_store(store_path, backend=backend) as store:
        np.testing.assert_allclose(
            store.read_markers(slice(None)), expected.GD, rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(store.taxa, expected.taxa)
        np.testing.assert_array_equal(
            store.marker_ids, expected.GM["SNP"].to_numpy(dtype=str)
        )


def test_numeric_store_import_none_preserves_missing_values_and_marker_metadata(
    tmp_path: Path,
) -> None:
    """The one-pass NumPy path must not alter ``none`` imputation semantics."""
    genotype_path = tmp_path / "genotype.tsv"
    marker_map = pd.DataFrame({
        "SNP": ["s1", "s2", "s3"],
        "Chromosome": [1, "X", 2],
        "Position": [10, 20, 30],
    })
    pd.DataFrame({
        "Taxa": ["first", "second"],
        "s1": [0.0, np.nan],
        "s2": [np.nan, 2.0],
        "s3": [1.0, 0.0],
    }).to_csv(genotype_path, sep="\t", index=False)
    expected = read_numeric(genotype_path, marker_map, impute_method="none")
    store_path = tmp_path / "numeric-store"

    import_numeric_genotype_store(
        store_path,
        genotype_path,
        marker_map,
        impute_method="none",
        backend="numpy",
        marker_chunk_size=1,
        marker_workspace_mib=0.00003,
    )

    with open_genotype_store(store_path, backend="numpy") as store:
        np.testing.assert_equal(store.read_markers(slice(None)), expected.GD)
        np.testing.assert_array_equal(store.taxa, expected.taxa)
        np.testing.assert_array_equal(
            store.marker_ids, marker_map["SNP"].to_numpy(dtype=str)
        )
        np.testing.assert_array_equal(
            store.chromosomes, marker_map["Chromosome"].to_numpy(dtype=str)
        )
        np.testing.assert_array_equal(
            store.positions, marker_map["Position"].to_numpy(dtype=np.float64)
        )


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
@pytest.mark.parametrize("major_allele_zero", [False, True])
def test_hapmap_store_import_matches_in_memory_reader(
    tmp_path: Path,
    backend: StorageBackend,
    major_allele_zero: bool,
) -> None:
    hapmap_path = tmp_path / "genotype.hmp.txt"
    _hapmap_data().to_csv(hapmap_path, sep="\t", index=False)
    expected = read_hapmap(
        hapmap_path,
        major_allele_zero=major_allele_zero,
        impute_method="mean",
    )
    suffix = {"numpy": "", "hdf5": ".h5", "zarr": ".zarr"}[backend]
    store_path = tmp_path / f"hapmap-store{suffix}"

    import_hapmap_genotype_store(
        store_path,
        hapmap_path,
        major_allele_zero=major_allele_zero,
        impute_method="mean",
        backend=backend,
        marker_chunk_size=2,
    )

    with open_genotype_store(store_path, backend=backend) as store:
        np.testing.assert_allclose(
            store.read_markers(slice(None)), expected.GD, rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(store.taxa, expected.taxa)
        np.testing.assert_array_equal(
            store.marker_ids, expected.GM["SNP"].to_numpy(dtype=str)
        )
        np.testing.assert_array_equal(
            store.chromosomes,
            expected.GM["Chromosome"].to_numpy(dtype=str),
        )
        np.testing.assert_array_equal(
            store.positions,
            expected.GM["Position"].to_numpy(dtype=np.float64),
        )


def test_hdf5_store_supports_sample_batched_tall_pca(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    base = _genotype_data()
    values = np.tile(base.GD, (5, 1))
    genotype = GenotypeData(
        values,
        base.GM,
        np.asarray([f"sample-{index}" for index in range(len(values))]),
    )
    path = tmp_path / "tall-genotype.h5"
    write_hdf5_genotype(path, genotype, marker_chunk_size=2)

    expected = compute_pca(
        values,
        n_components=2,
        maf_filter=0.0,
        marker_workspace_mib=32.0,
    )
    with HDF5GenotypeStore(path) as store:
        actual = compute_pca(
            store,
            n_components=2,
            maf_filter=0.0,
            marker_workspace_mib=0.0001,
        )

    np.testing.assert_allclose(actual.eigenvalues, expected.eigenvalues, rtol=1e-12)
    np.testing.assert_allclose(
        actual.scores @ actual.scores.T,
        expected.scores @ expected.scores.T,
        rtol=1e-11,
        atol=1e-11,
    )


def test_auto_backend_uses_hdf5_when_available(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    path = tmp_path / "genotype-store"

    write_genotype_store(path, _genotype_data())

    assert path.is_file()
    with open_genotype_store(path) as store:
        assert isinstance(store, HDF5GenotypeStore)
        np.testing.assert_array_equal(
            store.read_markers(slice(0, 4)),
            _genotype_data().GD,
        )


def test_auto_backend_uses_zarr_without_h5py(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("zarr")
    original_import = builtins.__import__

    def import_without_h5py(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "h5py":
            raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_h5py)

    path = tmp_path / "genotype"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_genotype_store(path, _genotype_data())
        with open_genotype_store(path) as store:
            assert isinstance(store, ZarrGenotypeStore)
            np.testing.assert_array_equal(
                store.read_markers(slice(0, 4)),
                _genotype_data().GD,
            )


def test_auto_backend_silently_falls_back_to_numpy_without_optional_backends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def import_without_optional_backends(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name in {"h5py", "zarr"}:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_optional_backends)

    path = tmp_path / "genotype"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_genotype_store(path, _genotype_data())
        with open_genotype_store(path) as store:
            assert isinstance(store, NumpyGenotypeStore)
            np.testing.assert_array_equal(
                store.read_markers(slice(0, 4)),
                _genotype_data().GD,
            )


def test_explicit_hdf5_backend_reports_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def import_without_h5py(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "h5py":
            raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_h5py)

    with pytest.raises(ImportError, match=r"pygapit-ng\[bigdata\]"):
        write_genotype_store(
            tmp_path / "genotype",
            _genotype_data(),
            backend="hdf5",
        )


def test_explicit_zarr_backend_reports_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def import_without_zarr(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "zarr":
            raise ModuleNotFoundError("No module named 'zarr'", name="zarr")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_zarr)

    with pytest.raises(ImportError, match=r"pygapit-ng\[bigdata\]"):
        write_genotype_store(
            tmp_path / "genotype.zarr",
            _genotype_data(),
            backend="zarr",
        )


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
def test_store_sample_indices_follow_numpy_semantics(
    tmp_path: Path,
    backend: StorageBackend,
) -> None:
    genotype = _genotype_data()
    path = tmp_path / f"genotype-{backend}"
    write_genotype_store(path, genotype, backend=backend)

    with open_genotype_store(path, backend=backend) as store:
        sample_indices = np.asarray([-1, 0, -1], dtype=np.int_)
        actual = store.read_markers(slice(1, 4), sample_indices)
        np.testing.assert_array_equal(
            actual,
            genotype.GD[sample_indices, 1:4],
        )


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
@pytest.mark.parametrize("sample_index", [-5, 4])
def test_store_rejects_out_of_bounds_sample_indices(
    tmp_path: Path,
    backend: StorageBackend,
    sample_index: int,
) -> None:
    path = tmp_path / f"genotype-{backend}"
    write_genotype_store(path, _genotype_data(), backend=backend)

    with (
        open_genotype_store(path, backend=backend) as store,
        pytest.raises(IndexError, match="outside"),
    ):
        store.read_markers(
            slice(None),
            np.asarray([sample_index], dtype=np.int_),
        )


@pytest.mark.parametrize("backend", ["numpy", "hdf5", "zarr"])
@pytest.mark.parametrize(("shape", "message"), [((0, 2), "sample"), ((2, 0), "marker")])
def test_store_writers_reject_empty_dimensions_before_creating_output(
    tmp_path: Path,
    backend: StorageBackend,
    shape: tuple[int, int],
    message: str,
) -> None:
    rows, columns = shape
    genotype = GenotypeData(
        np.empty(shape, dtype=np.float64),
        pd.DataFrame({
            "SNP": [f"s{index}" for index in range(columns)],
            "Chromosome": [1] * columns,
            "Position": np.arange(columns, dtype=np.float64),
        }),
        np.asarray([f"taxon-{index}" for index in range(rows)], dtype=str),
    )
    path = tmp_path / f"empty-{backend}"

    with pytest.raises(ValueError, match=message):
        write_genotype_store(path, genotype, backend=backend)

    assert not path.exists()


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_hdf5_writer_validates_marker_chunk_size(
    tmp_path: Path,
    chunk_size: object,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        write_hdf5_genotype(
            tmp_path / "genotype.h5",
            _genotype_data(),
            marker_chunk_size=cast(int, chunk_size),
        )


@pytest.mark.parametrize("backend", ["missing", "HDF5", True])
def test_generic_writer_rejects_unknown_backends(
    tmp_path: Path,
    backend: object,
) -> None:
    with pytest.raises(ValueError, match="backend must be"):
        write_genotype_store(
            tmp_path / "genotype",
            _genotype_data(),
            backend=cast(StorageBackend, backend),
        )


@pytest.mark.parametrize("backend", ["missing"])
def test_generic_reader_rejects_unknown_backend(
    tmp_path: Path, backend: object
) -> None:
    with pytest.raises(ValueError, match="backend must be"):
        open_genotype_store(
            tmp_path / "genotype",
            backend=cast(StorageBackend, backend),
        )


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
def test_blocks_survive_close_and_file_removal(
    tmp_path: Path, backend: StorageBackend
) -> None:
    import shutil

    path = tmp_path / "store"
    genotype = _genotype_data()
    write_genotype_store(path, genotype, backend=backend)
    store = open_genotype_store(path, backend=backend)
    block = store.read_markers(slice(1, 3))
    store.close()
    store.close()
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    np.testing.assert_array_equal(block, genotype.GD[:, 1:3])
    assert not block.flags.writeable
    with pytest.raises(ValueError, match="closed"):
        store.read_markers(slice(0, 1))


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
@pytest.mark.parametrize("complete,schema", [(False, 1), (True, 2)])
def test_store_rejects_incomplete_or_unsupported_format(
    tmp_path: Path, backend: StorageBackend, complete: bool, schema: int
) -> None:
    path = tmp_path / "store"
    write_genotype_store(path, _genotype_data(), backend=backend)
    if backend == "numpy":
        (path / "metadata.json").write_text(
            json.dumps({"complete": complete, "schema_version": schema}),
            encoding="utf-8",
        )
    elif backend == "hdf5":
        import h5py

        with h5py.File(path, "r+") as handle:
            handle.attrs["complete"] = complete
            handle.attrs["schema_version"] = schema
    else:
        from pygapit.io._storage_zarr import _require_zarr

        group = _require_zarr().open_group(str(path), mode="r+")
        group.attrs["complete"] = complete
        group.attrs["schema_version"] = schema
    with pytest.raises(ValueError):
        open_genotype_store(path, backend=backend)


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", "false"),
        ("complete", 1),
        ("schema_version", True),
        ("schema_version", 1.0),
    ],
)
def test_store_rejects_invalid_header_types(
    tmp_path: Path,
    backend: StorageBackend,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "store"
    write_genotype_store(path, _genotype_data(), backend=backend)
    if backend == "numpy":
        metadata: dict[str, object] = {"complete": True, "schema_version": 1}
        metadata[field] = value
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    elif backend == "hdf5":
        import h5py

        with h5py.File(path, "r+") as handle:
            handle.attrs[field] = value
    else:
        from pygapit.io._storage_zarr import _require_zarr

        group = _require_zarr().open_group(str(path), mode="r+")
        group.attrs[field] = value
    with pytest.raises(ValueError):
        open_genotype_store(path, backend=backend)


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
@pytest.mark.parametrize(
    "values",
    [np.ones(4, dtype=np.float64), np.ones((4, 4), dtype=np.int32)],
)
def test_store_rejects_invalid_genotype_array(
    tmp_path: Path,
    backend: StorageBackend,
    values: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> None:
    path = tmp_path / "store"
    write_genotype_store(path, _genotype_data(), backend=backend)
    if backend == "numpy":
        np.save(path / "genotype.npy", values)
    elif backend == "hdf5":
        import h5py

        with h5py.File(path, "r+") as handle:
            del handle["genotype"]
            handle["genotype"] = values
    else:
        from pygapit.io._storage_zarr import _require_zarr

        group = _require_zarr().open_group(str(path), mode="r+")
        del group["genotype"]
        array = group.create_array(
            "genotype",
            shape=values.shape,
            dtype=values.dtype,
            chunks=values.shape,
        )
        array[:] = values
    with pytest.raises(ValueError, match="two-dimensional float64"):
        open_genotype_store(path, backend=backend)


@pytest.mark.parametrize("backend", _STORAGE_BACKENDS)
@pytest.mark.parametrize("field", ["taxa", "marker_id"])
def test_store_rejects_misaligned_metadata(
    tmp_path: Path,
    backend: StorageBackend,
    field: str,
) -> None:
    path = tmp_path / "store"
    write_genotype_store(path, _genotype_data(), backend=backend)
    values = np.asarray(["only-one"])
    if backend == "numpy":
        np.save(path / f"{field}.npy", values)
    elif backend == "hdf5":
        import h5py

        name = "taxa" if field == "taxa" else "markers/id"
        with h5py.File(path, "r+") as handle:
            del handle[name]
            handle[name] = values.astype("S")
    else:
        from pygapit.io._storage_zarr import _require_zarr

        name = "taxa" if field == "taxa" else "markers/id"
        encoded = values.astype("S")
        group = _require_zarr().open_group(str(path), mode="r+")
        del group[name]
        array = group.create_array(
            name,
            shape=encoded.shape,
            dtype=encoded.dtype,
            chunks=encoded.shape,
        )
        array[:] = encoded
    with pytest.raises(ValueError, match="must contain"):
        open_genotype_store(path, backend=backend)


def test_numpy_preserves_json_parser_error(tmp_path: Path) -> None:
    path = tmp_path / "store"
    path.mkdir()
    (path / "metadata.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        NumpyGenotypeStore(path)


@pytest.mark.parametrize("metadata", [None, []])
def test_numpy_rejects_non_object_metadata(tmp_path: Path, metadata: object) -> None:
    path = tmp_path / "store"
    write_numpy_genotype(path, _genotype_data())
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        NumpyGenotypeStore(path)


if TYPE_CHECKING:
    from typing import assert_type

    def check_storage_types(path: Path) -> None:
        numpy_store = open_genotype_store(path, backend="numpy")
        hdf5_store = open_genotype_store(path, backend="hdf5")
        zarr_store = open_genotype_store(path, backend="zarr")
        assert_type(numpy_store, NumpyGenotypeStore)
        assert_type(hdf5_store, HDF5GenotypeStore)
        assert_type(zarr_store, ZarrGenotypeStore)
        array_filtered, array_indices = maf_filter(np.empty((2, 3)))
        store_filtered, store_indices = maf_filter(numpy_store)
        assert_type(array_filtered, FloatMatrix)
        assert_type(store_filtered, GenotypeView)
        assert_type(array_indices, IntVector)
        assert_type(store_indices, IntVector)
        for store in (numpy_store, hdf5_store, zarr_store):
            assert_type(store.read_markers(slice(None)), FloatMatrix)
            assert_type(store.taxa, StrVector)
            assert_type(store.marker_ids, StrVector)
            assert_type(store.chromosomes, StrVector)
            assert_type(store.positions, FloatVector)
