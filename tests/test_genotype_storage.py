"""Contracts for chunk-readable in-memory and optional HDF5 genotypes."""

from __future__ import annotations

import builtins
import json
import warnings
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

from pygapit._typing import FloatMatrix, IntVector
from pygapit.io.formats import GenotypeData, maf_filter
from pygapit.io.storage import (
    ArrayGenotypeStore,
    GenotypeView,
    HDF5GenotypeStore,
    LabeledGenotypeStore,
    NumpyGenotypeStore,
    StorageBackend,
    open_genotype_store,
    write_genotype_store,
    write_hdf5_genotype,
    write_numpy_genotype,
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
]


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


def test_auto_backend_silently_falls_back_to_numpy_without_h5py(
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


def test_explicit_hdf5_backend_warns_when_dependency_is_missing(
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

    with (
        pytest.warns(RuntimeWarning, match="backend='numpy'"),
        pytest.raises(ImportError, match=r"pygapit-ng\[bigdata\]"),
    ):
        write_genotype_store(
            tmp_path / "genotype",
            _genotype_data(),
            backend="hdf5",
        )


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
    else:
        import h5py

        with h5py.File(path, "r+") as handle:
            handle.attrs["complete"] = complete
            handle.attrs["schema_version"] = schema
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
    else:
        import h5py

        with h5py.File(path, "r+") as handle:
            handle.attrs[field] = value
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
    else:
        import h5py

        with h5py.File(path, "r+") as handle:
            del handle["genotype"]
            handle["genotype"] = values
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
    else:
        import h5py

        name = "taxa" if field == "taxa" else "markers/id"
        with h5py.File(path, "r+") as handle:
            del handle[name]
            handle[name] = values.astype("S")
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
    from typing_extensions import assert_type

    from pygapit._typing import FloatVector, StrVector

    def check_storage_types(path: Path) -> None:
        numpy_store = open_genotype_store(path, backend="numpy")
        hdf5_store = open_genotype_store(path, backend="hdf5")
        assert_type(numpy_store, NumpyGenotypeStore)
        assert_type(hdf5_store, HDF5GenotypeStore)
        array_filtered, array_indices = maf_filter(np.empty((2, 3)))
        store_filtered, store_indices = maf_filter(numpy_store)
        assert_type(array_filtered, FloatMatrix)
        assert_type(store_filtered, GenotypeView)
        assert_type(array_indices, IntVector)
        assert_type(store_indices, IntVector)
        for store in (numpy_store, hdf5_store):
            assert_type(store.read_markers(slice(None)), FloatMatrix)
            assert_type(store.taxa, StrVector)
            assert_type(store.marker_ids, StrVector)
            assert_type(store.chromosomes, StrVector)
            assert_type(store.positions, FloatVector)
