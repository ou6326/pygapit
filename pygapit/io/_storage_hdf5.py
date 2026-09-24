"""Optional HDF5 implementation for the generic genotype storage API."""

from __future__ import annotations

import typing as t
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pandas as pd

from .._typing import FloatMatrix, FloatVector, IntVector, StrVector
from ._genotype_store import normalize_sample_selection
from ._storage_source import (
    as_genotype_write_source,
    iter_genotype_write_blocks,
    remove_incomplete_genotype_store,
)

if t.TYPE_CHECKING:
    from collections.abc import Mapping

    import h5py
    from h5py._hl.dataset import AsStrView

    from ._storage_source import GenotypeWriteSource


_HDF5_DATASET = "genotype"


class HDF5GenotypeStore:
    """Read a pyGAPIT HDF5 genotype without loading the complete matrix."""

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        try:
            import h5py
        except ModuleNotFoundError as exc:
            if exc.name != "h5py":
                raise
            _raise_missing_h5py(exc)
        self._file: h5py.File = h5py.File(self.path, "r")
        try:
            header = t.cast("Mapping[str, object]", self._file.attrs)
            complete = header.get("complete")
            if not isinstance(complete, (bool, np.bool_)):
                raise ValueError("store complete must be a boolean")  # noqa: TRY004 - invalid persisted data
            if not complete:
                raise ValueError("genotype import is incomplete")
            schema = header.get("schema_version")
            if isinstance(schema, (bool, np.bool_)) or not isinstance(
                schema, (int, np.integer)
            ):
                raise ValueError("store schema_version must be an integer")  # noqa: TRY004 - invalid persisted data
            if schema != 1:
                raise ValueError("unsupported genotype schema")
            contents = t.cast("Mapping[str, object]", self._file)
            dataset = contents.get(_HDF5_DATASET)
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError("HDF5 genotype must be a dataset")  # noqa: TRY004 - invalid persisted data
            self._dataset: h5py.Dataset = dataset
            shape = t.cast(tuple[int, ...], self._dataset.shape)
            dtype = t.cast(np.dtype[np.generic], self._dataset.dtype)
            if len(shape) != 2 or dtype != np.dtype(np.float64):
                raise ValueError(
                    "HDF5 genotype dataset must be two-dimensional float64"
                )
            self.taxa: StrVector = _read_strings(self._file, "taxa")
            self.marker_ids: StrVector = _read_strings(self._file, "markers/id")
            self.chromosomes: StrVector = _read_strings(
                self._file, "markers/chromosome"
            )
            positions = t.cast("h5py.Dataset", self._file["markers/position"])
            self.positions: FloatVector = t.cast(FloatVector, positions[:])
            rows, columns = self.shape
            for name, values, expected in (
                ("taxa", self.taxa, rows),
                ("marker ids", self.marker_ids, columns),
                ("chromosomes", self.chromosomes, columns),
                ("marker positions", self.positions, columns),
            ):
                if values.ndim != 1 or len(values) != expected:
                    raise ValueError(f"{name} must contain {expected} values")
        except Exception:
            self._file.close()
            raise

    @property
    def shape(self) -> tuple[int, int]:
        shape = t.cast(tuple[int, ...], self._dataset.shape)
        return shape[0], shape[1]

    @property
    def marker_chunk_size(self) -> int | None:
        chunks = t.cast(tuple[int, ...] | None, self._dataset.chunks)
        return None if chunks is None else chunks[1]

    @property
    def marker_map(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SNP": self.marker_ids,
            "Chromosome": self.chromosomes,
            "Position": self.positions,
        })

    @property
    def closed(self) -> bool:
        return not bool(t.cast(int, self._file.id.valid))

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        if self.closed:
            raise ValueError("HDF5 genotype store is closed")
        selection = normalize_sample_selection(sample_indices, self.shape[0])
        if selection is None:
            selected = t.cast(FloatMatrix, self._dataset[:, marker_slice])
        elif isinstance(selection, slice):
            start, stop, step = selection.indices(self.shape[0])
            if step > 0:
                selected = t.cast(
                    FloatMatrix,
                    self._dataset[slice(start, stop, step), marker_slice],
                )
            else:
                requested: IntVector = np.arange(start, stop, step, dtype=np.int_)
                selected = self._read_arbitrary_samples(requested, marker_slice)
        else:
            selected = self._read_arbitrary_samples(selection, marker_slice)
        result = np.asarray(selected, dtype=np.float64)
        result.setflags(write=False)
        return result

    def _read_arbitrary_samples(
        self,
        sample_indices: IntVector,
        marker_slice: slice,
    ) -> FloatMatrix:
        unique_indices, inverse = np.unique(sample_indices, return_inverse=True)
        unique_block = t.cast(
            FloatMatrix,
            self._dataset[unique_indices, marker_slice],
        )
        return unique_block[inverse]

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def write_hdf5_genotype(
    path: str | Path,
    genotype: GenotypeWriteSource,
    *,
    marker_chunk_size: int = 1024,
) -> None:
    """Write validated genotype data to the optional HDF5 backend."""
    if marker_chunk_size <= 0:
        raise ValueError("marker_chunk_size must be positive")
    source = as_genotype_write_source(genotype)
    try:
        import h5py
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise
        _raise_missing_h5py(exc)

    rows, columns = source.shape
    marker_chunk_size = min(marker_chunk_size, columns)
    sample_chunk_size = min(rows, max(1, (8 * 1024**2) // (8 * marker_chunk_size)))
    string_dtype = t.cast(t.Callable[[str], np.dtype[np.object_]], h5py.string_dtype)(
        "utf-8"
    )

    target = Path(path)
    created = False
    try:
        with h5py.File(target, "x") as handle:
            created = True
            create_dataset = t.cast(_DatasetWriter, handle.create_dataset)
            handle.attrs["schema_version"] = 1
            handle.attrs["complete"] = False
            matrix = create_dataset(
                _HDF5_DATASET,
                shape=(rows, columns),
                dtype=np.float64,
                chunks=(sample_chunk_size, marker_chunk_size),
            )
            for sample_slice, marker_slice, block in iter_genotype_write_blocks(
                source, marker_chunk_size
            ):
                matrix[sample_slice, marker_slice] = block
            create_dataset(
                "taxa",
                data=source.taxa.astype(object),
                dtype=string_dtype,
            )
            create_dataset(
                "markers/id",
                data=source.marker_ids.astype(object),
                dtype=string_dtype,
            )
            create_dataset(
                "markers/chromosome",
                data=source.chromosomes.astype(object),
                dtype=string_dtype,
            )
            create_dataset(
                "markers/position",
                data=source.positions,
                dtype=np.float64,
            )
            handle.attrs["complete"] = True
    except BaseException:
        if created:
            remove_incomplete_genotype_store(target)
        raise


def _raise_missing_h5py(exc: ModuleNotFoundError) -> t.NoReturn:
    raise ImportError(
        "HDF5 genotype storage requires the optional 'bigdata' feature; "
        "install pygapit-ng[bigdata], or use backend='numpy'."
    ) from exc


class _DatasetWriter(t.Protocol):
    """The subset of h5py's unannotated create_dataset signature used here."""

    def __call__(
        self,
        name: str,
        *,
        data: object | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: object = None,
        chunks: tuple[int, int] | None = None,
    ) -> h5py.Dataset: ...


def _read_strings(handle: h5py.File, name: str) -> StrVector:
    dataset = t.cast("h5py.Dataset", handle[name])
    as_strings = t.cast("t.Callable[[], AsStrView]", dataset.asstr)
    return np.asarray(t.cast(StrVector, as_strings()[:]), dtype=str)
