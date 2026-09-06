"""Chunk-readable genotype storage with transparent optional backends."""

from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .._typing import FloatMatrix, IntVector, as_float_matrix
from ._storage_hdf5 import HDF5GenotypeStore, write_hdf5_genotype
from ._storage_numpy import NumpyGenotypeStore, write_numpy_genotype

if t.TYPE_CHECKING:
    from .formats import GenotypeData


StorageBackend: t.TypeAlias = t.Literal["auto", "numpy", "hdf5"]
_BACKEND_ERROR = "backend must be 'auto', 'numpy', or 'hdf5'"

__all__ = [
    "ArrayGenotypeStore",
    "GenotypeStore",
    "GenotypeView",
    "HDF5GenotypeStore",
    "NumpyGenotypeStore",
    "StorageBackend",
    "as_genotype_store",
    "open_genotype_store",
    "write_genotype_store",
    "write_hdf5_genotype",
    "write_numpy_genotype",
]


@t.runtime_checkable
class GenotypeStore(t.Protocol):
    """A sample-by-marker matrix that can return read-only marker blocks.

    Implementations must preserve the requested marker order and return a
    two-dimensional float64-compatible array. Callers must not mutate returned
    blocks. Calling ``read_markers`` must not materialize markers outside the
    requested slice.
    """

    @property
    def shape(self) -> tuple[int, int]: ...

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix: ...


class ArrayGenotypeStore:
    """Expose an in-memory matrix through the genotype-store contract."""

    def __init__(self, genotype: npt.ArrayLike) -> None:
        self._genotype: FloatMatrix = as_float_matrix(genotype, name="genotype matrix")

    @property
    def shape(self) -> tuple[int, int]:
        return self._genotype.shape

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        block = self._genotype[:, marker_slice]
        if sample_indices is not None:
            block = block[sample_indices]
        result = np.asarray(block, dtype=np.float64).view()
        result.setflags(write=False)
        return result


def _normalize_view_indices(
    indices: npt.NDArray[np.generic] | slice | None,
    size: int,
    *,
    name: str,
) -> IntVector | None:
    """Return a validated selection, preserving an identity selection as None."""
    if indices is None:
        return None
    if isinstance(indices, slice):
        start, stop, step = indices.indices(size)
        if start == 0 and stop == size and step == 1:
            return None
        return np.arange(start, stop, step, dtype=np.int_)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(indices.dtype, np.bool_) or not np.issubdtype(
        indices.dtype,
        np.integer,
    ):
        raise TypeError(f"{name} must contain integer indices, not {indices.dtype}")
    normalized = np.array(indices, dtype=np.int_, copy=True)
    if np.any(normalized < 0) or np.any(normalized >= size):
        raise IndexError(f"{name} contains an index outside [0, {size})")
    return normalized


def _read_sample_selection(
    sample_indices: IntVector | slice | None,
    size: int,
) -> IntVector | slice | None:
    """Validate a relative sample selection without expanding identity slices."""
    if sample_indices is None:
        return None
    if isinstance(sample_indices, slice):
        start, stop, step = sample_indices.indices(size)
        if start == 0 and stop == size and step == 1:
            return None
        return slice(start, stop, step)
    return _normalize_view_indices(sample_indices, size, name="sample_indices")


def _selection_length(selection: IntVector | slice | None, size: int) -> int:
    """Return the number of items selected without expanding a slice."""
    if selection is None:
        return size
    if isinstance(selection, slice):
        return len(range(*selection.indices(size)))
    return len(selection)


@t.final
class GenotypeView:
    """Logical sample/marker selection over a chunk-readable genotype store."""

    def __init__(
        self,
        parent: FloatMatrix | GenotypeStore,
        *,
        sample_indices: npt.NDArray[np.generic] | slice | None = None,
        marker_indices: npt.NDArray[np.generic] | slice | None = None,
    ) -> None:
        self._parent: GenotypeStore = as_genotype_store(parent)
        parent_samples, parent_markers = self._parent.shape
        self._sample_indices: IntVector | None = _normalize_view_indices(
            sample_indices,
            parent_samples,
            name="sample_indices",
        )
        self._marker_indices: IntVector | None = _normalize_view_indices(
            marker_indices,
            parent_markers,
            name="marker_indices",
        )

    @property
    def shape(self) -> tuple[int, int]:
        parent_samples, parent_markers = self._parent.shape
        sample_count = (
            parent_samples
            if self._sample_indices is None
            else len(self._sample_indices)
        )
        marker_count = (
            parent_markers
            if self._marker_indices is None
            else len(self._marker_indices)
        )
        return sample_count, marker_count

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        """Read a view-relative marker block without materializing its parent."""
        sample_count, marker_count = self.shape
        direct_samples = _read_sample_selection(sample_indices, sample_count)
        if self._sample_indices is None:
            parent_samples = direct_samples
        else:
            sample_positions = _normalize_view_indices(
                sample_indices,
                sample_count,
                name="sample_indices",
            )
            parent_samples = (
                self._sample_indices
                if sample_positions is None
                else self._sample_indices[sample_positions]
            )
        if self._marker_indices is None:
            block = self._parent.read_markers(marker_slice, parent_samples)
            result = np.array(block, dtype=np.float64, copy=True)
            result.setflags(write=False)
            return result

        marker_positions = _normalize_view_indices(
            marker_slice,
            marker_count,
            name="marker_slice",
        )
        parent_markers = (
            self._marker_indices
            if marker_positions is None
            else self._marker_indices[marker_positions]
        )
        result = np.empty(
            (
                _selection_length(parent_samples, self._parent.shape[0]),
                len(parent_markers),
            ),
            dtype=np.float64,
        )
        if result.shape[0] == 0 or len(parent_markers) == 0:
            result.setflags(write=False)
            return result

        output_start = 0
        run_start = 0
        while run_start < len(parent_markers):
            run_stop = run_start + 1
            while (
                run_stop < len(parent_markers)
                and parent_markers[run_stop] == parent_markers[run_stop - 1] + 1
            ):
                run_stop += 1
            start = parent_markers[run_start]
            stop = parent_markers[run_stop - 1] + 1
            block = self._parent.read_markers(
                slice(start, stop),
                parent_samples,
            )
            width = run_stop - run_start
            result[:, output_start : output_start + width] = block
            output_start += width
            run_start = run_stop

        result.setflags(write=False)
        return result


def as_genotype_store(genotype: npt.ArrayLike | GenotypeStore) -> GenotypeStore:
    """Return an existing store or adapt an ordinary in-memory matrix."""
    if isinstance(genotype, GenotypeStore):
        return genotype
    return ArrayGenotypeStore(genotype)


@t.overload
def open_genotype_store(
    path: str | Path, *, backend: t.Literal["numpy"]
) -> NumpyGenotypeStore: ...


@t.overload
def open_genotype_store(
    path: str | Path, *, backend: t.Literal["hdf5"]
) -> HDF5GenotypeStore: ...


@t.overload
def open_genotype_store(
    path: str | Path, *, backend: StorageBackend = "auto"
) -> NumpyGenotypeStore | HDF5GenotypeStore: ...


def open_genotype_store(
    path: str | Path,
    *,
    backend: StorageBackend = "auto",
) -> NumpyGenotypeStore | HDF5GenotypeStore:
    """Open a disk-backed genotype, detecting its backend by default."""
    target = Path(path)
    if backend == "auto":
        backend = "numpy" if target.is_dir() else "hdf5"
    try:
        reader = {"numpy": NumpyGenotypeStore, "hdf5": HDF5GenotypeStore}[backend]
    except KeyError:
        raise ValueError(_BACKEND_ERROR) from None
    return reader(target)


def write_genotype_store(
    path: str | Path,
    genotype: GenotypeData,
    *,
    backend: StorageBackend = "auto",
    marker_chunk_size: int = 1024,
) -> None:
    """Write a store, transparently falling back to NumPy when needed.

    ``auto`` respects an HDF5 filename suffix. Otherwise it uses HDF5 when
    ``h5py`` is installed and the dependency-free NumPy backend when it is not.
    """
    target = Path(path)
    if backend != "auto":
        try:
            writer = {
                "numpy": write_numpy_genotype,
                "hdf5": write_hdf5_genotype,
            }[backend]
        except KeyError:
            raise ValueError(_BACKEND_ERROR) from None
        writer(target, genotype, marker_chunk_size=marker_chunk_size)
        return
    if target.suffix.lower() in {".h5", ".hdf5"}:
        write_hdf5_genotype(target, genotype, marker_chunk_size=marker_chunk_size)
        return

    try:
        import h5py
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise
        write_numpy_genotype(
            target,
            genotype,
            marker_chunk_size=marker_chunk_size,
        )
    else:
        _ = h5py
        write_hdf5_genotype(
            target,
            genotype,
            marker_chunk_size=marker_chunk_size,
        )
