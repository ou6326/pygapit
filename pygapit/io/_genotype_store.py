"""Backend-independent chunk-readable genotype contracts and views."""

from __future__ import annotations

import typing as t

import numpy as np
import numpy.typing as npt

from .._typing import FloatMatrix, FloatVector, IntVector, StrVector, as_float_matrix

_MAX_MARKER_READ_AMPLIFICATION = 4


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


@t.runtime_checkable
class MarkerChunkedGenotypeStore(GenotypeStore, t.Protocol):
    """A store exposing a positive physical marker-chunk width, when known."""

    @property
    def marker_chunk_size(self) -> int | None: ...


@t.runtime_checkable
class LabeledGenotypeStore(GenotypeStore, t.Protocol):
    """A genotype store carrying aligned sample and marker metadata."""

    @property
    def taxa(self) -> StrVector: ...

    @property
    def marker_ids(self) -> StrVector: ...

    @property
    def chromosomes(self) -> StrVector: ...

    @property
    def positions(self) -> FloatVector: ...


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
            sample_indices, parent_samples, name="sample_indices"
        )
        self._marker_indices: IntVector | None = _normalize_view_indices(
            marker_indices, parent_markers, name="marker_indices"
        )

    @property
    def shape(self) -> tuple[int, int]:
        parent_samples, parent_markers = self._parent.shape
        return (
            parent_samples
            if self._sample_indices is None
            else len(self._sample_indices),
            parent_markers
            if self._marker_indices is None
            else len(self._marker_indices),
        )

    @property
    def marker_chunk_size(self) -> int | None:
        if self._marker_indices is not None:
            return None
        if isinstance(self._parent, MarkerChunkedGenotypeStore):
            return self._parent.marker_chunk_size
        return None

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        sample_count, marker_count = self.shape
        direct_samples = _read_sample_selection(sample_indices, sample_count)
        if self._sample_indices is None:
            parent_samples = direct_samples
        else:
            sample_positions = _normalize_view_indices(
                sample_indices, sample_count, name="sample_indices"
            )
            parent_samples = (
                self._sample_indices
                if sample_positions is None
                else self._sample_indices[sample_positions]
            )
        if self._marker_indices is None:
            result = np.array(
                self._parent.read_markers(marker_slice, parent_samples),
                dtype=np.float64,
                copy=True,
            )
            result.setflags(write=False)
            return result

        marker_positions = _normalize_view_indices(
            marker_slice, marker_count, name="marker_slice"
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
        if 0 in result.shape:
            result.setflags(write=False)
            return result

        planned = np.zeros(len(parent_markers), dtype=np.bool_)
        if isinstance(self._parent, MarkerChunkedGenotypeStore):
            chunk_size = self._parent.marker_chunk_size
            if chunk_size is not None and chunk_size > 0:
                marker_chunks = parent_markers // chunk_size
                for chunk_index in np.unique(marker_chunks):
                    positions = np.flatnonzero(marker_chunks == chunk_index)
                    chunk_markers = parent_markers[positions]
                    unique_markers = np.unique(chunk_markers)
                    read_start = int(unique_markers[0])
                    read_stop = int(unique_markers[-1]) + 1
                    if read_stop - read_start > _MAX_MARKER_READ_AMPLIFICATION * len(
                        unique_markers
                    ):
                        continue
                    block = self._parent.read_markers(
                        slice(read_start, read_stop),
                        parent_samples,
                    )
                    result[:, positions] = block[:, chunk_markers - read_start]
                    planned[positions] = True

        run_start = 0
        while run_start < len(parent_markers):
            if planned[run_start]:
                run_start += 1
                continue
            run_stop = run_start + 1
            while (
                run_stop < len(parent_markers)
                and not planned[run_stop]
                and parent_markers[run_stop] == parent_markers[run_stop - 1] + 1
            ):
                run_stop += 1
            block = self._parent.read_markers(
                slice(parent_markers[run_start], parent_markers[run_stop - 1] + 1),
                parent_samples,
            )
            width = run_stop - run_start
            result[:, run_start:run_stop] = block[:, :width]
            run_start = run_stop
        result.setflags(write=False)
        return result


def as_genotype_store(genotype: npt.ArrayLike | GenotypeStore) -> GenotypeStore:
    """Return an existing store or adapt an ordinary in-memory matrix."""
    if isinstance(genotype, GenotypeStore):
        return genotype
    return ArrayGenotypeStore(genotype)
