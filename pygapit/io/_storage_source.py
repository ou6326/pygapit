"""Internal adapter for chunked writes from memory or an existing store."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, cast, final, runtime_checkable

import numpy as np
import pandas as pd

from .._typing import FloatMatrix, FloatVector, IntVector, StrVector
from ._genotype_store import LabeledGenotypeStore

if TYPE_CHECKING:
    from .._typing import Slice


class _InMemoryGenotypeData(Protocol):
    GD: FloatMatrix
    GM: pd.DataFrame
    taxa: StrVector


@runtime_checkable
class MarkerBlockWriteSource(Protocol):
    """A labeled one-pass source yielding consecutive marker blocks."""

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def taxa(self) -> StrVector: ...

    @property
    def marker_ids(self) -> StrVector: ...

    @property
    def chromosomes(self) -> StrVector: ...

    @property
    def positions(self) -> FloatVector: ...

    def iter_marker_blocks(
        self, marker_chunk_size: int
    ) -> Iterator[tuple[Slice, FloatMatrix]]: ...


@runtime_checkable
class SampleBlockWriteSource(Protocol):
    """A labeled one-pass source yielding consecutive sample blocks."""

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def taxa(self) -> StrVector: ...

    @property
    def marker_ids(self) -> StrVector: ...

    @property
    def chromosomes(self) -> StrVector: ...

    @property
    def positions(self) -> FloatVector: ...

    def iter_sample_blocks(
        self,
    ) -> Iterator[tuple[Slice, FloatMatrix]]: ...


type PreparedGenotypeWriteSource = (
    LabeledGenotypeStore | MarkerBlockWriteSource | SampleBlockWriteSource
)
type GenotypeWriteSource = _InMemoryGenotypeData | PreparedGenotypeWriteSource


@final
class _GenotypeDataSource:
    """Expose ``GenotypeData`` only to storage writers as a labeled source."""

    def __init__(self, genotype: _InMemoryGenotypeData) -> None:
        self._genotype: _InMemoryGenotypeData = genotype
        self._marker_ids = cast(
            StrVector,
            np.asarray(genotype.GM["SNP"].to_numpy(dtype=str), dtype=str),
        )
        self._chromosomes = cast(
            StrVector,
            np.asarray(genotype.GM["Chromosome"].to_numpy(dtype=str), dtype=str),
        )
        self._positions = cast(
            FloatVector,
            np.asarray(genotype.GM["Position"].to_numpy(dtype=np.float64)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._genotype.GD.shape

    @property
    def taxa(self) -> StrVector:
        return self._genotype.taxa

    @property
    def marker_ids(self) -> StrVector:
        return self._marker_ids

    @property
    def chromosomes(self) -> StrVector:
        return self._chromosomes

    @property
    def positions(self) -> FloatVector:
        return self._positions

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        block = self._genotype.GD[:, marker_slice]
        if sample_indices is not None:
            block = block[sample_indices]
        result = np.asarray(block, dtype=np.float64).view()
        result.setflags(write=False)
        return result


def as_genotype_write_source(
    genotype: GenotypeWriteSource,
) -> PreparedGenotypeWriteSource:
    """Adapt in-memory genotype data without changing its public runtime role."""
    if isinstance(
        genotype,
        (LabeledGenotypeStore, MarkerBlockWriteSource, SampleBlockWriteSource),
    ):
        source = genotype
    else:
        source = _GenotypeDataSource(genotype)
    rows, columns = source.shape
    if rows <= 0:
        raise ValueError("genotype source must contain at least one sample")
    if columns <= 0:
        raise ValueError("genotype source must contain at least one marker")
    return source


def iter_genotype_write_blocks(
    source: PreparedGenotypeWriteSource,
    marker_chunk_size: int,
) -> Iterator[tuple[Slice, Slice, FloatMatrix]]:
    """Yield a complete sequence of validated sample-by-marker blocks."""
    rows, columns = source.shape
    if isinstance(source, SampleBlockWriteSource):
        expected_start = 0
        for sample_slice, block in source.iter_sample_blocks():
            start, stop, step = sample_slice.indices(rows)
            values = np.asarray(block, dtype=np.float64)
            if step != 1 or start != expected_start or stop <= start:
                raise ValueError("sample blocks must be consecutive non-empty slices")
            if values.shape != (stop - start, columns):
                raise ValueError(
                    "sample block shape does not match its source slice; "
                    f"expected {(stop - start, columns)}, got {values.shape}"
                )
            yield slice(start, stop), slice(0, columns), values
            expected_start = stop
        if expected_start != rows:
            raise ValueError(
                "sample blocks did not cover the complete source; "
                f"expected {rows}, got {expected_start}"
            )
        return

    if isinstance(source, MarkerBlockWriteSource):
        expected_start = 0
        for marker_slice, block in source.iter_marker_blocks(marker_chunk_size):
            start, stop, step = marker_slice.indices(columns)
            values = np.asarray(block, dtype=np.float64)
            if step != 1 or start != expected_start or stop <= start:
                raise ValueError("marker blocks must be consecutive non-empty slices")
            if values.shape != (rows, stop - start):
                raise ValueError(
                    "marker block shape does not match its source slice; "
                    f"expected {(rows, stop - start)}, got {values.shape}"
                )
            yield slice(0, rows), slice(start, stop), values
            expected_start = stop
        if expected_start != columns:
            raise ValueError(
                "marker blocks did not cover the complete source; "
                f"expected {columns}, got {expected_start}"
            )
        return

    for start in range(0, columns, marker_chunk_size):
        stop = min(start + marker_chunk_size, columns)
        yield (
            slice(0, rows),
            slice(start, stop),
            source.read_markers(slice(start, stop)),
        )
