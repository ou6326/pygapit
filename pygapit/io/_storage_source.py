"""Internal adapter for chunked writes from memory or an existing store."""

from __future__ import annotations

from typing import Protocol, TypeAlias, cast, final

import numpy as np
import pandas as pd

from .._typing import FloatMatrix, FloatVector, IntVector, StrVector
from ._genotype_store import LabeledGenotypeStore


class _InMemoryGenotypeData(Protocol):
    GD: FloatMatrix
    GM: pd.DataFrame
    taxa: StrVector


GenotypeWriteSource: TypeAlias = _InMemoryGenotypeData | LabeledGenotypeStore


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
) -> LabeledGenotypeStore:
    """Adapt in-memory genotype data without changing its public runtime role."""
    if isinstance(genotype, LabeledGenotypeStore):
        return genotype
    return _GenotypeDataSource(genotype)
