"""Dependency-free NumPy implementation for the genotype storage API."""

from __future__ import annotations

import json
import typing as t
from pathlib import Path
from types import TracebackType
from typing import Self, TypedDict

import numpy as np
import pandas as pd

from .._typing import FloatMatrix, FloatVector, IntVector, StrVector
from ._storage_source import as_genotype_write_source, iter_genotype_write_blocks

if t.TYPE_CHECKING:
    from ._storage_source import GenotypeWriteSource


_NUMPY_METADATA = "metadata.json"
_NUMPY_GENOTYPE = "genotype.npy"


class _NumpyStoreMetadata(TypedDict):
    schema_version: int
    complete: bool


class NumpyGenotypeStore:
    """Read a dependency-free, memory-mapped pyGAPIT genotype directory."""

    def __init__(self, path: str | Path):
        self.path: Path = Path(path)
        with (self.path / _NUMPY_METADATA).open(encoding="utf-8") as stream:
            metadata: _NumpyStoreMetadata = json.load(stream)
        try:
            complete = metadata.get("complete")
            schema = metadata.get("schema_version")
        except AttributeError as exc:
            raise ValueError("store metadata must be a JSON object") from exc
        if type(complete) is not bool:
            raise ValueError("store complete must be a boolean")
        if not complete:
            raise ValueError("genotype import is incomplete")
        if type(schema) is not int:
            raise ValueError("store schema_version must be an integer")
        if schema != 1:
            raise ValueError("unsupported genotype schema")
        loaded = np.lib.format.open_memmap(
            self.path / _NUMPY_GENOTYPE,
            mode="r",
        )
        if loaded.ndim != 2 or loaded.dtype != np.dtype(np.float64):
            raise ValueError("NumPy genotype dataset must be two-dimensional float64")
        self._genotype: FloatMatrix | None = t.cast(FloatMatrix, loaded)
        self._shape: tuple[int, int] = self._genotype.shape
        self.taxa: StrVector = t.cast(
            StrVector, np.load(self.path / "taxa.npy", allow_pickle=False)
        )
        self.marker_ids: StrVector = t.cast(
            StrVector, np.load(self.path / "marker_id.npy", allow_pickle=False)
        )
        self.chromosomes: StrVector = t.cast(
            StrVector, np.load(self.path / "marker_chromosome.npy", allow_pickle=False)
        )
        self.positions: FloatVector = t.cast(
            FloatVector, np.load(self.path / "marker_position.npy", allow_pickle=False)
        )
        rows, columns = self._shape
        for name, values, expected in (
            ("taxa", self.taxa, rows),
            ("marker ids", self.marker_ids, columns),
            ("chromosomes", self.chromosomes, columns),
            ("marker positions", self.positions, columns),
        ):
            if values.ndim != 1 or len(values) != expected:
                raise ValueError(f"{name} must contain {expected} values")

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def marker_map(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SNP": self.marker_ids,
            "Chromosome": self.chromosomes,
            "Position": self.positions,
        })

    @property
    def closed(self) -> bool:
        return self._genotype is None

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        if self._genotype is None:
            raise ValueError("NumPy genotype store is closed")
        block = self._genotype[:, marker_slice]
        if sample_indices is not None:
            block = block[sample_indices]
        result = np.array(block, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result

    def close(self):
        # Blocks own their memory. Release the mmap by dropping its last
        # reference, without accessing NumPy's private _mmap attribute.
        self._genotype = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        self.close()


def write_numpy_genotype(
    path: str | Path,
    genotype: GenotypeWriteSource,
    *,
    marker_chunk_size: int = 1024,
):
    """Write a dependency-free, memory-mapped genotype directory."""
    if marker_chunk_size <= 0:
        raise ValueError("marker_chunk_size must be positive")
    source = as_genotype_write_source(genotype)
    target = Path(path)
    target.mkdir(parents=False, exist_ok=False)
    metadata_path = target / _NUMPY_METADATA
    metadata = _NumpyStoreMetadata(
        schema_version=1,
        complete=False,
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rows, columns = source.shape
    matrix = t.cast(
        np.memmap[tuple[int, int], np.dtype[np.float64]],
        np.lib.format.open_memmap(
            target / _NUMPY_GENOTYPE, mode="w+", dtype=np.float64, shape=(rows, columns)
        ),
    )
    for sample_slice, marker_slice, block in iter_genotype_write_blocks(
        source, marker_chunk_size
    ):
        matrix[sample_slice, marker_slice] = block
    matrix.flush()
    del matrix

    np.save(target / "taxa.npy", np.asarray(source.taxa, dtype=str))
    np.save(
        target / "marker_id.npy",
        source.marker_ids,
    )
    np.save(
        target / "marker_chromosome.npy",
        source.chromosomes,
    )
    np.save(
        target / "marker_position.npy",
        source.positions,
    )
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
