"""Optional Zarr implementation for the generic genotype storage API."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, NoReturn, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .._typing import FloatMatrix, FloatVector, IntVector, StrVector
from ._genotype_store import normalize_sample_selection
from ._storage_source import as_genotype_write_source, iter_genotype_write_blocks
from ._zarr_typing import ZarrArray, ZarrGroup, ZarrModule, as_zarr_module

if TYPE_CHECKING:
    from ._storage_source import GenotypeWriteSource

_ZARR_DATASET = "genotype"


def _require_zarr() -> ZarrModule:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        if exc.name != "zarr":
            raise
        _raise_missing_zarr(exc)
    return as_zarr_module(zarr)


class ZarrGenotypeStore:
    """Read a pyGAPIT Zarr genotype without loading the complete matrix."""

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        zarr = _require_zarr()
        group = zarr.open_group(str(self.path), mode="r")
        complete = group.attrs.get("complete")
        if type(complete) is not bool:
            raise ValueError("store complete must be a boolean")
        if not complete:
            raise ValueError("genotype import is incomplete")
        schema = group.attrs.get("schema_version")
        if type(schema) is not int:
            raise ValueError("store schema_version must be an integer")
        if schema != 1:
            raise ValueError("unsupported genotype schema")
        genotype = group.get(_ZARR_DATASET)
        if genotype is None or not isinstance(genotype, zarr.Array):
            raise ValueError("Zarr genotype must be an array")
        if genotype.ndim != 2 or genotype.dtype != np.dtype(np.float64):
            raise ValueError("Zarr genotype array must be two-dimensional float64")
        self._group: ZarrGroup | None = group
        self._genotype: ZarrArray | None = genotype
        shape = genotype.shape
        self._shape: tuple[int, int] = shape[0], shape[1]
        self.taxa: StrVector = _read_strings(zarr, group, "taxa")
        self.marker_ids: StrVector = _read_strings(zarr, group, "markers/id")
        self.chromosomes: StrVector = _read_strings(zarr, group, "markers/chromosome")
        positions = group.get("markers/position")
        if positions is None or not isinstance(positions, zarr.Array):
            raise ValueError("Zarr marker positions must be an array")
        self.positions: FloatVector = np.asarray(positions[:], dtype=np.float64)
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
    def marker_chunk_size(self) -> int | None:
        return None if self._genotype is None else self._genotype.chunks[1]

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
        genotype = self._genotype
        if genotype is None:
            raise ValueError("Zarr genotype store is closed")
        selection = normalize_sample_selection(sample_indices, self.shape[0])
        if selection is None:
            selected = genotype[:, marker_slice]
        elif isinstance(selection, slice):
            start, stop, step = selection.indices(self.shape[0])
            if step > 0:
                selected = genotype[slice(start, stop, step), marker_slice]
            else:
                requested: IntVector = np.arange(start, stop, step, dtype=np.int_)
                unique, inverse = np.unique(requested, return_inverse=True)
                selected = genotype.oindex[unique, marker_slice][inverse]
        else:
            unique, inverse = np.unique(selection, return_inverse=True)
            selected = genotype.oindex[unique, marker_slice][inverse]
        result = np.array(selected, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result

    def close(self) -> None:
        self._genotype = None
        self._group = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def write_zarr_genotype(
    path: str | Path,
    genotype: GenotypeWriteSource,
    *,
    marker_chunk_size: int = 1024,
) -> None:
    """Write validated genotype data as a Zarr format 2 directory."""
    if marker_chunk_size <= 0:
        raise ValueError("marker_chunk_size must be positive")
    source = as_genotype_write_source(genotype)
    zarr = _require_zarr()
    rows, columns = source.shape
    marker_chunk_size = min(marker_chunk_size, columns)
    sample_chunk_size = min(rows, max(1, (8 * 1024**2) // (8 * marker_chunk_size)))
    group = zarr.open_group(str(Path(path)), mode="w-", zarr_format=2)
    group.attrs["schema_version"] = 1
    group.attrs["complete"] = False
    matrix = _create_array(
        group,
        _ZARR_DATASET,
        shape=(rows, columns),
        dtype=np.dtype(np.float64),
        chunks=(sample_chunk_size, marker_chunk_size),
    )
    for sample_slice, marker_slice, block in iter_genotype_write_blocks(
        source, marker_chunk_size
    ):
        matrix[sample_slice, marker_slice] = block
    _write_strings(group, "taxa", source.taxa)
    _write_strings(group, "markers/id", source.marker_ids)
    _write_strings(group, "markers/chromosome", source.chromosomes)
    positions = source.positions
    position_array = _create_array(
        group,
        "markers/position",
        shape=positions.shape,
        dtype=np.dtype(np.float64),
        chunks=(max(1, min(marker_chunk_size, columns)),),
    )
    position_array[:] = positions
    group.attrs["complete"] = True


def _create_array(
    group: ZarrGroup,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
    chunks: tuple[int, ...],
) -> ZarrArray:
    return group.create_array(name, shape=shape, dtype=dtype, chunks=chunks)


def _write_strings(
    group: ZarrGroup,
    name: str,
    values: npt.NDArray[np.str_],
) -> None:
    encoded = np.asarray([value.encode("utf-8") for value in values], dtype=np.bytes_)
    array = _create_array(
        group,
        name,
        shape=encoded.shape,
        dtype=encoded.dtype,
        chunks=(max(1, len(encoded)),),
    )
    array[:] = encoded


def _read_strings(zarr: ZarrModule, group: ZarrGroup, name: str) -> StrVector:
    array = group.get(name)
    if array is None or not isinstance(array, zarr.Array):
        raise ValueError(f"Zarr {name} must be an array")
    raw = np.asarray(array[:])
    if raw.ndim != 1 or raw.dtype.kind != "S":
        raise ValueError(f"Zarr {name} must contain UTF-8 bytes")
    return np.asarray([value.decode("utf-8") for value in raw.tolist()], dtype=str)


def _raise_missing_zarr(exc: ModuleNotFoundError) -> NoReturn:
    raise ImportError(
        "Zarr genotype storage requires the optional 'bigdata' feature; "
        "install pygapit-ng[bigdata], or use backend='numpy'."
    ) from exc
