"""Chunk-readable genotype storage with transparent optional backends."""

from __future__ import annotations

import typing as t
from pathlib import Path

from ._genotype_store import (
    ArrayGenotypeStore,
    GenotypeStore,
    GenotypeView,
    LabeledGenotypeStore,
    MarkerChunkedGenotypeStore,
    as_genotype_store,
)
from ._storage_hdf5 import HDF5GenotypeStore, write_hdf5_genotype
from ._storage_numpy import NumpyGenotypeStore, write_numpy_genotype
from ._storage_zarr import ZarrGenotypeStore, write_zarr_genotype

if t.TYPE_CHECKING:
    from .formats import GenotypeData


StorageBackend: t.TypeAlias = t.Literal["auto", "numpy", "hdf5", "zarr"]
_BACKEND_ERROR = "backend must be 'auto', 'numpy', 'hdf5', or 'zarr'"

__all__ = [
    "ArrayGenotypeStore",
    "GenotypeStore",
    "GenotypeView",
    "HDF5GenotypeStore",
    "LabeledGenotypeStore",
    "MarkerChunkedGenotypeStore",
    "NumpyGenotypeStore",
    "StorageBackend",
    "ZarrGenotypeStore",
    "as_genotype_store",
    "open_genotype_store",
    "write_genotype_store",
    "write_hdf5_genotype",
    "write_numpy_genotype",
    "write_zarr_genotype",
]


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
    path: str | Path, *, backend: t.Literal["zarr"]
) -> ZarrGenotypeStore: ...


@t.overload
def open_genotype_store(
    path: str | Path, *, backend: StorageBackend = "auto"
) -> NumpyGenotypeStore | HDF5GenotypeStore | ZarrGenotypeStore: ...


def open_genotype_store(
    path: str | Path,
    *,
    backend: StorageBackend = "auto",
) -> NumpyGenotypeStore | HDF5GenotypeStore | ZarrGenotypeStore:
    """Open a disk-backed genotype, detecting its backend by default."""
    target = Path(path)
    if backend == "auto":
        is_zarr = target.is_dir() and (
            (target / ".zgroup").is_file() or (target / "zarr.json").is_file()
        )
        backend = "zarr" if is_zarr else "numpy" if target.is_dir() else "hdf5"
    try:
        reader = {
            "numpy": NumpyGenotypeStore,
            "hdf5": HDF5GenotypeStore,
            "zarr": ZarrGenotypeStore,
        }[backend]
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

    ``auto`` respects explicit HDF5 and Zarr filename suffixes. Otherwise it
    prefers HDF5, then Zarr, and silently falls back to dependency-free NumPy.
    """
    target = Path(path)
    if backend != "auto":
        try:
            writer = {
                "numpy": write_numpy_genotype,
                "hdf5": write_hdf5_genotype,
                "zarr": write_zarr_genotype,
            }[backend]
        except KeyError:
            raise ValueError(_BACKEND_ERROR) from None
        writer(target, genotype, marker_chunk_size=marker_chunk_size)
        return
    if target.suffix.lower() in {".h5", ".hdf5"}:
        write_hdf5_genotype(target, genotype, marker_chunk_size=marker_chunk_size)
        return
    if target.suffix.lower() == ".zarr":
        write_zarr_genotype(target, genotype, marker_chunk_size=marker_chunk_size)
        return

    try:
        import h5py
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise
        try:
            import zarr
        except ModuleNotFoundError as zarr_exc:
            if zarr_exc.name != "zarr":
                raise
            write_numpy_genotype(
                target,
                genotype,
                marker_chunk_size=marker_chunk_size,
            )
        else:
            _ = zarr
            write_zarr_genotype(
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
