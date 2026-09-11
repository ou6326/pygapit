"""Chunk-readable genotype storage with transparent optional backends."""

from __future__ import annotations

import typing as t
from pathlib import Path

from ._genotype_store import (
    ArrayGenotypeStore,
    GenotypeStore,
    GenotypeView,
    LabeledGenotypeStore,
    as_genotype_store,
)
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
    "LabeledGenotypeStore",
    "NumpyGenotypeStore",
    "StorageBackend",
    "as_genotype_store",
    "open_genotype_store",
    "write_genotype_store",
    "write_hdf5_genotype",
    "write_numpy_genotype",
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
