"""Shared storage API types without backend imports."""

from typing import Literal, TypeAlias

StorageBackend: TypeAlias = Literal["auto", "numpy", "hdf5", "zarr"]
