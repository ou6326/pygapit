"""Shared storage API types without backend imports."""

from typing import Literal

type StorageBackend = Literal["auto", "numpy", "hdf5", "zarr"]
