"""Narrow structural types for the Zarr APIs used by genotype storage."""

from __future__ import annotations

import typing as t
from collections.abc import MutableMapping
from types import ModuleType

import numpy as np
import numpy.typing as npt

from .._typing import IntVector

type ZarrIndex = slice | tuple[slice, slice] | tuple[IntVector, slice]


class ZarrIndexer(t.Protocol):
    def __getitem__(self, key: tuple[IntVector, slice]) -> npt.NDArray[np.generic]: ...


class ZarrArray(t.Protocol):
    shape: tuple[int, ...]
    ndim: int
    dtype: np.dtype[np.generic]
    chunks: tuple[int, ...]

    @property
    def oindex(self) -> ZarrIndexer: ...

    def __getitem__(self, key: ZarrIndex) -> npt.NDArray[np.generic]: ...

    def __setitem__(self, key: ZarrIndex, value: npt.ArrayLike) -> None: ...


class ZarrGroup(t.Protocol):
    attrs: MutableMapping[str, object]

    def get(self, name: str) -> ZarrArray | None: ...

    def __delitem__(self, name: str) -> None: ...

    def create_array(
        self,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype[np.generic],
        chunks: tuple[int, ...],
    ) -> ZarrArray: ...


class ZarrModule(t.Protocol):
    Array: type[ZarrArray]

    def open_group(
        self,
        store: str,
        *,
        mode: t.Literal["r", "r+", "w-"],
        zarr_format: int | None = None,
    ) -> ZarrGroup: ...


def as_zarr_module(module: ModuleType) -> ZarrModule:
    """Type the partly annotated optional dependency at its import boundary."""
    return t.cast(ZarrModule, t.cast(object, module))
