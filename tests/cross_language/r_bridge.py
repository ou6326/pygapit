"""Typed boundary around the dynamic :mod:`rpy2.robjects` API."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast, final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_DLL_DIRECTORY_HANDLES: list[object] = []


def _configure_pixi_r() -> None:
    """Point embedded rpy2 at the R installation in the active Pixi prefix."""
    r_home = Path(sys.prefix) / "Lib" / "R"
    if not r_home.is_dir():
        return
    os.environ["R_HOME"] = str(r_home)
    if sys.platform == "win32":
        r_bin = r_home / "bin" / "x64"
        os.environ["PATH"] = f"{r_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(r_bin)))


class RUnavailableError(RuntimeError):
    """Raised when rpy2 cannot connect to a working R runtime."""


class _RCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _RAccessor(_RCallable, Protocol):
    def __getitem__(self, name: str) -> _RCallable: ...


class _RGlobalEnv(Protocol):
    def __getitem__(self, name: str) -> _RCallable: ...


class _RList(Protocol):
    def rx2(self, name: str) -> object: ...


class _RObjects(Protocol):
    r: _RAccessor
    globalenv: _RGlobalEnv

    def FloatVector(self, values: list[float]) -> object: ...
    def StrVector(self, values: list[str]) -> object: ...


@final
class RBridge:
    """Convert typed NumPy values at the Python/R boundary."""

    def __init__(self, robjects: _RObjects, version: str) -> None:
        self._robjects = robjects
        self.version = version

    @classmethod
    def connect(cls) -> RBridge:
        """Import rpy2 and verify that the configured R runtime evaluates code."""
        try:
            _configure_pixi_r()
            module = import_module("rpy2.robjects")
            robjects = cast(_RObjects, cast(object, module))
            version_value = robjects.r("R.version.string")
            version_array = np.asarray(version_value, dtype=np.str_)
            return cls(robjects, str(version_array[0]))
        except Exception as exc:
            raise RUnavailableError(
                f"R runtime is unavailable through rpy2: {exc}"
            ) from exc

    def source_function(self, root: Path, filename: str, symbol: str) -> _RCallable:
        """Source one GAPIT file and return a function defined by it."""
        self.source(root, filename)
        return self._robjects.globalenv[symbol]

    def source(self, root: Path, filename: str) -> None:
        """Source one GAPIT file into the shared R global environment."""
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundled GAPIT source file is missing: {path}")
        self._robjects.r["source"](str(path))

    def source_for_regular_matrices(self, root: Path, filename: str) -> None:
        """Source GAPIT code with big.matrix probes fixed to false.

        GAPIT uses ``bigmemory::is.big.matrix`` even when callers supply an
        ordinary matrix.  Alignment tests do not need the optional bigmemory
        package, so this in-memory variant preserves the regular-matrix branch
        without modifying the pinned checkout.
        """
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundled GAPIT source file is missing: {path}")
        source = path.read_text(encoding="utf-8")
        probe = "bigmemory::is.big.matrix"
        if probe not in source:
            raise ValueError(f"GAPIT source does not contain {probe}: {path}")
        source = source.replace(probe, ".pygapit_is_big_matrix")
        self.evaluate(".pygapit_is_big_matrix <- function(x) FALSE")
        self.evaluate(source)

    def evaluate(self, expression: str) -> object:
        """Evaluate an R expression and return its dynamic value."""
        return self._robjects.r(expression)

    def function(self, expression: str) -> _RCallable:
        """Resolve an R function expression such as ``stats::p.adjust``."""
        return cast(_RCallable, self._robjects.r(expression))

    def float_vector(self, values: FloatArray) -> object:
        """Create an R numeric vector without implicit NumPy conversion."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        return self._robjects.FloatVector(flat.tolist())

    def matrix(
        self, values: FloatArray, *, column_names: Sequence[str] | None = None
    ) -> object:
        """Create an R matrix preserving the NumPy row/column arrangement."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional matrix, got shape {array.shape}"
            )
        vector = self._robjects.FloatVector(array.ravel(order="F").tolist())
        result = self._robjects.r["matrix"](
            vector,
            nrow=int(array.shape[0]),
            ncol=int(array.shape[1]),
        )
        if column_names is not None:
            if len(column_names) != array.shape[1]:
                raise ValueError("column_names must match the matrix column count")
            result = self._robjects.r["colnames<-"](
                result, self._robjects.StrVector(list(column_names))
            )
        return result

    @staticmethod
    def float_array(value: object) -> FloatArray:
        """Convert an R numeric vector or matrix to a float64 NumPy array."""
        return np.asarray(value, dtype=np.float64)

    def is_null(self, value: object) -> bool:
        """Return whether an R value is ``NULL`` without coercing it to NumPy."""
        result = np.asarray(self._robjects.r["is.null"](value), dtype=np.bool_)
        return bool(result[0])

    @staticmethod
    def component(value: object, name: str) -> object:
        """Extract a named component from an R list result."""
        return cast(_RList, value).rx2(name)
