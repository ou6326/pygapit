"""Typed boundary around the dynamic :mod:`rpy2.robjects` API."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast, final, overload

import numpy as np
from numpy.typing import NDArray

from pygapit._typing import FloatMatrix, FloatVector

FloatArray = NDArray[np.float64]
_DLL_DIRECTORY_HANDLES: list[object] = []
RResultT = TypeVar("RResultT", bound="RObject")
RResultT_co = TypeVar("RResultT_co", bound="RObject", covariant=True)


def _configure_pixi_r():
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


class RObject(Protocol):
    """Opaque base type for values owned by the embedded R runtime."""

    @property
    def rclass(self) -> Sequence[str]: ...


class RVector(RObject, Protocol):
    """R vector value."""


class RMatrix(RVector, Protocol):
    """R matrix value."""


class RDataFrame(RObject, Protocol):
    """R data.frame value."""


class RList(RObject, Protocol):
    """Named R list value."""

    def rx2(self, name: str) -> RObject: ...


class RNull(RObject, Protocol):
    """R NULL value."""


class RCallable(Protocol[RResultT_co]):
    """Callable R function with a statically known result type."""

    def __call__(self, *args: object, **kwargs: object) -> RResultT_co: ...


class _RAccessor(Protocol):
    @overload
    def __call__(self, expression: Literal["NULL"]) -> RNull: ...

    @overload
    def __call__(self, expression: str) -> RObject: ...

    @overload
    def __getitem__(
        self, name: Literal["matrix", "colnames<-"]
    ) -> RCallable[RMatrix]: ...

    @overload
    def __getitem__(self, name: Literal["is.null"]) -> RCallable[RVector]: ...

    @overload
    def __getitem__(self, name: str) -> RCallable[RObject]: ...


class _RGlobalEnv(Protocol):
    def __getitem__(self, name: str) -> RCallable[RObject]: ...


class _RObjects(Protocol):
    r: _RAccessor
    globalenv: _RGlobalEnv

    def FloatVector(self, values: list[float]) -> RVector: ...
    def StrVector(self, values: list[str]) -> RVector: ...


@final
class RBridge:
    """Convert typed NumPy values at the Python/R boundary."""

    def __init__(self, robjects: _RObjects, version: str):
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

    @overload
    def source_function(
        self,
        root: Path,
        filename: str,
        symbol: str,
        *,
        returns: type[RResultT],
    ) -> RCallable[RResultT]: ...

    @overload
    def source_function(
        self,
        root: Path,
        filename: str,
        symbol: str,
    ) -> RCallable[RObject]: ...

    def source_function(
        self,
        root: Path,
        filename: str,
        symbol: str,
        *,
        returns: type[RObject] | None = None,
    ) -> RCallable[RObject]:
        """Source one GAPIT file and return a function defined by it."""
        self.source(root, filename)
        return self._robjects.globalenv[symbol]

    def source(self, root: Path, filename: str):
        """Source one GAPIT file into the shared R global environment."""
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundled GAPIT source file is missing: {path}")
        self._robjects.r["source"](str(path))

    def source_for_regular_matrices(
        self,
        root: Path,
        filename: str,
        *,
        replacements: Mapping[str, str] | None = None,
    ):
        """Source GAPIT code with big.matrix probes fixed to false.

        GAPIT uses ``bigmemory::is.big.matrix`` even when callers supply an
        ordinary matrix.  Alignment tests do not need the optional bigmemory
        package, so this in-memory variant preserves the regular-matrix branch
        without modifying the pinned checkout. Optional replacements support
        narrowly characterized divergences and must each match exactly once.
        """
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundled GAPIT source file is missing: {path}")
        source = path.read_text(encoding="utf-8")
        probe = "bigmemory::is.big.matrix"
        if probe not in source:
            raise ValueError(f"GAPIT source does not contain {probe}: {path}")
        source = source.replace(probe, ".pygapit_is_big_matrix")
        if replacements is not None:
            for original, replacement in replacements.items():
                match_count = source.count(original)
                if match_count != 1:
                    raise ValueError(
                        "GAPIT source replacement must match exactly once; "
                        f"found {match_count} matches for {original!r}"
                    )
                source = source.replace(original, replacement)
        self.evaluate(".pygapit_is_big_matrix <- function(x) FALSE")
        self.evaluate(source)

    @overload
    def evaluate(self, expression: Literal["NULL"]) -> RNull: ...

    @overload
    def evaluate(self, expression: str) -> RObject: ...

    def evaluate(self, expression: str) -> RObject:
        """Evaluate an R expression and return its dynamic value."""
        return self._robjects.r(expression)

    @overload
    def function(
        self, expression: Literal["as.data.frame"]
    ) -> RCallable[RDataFrame]: ...

    @overload
    def function(self, expression: Literal["Blink", "FarmCPU"]) -> RCallable[RList]: ...

    @overload
    def function(
        self, expression: Literal["stats::p.adjust"]
    ) -> RCallable[RVector]: ...

    @overload
    def function(
        self,
        expression: str,
        *,
        returns: type[RResultT],
    ) -> RCallable[RResultT]: ...

    @overload
    def function(self, expression: str) -> RCallable[RObject]: ...

    def function(
        self,
        expression: str,
        *,
        returns: type[RObject] | None = None,
    ) -> RCallable[RObject]:
        """Resolve an R function expression such as ``stats::p.adjust``."""
        return cast(RCallable[RObject], cast(object, self._robjects.r(expression)))

    def float_vector(self, values: FloatVector) -> RVector:
        """Create an R numeric vector without implicit NumPy conversion."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        return self._robjects.FloatVector(flat.tolist())

    def matrix(
        self, values: FloatMatrix, *, column_names: Sequence[str] | None = None
    ) -> RMatrix:
        """Create an R matrix preserving the NumPy row/column arrangement."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional matrix, got shape {array.shape}"
            )
        vector = self._robjects.FloatVector(array.ravel(order="F").tolist())
        result = self._robjects.r["matrix"](
            vector,
            nrow=array.shape[0],
            ncol=array.shape[1],
        )
        if column_names is not None:
            if len(column_names) != array.shape[1]:
                raise ValueError("column_names must match the matrix column count")
            result = self._robjects.r["colnames<-"](
                result, self._robjects.StrVector(list(column_names))
            )
        return result

    @staticmethod
    def float_array(value: RObject) -> FloatArray:
        """Convert an R numeric vector or matrix to a float64 NumPy array."""
        return np.asarray(value, dtype=np.float64)

    def is_null(self, value: RObject) -> bool:
        """Return whether an R value is ``NULL`` without coercing it to NumPy."""
        result = np.asarray(self._robjects.r["is.null"](value), dtype=np.bool_)
        return bool(result[0])

    @staticmethod
    def component(value: RObject, name: str) -> RObject:
        """Extract a named component from an R list result."""
        return cast(RList, value).rx2(name)
