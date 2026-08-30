"""Shared NumPy array types used across PyGAPIT."""

from __future__ import annotations

import typing as t

import numpy as np

Array: t.TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
Vector: t.TypeAlias = np.ndarray[tuple[int], np.dtype[np.generic]]
Matrix: t.TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.generic]]

NumericVector: t.TypeAlias = np.ndarray[tuple[int], np.dtype[np.number]]

FloatVector: t.TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatMatrix: t.TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

IntVector: t.TypeAlias = np.ndarray[tuple[int], np.dtype[np.int_]]
BoolVector: t.TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
StrVector: t.TypeAlias = np.ndarray[tuple[int], np.dtype[np.str_]]

LabelVector: t.TypeAlias = StrVector | NumericVector
ArrayT = t.TypeVar("ArrayT", bound=Array)


def readonly_copy(values: ArrayT) -> ArrayT:
    """Return an independent NumPy array whose contents cannot be mutated."""
    result = values.copy()
    result.setflags(write=False)
    return result


def as_float_vector(values: object, *, name: str = "array") -> FloatVector:
    """Convert values to a one-dimensional float64 array."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {result.ndim}D")
    return result


def as_float_matrix(values: object, *, name: str = "array") -> FloatMatrix:
    """Convert values to a two-dimensional float64 array."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {result.ndim}D")
    return result


def as_str_vector(values: object, *, name: str = "array") -> StrVector:
    """Convert values to a one-dimensional Unicode array."""
    result = np.asarray(values, dtype=str)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {result.ndim}D")
    return result


def require_length(values: Vector, expected: int, *, name: str) -> None:
    """Require a vector to have the expected number of elements."""
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}; got {len(values)}")


def require_row_count(values: Matrix, expected: int, *, name: str) -> None:
    """Require a matrix to have the expected number of rows."""
    if values.shape[0] != expected:
        raise ValueError(f"{name} must have {expected} rows; got {values.shape[0]}")


def require_square(values: Matrix, *, name: str, size: int | None = None) -> None:
    """Require a square matrix, optionally with an exact side length."""
    rows, columns = values.shape
    if rows != columns:
        raise ValueError(f"{name} must be square; got shape {values.shape}")
    if size is not None and rows != size:
        raise ValueError(f"{name} must have shape ({size}, {size}); got {values.shape}")
