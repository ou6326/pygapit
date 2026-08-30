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


def as_float_vector(values: object) -> FloatVector:
    """Convert values to a one-dimensional float64 array."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"Expected a one-dimensional array, got {result.ndim}D")
    return result


def as_str_vector(values: object) -> StrVector:
    """Convert values to a one-dimensional Unicode array."""
    result = np.asarray(values, dtype=str)
    if result.ndim != 1:
        raise ValueError(f"Expected a one-dimensional array, got {result.ndim}D")
    return result
