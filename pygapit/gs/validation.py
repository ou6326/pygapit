"""Fold-local genomic prediction with explicit out-of-fold results.

RR-BLUP uses y = intercept + Z b + e, K = Z Z' / m and lambda = m * delta.
See Endelman (2011), doi:10.3835/plantgenome2011.08.0024. RR-BLUP learns
marker means and variance components on training samples only. gBLUP fits
each training fold over a fixed, externally supplied kinship.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    Matrix,
    Vector,
    as_float_matrix,
    readonly_copy,
    require_square,
)
from ..stats.emma import emma_remle
from ._ridge import _finite_phenotype, _genotypes, _ridge_fit

__all__ = [
    "PredictionCVResult",
    "_ridge_fit",
    "cross_validate_gblup",
    "cross_validate_rrblup",
]


@dataclass(frozen=True, slots=True)
class PredictionCVResult:
    """Phenotype predictions in input order; fold IDs are normalized to 0..k-1.

    Pearson correlations are NaN for constant vectors or singleton test folds.
    ``regularization`` holds the fitted lambda (RR-BLUP) or delta (gBLUP)
    for each fold. No full-data fit is used to produce these predictions.
    """

    observed: FloatVector
    predictions: FloatVector
    fold_ids: IntVector
    pearson_r: float
    rmse: float
    fold_pearson_r: FloatVector
    fold_rmse: FloatVector
    regularization: FloatVector
    method: str

    def __post_init__(self) -> None:
        for field in (
            "observed",
            "predictions",
            "fold_ids",
            "fold_pearson_r",
            "fold_rmse",
            "regularization",
        ):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def _phenotype(y: Vector) -> FloatVector:
    return _finite_phenotype(y, minimum_samples=3)


def _folds(
    n: int,
    n_folds: object,
    seed: int | None,
    groups: object | None,
    fold_ids: Vector | None,
) -> IntVector:
    if fold_ids is not None:
        if groups is not None or seed is not None:
            raise ValueError("fold_ids cannot be combined with groups or seed")
        labels = np.asarray(fold_ids)
        if labels.shape != (n,) or labels.dtype.kind not in "iu":
            raise ValueError(
                "fold_ids must be an integer vector with one ID per sample"
            )
        _, inverse = np.unique(labels, return_inverse=True)
        result = np.asarray(inverse, dtype=int)
    else:
        if (
            isinstance(n_folds, bool)
            or not isinstance(n_folds, int)
            or not 2 <= n_folds <= n
        ):
            raise ValueError(
                "n_folds must be an integer between 2 and the sample count"
            )
        result = np.empty(n, dtype=int)
        rng = np.random.default_rng(seed)
        if groups is None:
            order = np.arange(n) if seed is None else rng.permutation(n)
            for fold, test in enumerate(np.array_split(order, n_folds)):
                result[test] = fold
        else:
            labels = np.asarray(groups)
            if labels.shape == (n,) and labels.dtype.kind == "O":
                values = t.cast(list[object], labels.tolist())
                string_ids = all(isinstance(value, str) for value in values)
                integer_ids = all(
                    isinstance(value, (int, np.integer))
                    and not isinstance(value, (bool, np.bool_))
                    for value in values
                )
                if not (string_ids or integer_ids):
                    raise ValueError(
                        "groups must contain non-missing strings or integers of one kind"
                    )
                # Keep integers as objects to avoid overflow or rounding large IDs.
                # np.unique can sort homogeneous object integers without coercion.
                if string_ids:
                    labels = np.asarray(values, dtype=str)
            elif labels.shape != (n,) or labels.dtype.kind not in "iuUS":
                raise ValueError(
                    "groups must be an integer or string vector, one per sample"
                )
            _, inverse, counts = np.unique(
                labels, return_inverse=True, return_counts=True
            )
            if len(counts) < n_folds:
                raise ValueError("groups must contain at least n_folds distinct groups")
            order = (
                np.arange(len(counts)) if seed is None else rng.permutation(len(counts))
            )
            order = order[np.argsort(-counts[order], kind="stable")]
            sizes = np.zeros(n_folds, dtype=int)
            for group in order:
                fold = int(np.argmin(sizes))
                result[inverse == group] = fold
                sizes[fold] += counts[group]
    sizes = np.bincount(result)
    if len(sizes) < 2 or np.any(n - sizes < 2):
        raise ValueError(
            "CV requires at least two folds and two training samples per fold"
        )
    return result


def _correlation(y: FloatVector, pred: FloatVector) -> float:
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(pred) == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def _result(
    y: FloatVector,
    predictions: FloatVector,
    ids: IntVector,
    penalties: FloatVector,
    method: str,
) -> PredictionCVResult:
    fold_r = np.array([
        _correlation(y[ids == f], predictions[ids == f]) for f in range(len(penalties))
    ])
    fold_rmse = np.array([
        np.sqrt(np.mean((y[ids == f] - predictions[ids == f]) ** 2))
        for f in range(len(penalties))
    ])
    return PredictionCVResult(
        y,
        predictions,
        ids,
        _correlation(y, predictions),
        float(np.sqrt(np.mean((y - predictions) ** 2))),
        fold_r,
        fold_rmse,
        penalties,
        method,
    )


def cross_validate_rrblup(
    phenotype: Vector,
    genotype: Matrix,
    *,
    n_folds: int = 5,
    seed: int | None = None,
    groups: object | None = None,
    fold_ids: Vector | None = None,
    lambda_: float | None = None,
) -> PredictionCVResult:
    """Validate RR-BLUP with fold-local imputation, centering and REML.

    With seed=None, ungrouped folds are contiguous and deterministic. A seed
    shuffles samples (or group tie order). Groups never cross folds.
    Groups may be NumPy arrays or Pandas Series of non-missing strings/integers.
    Explicit integer fold_ids override n_folds and cannot be combined with groups/seed.
    A supplied positive lambda is fixed by the caller; otherwise each training
    fold estimates its own value. Missing phenotypes must be removed by callers.
    """
    y = _phenotype(phenotype)
    z = _genotypes(genotype, len(y))
    ids = _folds(len(y), n_folds, seed, groups, fold_ids)
    penalties = np.empty(int(ids.max()) + 1)
    predictions = np.empty(len(y))
    for fold in range(len(penalties)):
        test = ids == fold
        _, predictions[test], penalties[fold] = _ridge_fit(
            y[~test], z[~test], z[test], lambda_
        )
    return _result(y, predictions, ids, penalties, "RR-BLUP")


def cross_validate_gblup(
    phenotype: Vector,
    kinship: Matrix,
    *,
    n_folds: int = 5,
    seed: int | None = None,
    groups: object | None = None,
    fold_ids: Vector | None = None,
) -> PredictionCVResult:
    """Validate intercept-only gBLUP, estimating REML and GLS mean per fold.

    Kinship must be a finite symmetric positive-semidefinite matrix built
    without phenotype-based selection. It is treated as a fixed input: this
    API cannot undo leakage introduced when constructing a supplied kinship.
    Full-dataset genotype centering in that input is transductive preprocessing,
    not strict training-only preprocessing, even without phenotype leakage.
    Split options follow cross_validate_rrblup. No full-data fallback is used.
    The full kinship is checked once using a symmetric copy; negative
    eigenvalues within eps * n * max(spectral_radius, 1) are treated as roundoff.
    """
    y = _phenotype(phenotype)
    k = as_float_matrix(kinship, name="kinship matrix")
    require_square(k, size=len(y), name="kinship matrix")
    if not np.all(np.isfinite(k)) or not np.allclose(k, k.T, rtol=1e-10, atol=1e-12):
        raise ValueError("kinship must be finite and symmetric")
    # Check the full matrix once: acceptable principal submatrices alone do not
    # guarantee a valid covariance between training and test individuals.
    # Use the same symmetric copy for validation and every subsequent solve.
    k = k * 0.5 + k.T * 0.5
    eigenvalues = np.linalg.eigvalsh(k)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = np.finfo(float).eps * len(y) * scale
    if eigenvalues[0] < -tolerance:
        raise ValueError("kinship must be positive semidefinite")
    ids = _folds(len(y), n_folds, seed, groups, fold_ids)
    predictions = np.empty(len(y))
    penalties = np.empty(int(ids.max()) + 1)
    for fold in range(len(penalties)):
        test = ids == fold
        yt = y[~test]
        kt = k[np.ix_(~test, ~test)]
        fit = emma_remle(yt, np.ones((len(yt), 1)), kt)
        penalties[fold] = fit.delta
        covariance = kt.copy()
        covariance.flat[:: len(yt) + 1] += fit.delta
        solve_rhs: FloatMatrix = np.column_stack((yt, np.ones(len(yt))))
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            solved: FloatMatrix = cho_solve(
                factor,
                solve_rhs,
                check_finite=False,
            )
        except np.linalg.LinAlgError:
            solved = np.linalg.solve(covariance, solve_rhs)
        mean = float(np.sum(solved[:, 0]) / np.sum(solved[:, 1]))
        predictions[test] = mean + k[np.ix_(test, ~test)] @ (
            solved[:, 0] - mean * solved[:, 1]
        )
    return _result(y, predictions, ids, penalties, "gBLUP")
