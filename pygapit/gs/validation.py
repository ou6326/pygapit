"""Fold-local genomic prediction with explicit out-of-fold results.

RR-BLUP uses y = intercept + Z b + e, K = Z Z' / m and lambda = m * delta.
See Endelman (2011), doi:10.3835/plantgenome2011.08.0024. Marker means
and variance components are learned on training samples only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    Matrix,
    Vector,
    as_float_matrix,
    as_float_vector,
    readonly_copy,
    require_row_count,
    require_square,
)
from ..stats.emma import emma_remle


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
    result = as_float_vector(y, name="phenotype")
    if len(result) < 3 or not np.all(np.isfinite(result)):
        raise ValueError("phenotype must contain at least three finite observations")
    return result


def _folds(
    n: int,
    n_folds: object,
    seed: int | None,
    groups: Vector | None,
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
            if labels.shape != (n,) or labels.dtype.kind not in "iuUS":
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


def _genotypes(genotype: Matrix, n: int) -> FloatMatrix:
    result = as_float_matrix(genotype, name="genotype matrix")
    require_row_count(result, n, name="genotype matrix")
    if result.shape[1] == 0 or np.any(np.isinf(result)):
        raise ValueError("genotype must contain markers and cannot contain infinity")
    return result


def _ridge_fit(
    y: FloatVector,
    train: FloatMatrix,
    test: FloatMatrix,
    lambda_: float | None,
) -> tuple[FloatVector, FloatVector, float]:
    """Return training GEBV, test phenotype prediction, and marker penalty."""
    if lambda_ is not None and (not np.isfinite(lambda_) or lambda_ <= 0):
        raise ValueError("lambda_ must be finite and positive")
    counts = np.sum(~np.isnan(train), axis=0)
    means = np.divide(
        np.nansum(train, axis=0), counts, out=np.zeros(train.shape[1]), where=counts > 0
    )
    z = np.where(np.isnan(train), means, train) - means
    z_test = np.where(np.isnan(test), means, test) - means
    # A marker wholly unobserved in training carries no predictive information.
    z_test[:, counts == 0] = 0.0
    n, m = z.shape
    intercept = float(np.mean(y))
    centered_y = y - intercept
    if lambda_ is None:
        fit = emma_remle(y, np.ones((n, 1)), z @ z.T / m)
        penalty = fit.delta * m
    else:
        penalty = lambda_
    if m <= n:
        effects = np.linalg.solve(z.T @ z + penalty * np.eye(m), z.T @ centered_y)
    else:
        dual = np.linalg.solve(z @ z.T + penalty * np.eye(n), centered_y)
        effects = z.T @ dual
    return z @ effects, z_test @ effects + intercept, penalty


def cross_validate_rrblup(
    phenotype: Vector,
    genotype: Matrix,
    *,
    n_folds: int = 5,
    seed: int | None = None,
    groups: Vector | None = None,
    fold_ids: Vector | None = None,
    lambda_: float | None = None,
) -> PredictionCVResult:
    """Validate RR-BLUP with fold-local imputation, centering and REML.

    With seed=None, ungrouped folds are contiguous and deterministic. A seed
    shuffles samples (or group tie order). Groups never cross folds. Explicit
    integer fold_ids override n_folds and cannot be combined with groups/seed.
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
    groups: Vector | None = None,
    fold_ids: Vector | None = None,
) -> PredictionCVResult:
    """Validate intercept-only gBLUP, estimating REML and GLS mean per fold.

    Kinship must be a finite symmetric positive-semidefinite matrix built
    without phenotype-based selection. It is treated as a fixed input: this
    API cannot undo leakage introduced when constructing a supplied kinship.
    Split options follow cross_validate_rrblup. No full-data fallback is used.
    """
    y = _phenotype(phenotype)
    k = as_float_matrix(kinship, name="kinship matrix")
    require_square(k, size=len(y), name="kinship matrix")
    if not np.all(np.isfinite(k)) or not np.allclose(k, k.T, rtol=1e-10, atol=1e-12):
        raise ValueError("kinship must be finite and symmetric")
    ids = _folds(len(y), n_folds, seed, groups, fold_ids)
    predictions = np.empty(len(y))
    penalties = np.empty(int(ids.max()) + 1)
    for fold in range(len(penalties)):
        test = ids == fold
        yt = y[~test]
        kt = k[np.ix_(~test, ~test)]
        fit = emma_remle(yt, np.ones((len(yt), 1)), kt)
        penalties[fold] = fit.delta
        solved = np.linalg.solve(
            kt + fit.delta * np.eye(len(yt)), np.column_stack((yt, np.ones(len(yt))))
        )
        mean = float(np.sum(solved[:, 0]) / np.sum(solved[:, 1]))
        predictions[test] = mean + k[np.ix_(test, ~test)] @ (
            solved[:, 0] - mean * solved[:, 1]
        )
    return _result(y, predictions, ids, penalties, "gBLUP")
