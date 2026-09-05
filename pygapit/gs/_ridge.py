"""Shared RR-BLUP validation and fitting internals."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .._typing import (
    FloatMatrix,
    FloatVector,
    Matrix,
    Vector,
    as_float_matrix,
    as_float_vector,
    require_row_count,
)
from ..stats.emma import emma_remle

__all__ = ["_finite_phenotype", "_genotypes", "_ridge_components", "_ridge_fit"]


def _finite_phenotype(y: Vector, *, minimum_samples: int) -> FloatVector:
    result = as_float_vector(y, name="phenotype")
    if len(result) < minimum_samples or not np.all(np.isfinite(result)):
        number = "two" if minimum_samples == 2 else "three"
        raise ValueError(
            f"phenotype must contain at least {number} finite observations"
        )
    return result


def _genotypes(genotype: Matrix, n: int) -> FloatMatrix:
    result = as_float_matrix(genotype, name="genotype matrix")
    require_row_count(result, n, name="genotype matrix")
    if result.shape[1] == 0 or np.any(np.isinf(result)):
        raise ValueError("genotype must contain markers and cannot contain infinity")
    return result


def _solve_penalized_gram(
    gram: FloatMatrix,
    rhs: FloatVector,
    penalty: float,
) -> FloatVector:
    """Solve a positive-penalty Gram system through its Cholesky factor."""
    system = gram.copy()
    system.flat[:: len(system) + 1] += penalty
    try:
        factor = cho_factor(system, lower=True, check_finite=False)
        solution: FloatVector = cho_solve(factor, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        solution = np.linalg.solve(system, rhs)
    return solution


def _ridge_components(
    y: FloatVector,
    train: FloatMatrix,
    test: FloatMatrix,
    lambda_: float | None,
) -> tuple[FloatVector, FloatVector, float, FloatVector, FloatVector, float]:
    """Fit RR-BLUP and return predictions, coefficients, means, and penalty."""
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
    sample_gram: FloatMatrix | None = None
    if lambda_ is None:
        sample_gram = z @ z.T
        fit = emma_remle(y, np.ones((n, 1)), sample_gram / m)
        penalty = fit.delta * m
    else:
        penalty = lambda_
    if m <= n:
        effects = _solve_penalized_gram(z.T @ z, z.T @ centered_y, penalty)
    else:
        if sample_gram is None:
            sample_gram = z @ z.T
        dual = _solve_penalized_gram(sample_gram, centered_y, penalty)
        effects = z.T @ dual
    return (
        z @ effects,
        z_test @ effects + intercept,
        intercept,
        effects,
        means,
        penalty,
    )


def _ridge_fit(
    y: FloatVector,
    train: FloatMatrix,
    test: FloatMatrix,
    lambda_: float | None,
) -> tuple[FloatVector, FloatVector, float]:
    """Return training GEBV, test phenotype prediction, and marker penalty."""
    gebv, prediction, _intercept, _effects, _means, penalty = _ridge_components(
        y,
        train,
        test,
        lambda_,
    )
    return gebv, prediction, penalty
