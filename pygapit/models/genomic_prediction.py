"""Standalone prediction compatibility interfaces and simplified BayesB.

RR_BLUP and GBLUP share canonical REML and fold-local validation machinery.
For detailed out-of-fold results use pygapit.gs.validation.
"""

from __future__ import annotations

import numpy as np

from .._typing import (
    FloatVector,
    Matrix,
    Vector,
    as_float_matrix,
    as_float_vector,
    require_row_count,
)


def RR_BLUP(
    phenotype: Vector,
    genotype: Matrix,
    lambda_: float | None = None,
    n_folds: int = 5,
) -> tuple[FloatVector, float]:
    """Return centered training GEBV and fold-local Pearson accuracy.

    Fits an unpenalized intercept and marker effects. Missing markers are
    mean-imputed within each training fold. Automatic penalties use shared
    EMMA REML with K = centered_Z @ centered_Z.T / m.
    Use cross_validate_rrblup for explicit folds, groups, seeds and OOF output.
    """
    from ..gs.validation import (
        _genotypes,
        _phenotype,
        _ridge_fit,
        cross_validate_rrblup,
    )

    y = _phenotype(phenotype)
    z = _genotypes(genotype, len(y))
    cv = cross_validate_rrblup(y, z, lambda_=lambda_, n_folds=n_folds)
    gebv, _, _ = _ridge_fit(y, z, z, lambda_)
    return gebv, cv.pearson_r


def GBLUP(
    phenotype: Vector,
    kinship: Matrix | None = None,
    n_folds: int = 5,
    K: Matrix | None = None,
) -> tuple[FloatVector, float]:
    """Return canonical training GEBV and fold-local phenotype CV accuracy.

    Each validation fold estimates its own REML parameters and GLS mean.
    Use cross_validate_gblup for explicit folds, groups, seeds and OOF output.
    """
    from ..gs.blup import gblup
    from ..gs.validation import _phenotype, cross_validate_gblup

    if kinship is None:
        kinship = K
    if kinship is None:
        raise ValueError("kinship matrix required (pass as kinship= or K=)")
    y = _phenotype(phenotype)
    k = as_float_matrix(kinship, name="kinship matrix")
    cv = cross_validate_gblup(y, k, n_folds=n_folds)
    fitted = gblup(y, np.ones((len(y), 1)), k)
    return fitted.gebv, cv.pearson_r


def BayesB(
    phenotype: Vector,
    genotype: Matrix,
    n_iter: int = 5000,
    burn_in: int = 1000,
    pi: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[FloatVector, FloatVector]:
    """
    Simplified BayesB via Gibbs sampling.
    (pi = prior probability of a SNP having zero effect)

    Parameters
    ----------
    phenotype : np.ndarray (n,)
    genotype  : np.ndarray (n, m)
    n_iter    : int -- total Gibbs iterations
    burn_in   : int -- burn-in iterations (discarded)
    pi        : float -- sparsity prior
    rng       : np.random.Generator, optional -- random number generator

    Returns
    -------
    beta_hat  : np.ndarray (m,) -- posterior mean marker effects
    gebv      : np.ndarray (n,) -- genomic estimated breeding values
    """
    print(f"[PyGAPIT] Running BayesB ({n_iter} iterations, burn-in={burn_in}) ...")
    rng = np.random.default_rng() if rng is None else rng
    y = as_float_vector(phenotype, name="phenotype").copy()
    Z = np.nan_to_num(as_float_matrix(genotype, name="genotype matrix"))
    n, m = Z.shape
    require_row_count(Z, len(y), name="genotype matrix")
    mu = np.mean(y)
    y -= mu

    beta = np.zeros(m)
    delta = np.ones(m, dtype=bool)
    Ve = np.var(y) * (1 - 0.5)
    Vb = np.var(y) * 0.5 / max(m * (1 - pi), 1e-8)

    beta_samples = np.zeros((n_iter - burn_in, m))

    for it in range(n_iter):
        residual = y - Z @ beta

        for j in rng.permutation(m):
            zj = Z[:, j]
            zz = np.dot(zj, zj)
            residual += zj * beta[j]  # un-residualise this SNP

            mean_j = np.dot(zj, residual) / (zz + Ve / max(Vb, 1e-15))
            var_j = Ve / (zz + Ve / max(Vb, 1e-15))

            # Inclusion log-odds
            log_p1 = (
                0.5 * mean_j**2 / max(var_j, 1e-15)
                - 0.5 * np.log(max(Ve / max(Vb, 1e-15) + zz, 1e-15))
                + np.log((1 - pi) / max(pi, 1e-15))
            )
            p_incl = 1.0 / (1.0 + np.exp(-np.clip(log_p1, -30, 30)))

            if rng.random() < p_incl:
                delta[j] = True
                beta[j] = rng.normal(mean_j, np.sqrt(max(var_j, 0)))
            else:
                delta[j] = False
                beta[j] = 0.0

            residual -= zj * beta[j]

        Ve = _sample_var(residual, n, rng, a=4, b=np.var(y) * 0.5)
        Vb = _sample_var(
            beta[delta],
            max(int(delta.sum()), 1),
            rng,
            a=4,
            b=np.var(y) * 0.5 / max(m * (1 - pi), 1e-8),
        )

        if it >= burn_in:
            beta_samples[it - burn_in] = beta

        if (it + 1) % 500 == 0:
            print(f"[PyGAPIT]  BayesB iter {it + 1}/{n_iter}, Ve={Ve:.4f}")

    beta_hat = beta_samples.mean(axis=0)
    gebv = Z @ beta_hat + mu
    print(f"[PyGAPIT]  BayesB done. Non-zero loci: {(np.abs(beta_hat) > 1e-6).sum()}")
    return beta_hat, gebv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sample_var(
    residuals: FloatVector,
    n: int,
    rng: np.random.Generator,
    a: float = 4,
    b: float = 1,
) -> float:
    """Sample from Scaled-Inverse-Chi-Squared distribution."""
    if n < 2:
        return b / a
    shape = (a + n) / 2
    sum_squares = np.sum(residuals**2).item()
    scale = (a * b + sum_squares) / 2
    return scale / rng.gamma(shape)
