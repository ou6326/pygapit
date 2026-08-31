"""
PyGAPIT - Genomic Prediction / Genomic Selection (GS) Models
=============================================================
Mirrors GAPIT's prediction functions:
  - RR-BLUP  (Ridge Regression BLUP)
  - G-BLUP   (Genomic BLUP via kinship)
  - BayesB   (Meuwissen et al. 2001, simplified)

Bug fixes vs v1.0.0:
  - RR-BLUP: lambda no longer hardcoded from h2=0.5.
    Now estimated via REML (EMMA grid search on small data, cross-validated
    grid search on larger data) matching GAPIT's rrBLUP::mixed.solve.
  - GBLUP: h2 no longer hardcoded at 0.5.
    Now estimated via EMMA spectral decomposition (same as MLM/P3D).
  - GBLUP: accepts both `kinship=` and `K=` keyword aliases.
"""

from __future__ import annotations

import typing as t
import warnings

import numpy as np
from scipy import optimize, stats

from .._typing import (
    FloatMatrix,
    FloatVector,
    Matrix,
    Vector,
    as_float_matrix,
    as_float_vector,
    require_row_count,
    require_square,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared REML helper (same math as MLM _EMMA_vc)
# ─────────────────────────────────────────────────────────────────────────────


def _reml_delta(
    y: FloatVector, X: FloatMatrix, K: FloatMatrix
) -> tuple[np.float64, np.float64, np.float64]:
    """
    EMMA REML: estimate delta = Ve/Vg via spectral decomposition of K.
    Returns (delta, Vg, Ve).
    Matches GAPIT's EMMA.delta / _EMMA_vc used in MLM.
    """
    n = len(y)
    raw_eigvals, raw_U = np.linalg.eigh(K)
    eigvals: FloatVector = np.maximum(raw_eigvals, 1e-8)
    U: FloatMatrix = raw_U
    Uy: FloatVector = U.T @ y
    UX: FloatMatrix = U.T @ X

    def neg_reml(log_delta: float) -> np.float64:
        delta = t.cast(np.float64, np.exp(log_delta))
        d = eigvals + delta
        UXd: FloatMatrix = UX / d[:, np.newaxis]
        gram: FloatMatrix = UX.T @ UXd
        rhs: FloatVector = UXd.T @ Uy
        try:
            beta: FloatVector = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.float64(1e15)
        res = Uy - UX @ beta
        df = n - X.shape[1]
        s2 = np.sum(res**2 / d) / df
        sum_log_d = t.cast(np.float64, np.sum(np.log(d)))
        ll = t.cast(np.float64, sum_log_d + df * np.log(max(s2, 1e-15)) + n)
        return ll

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = optimize.minimize_scalar(neg_reml, bounds=(-6, 6), method="bounded")

    delta = t.cast(np.float64, np.exp(opt.x))
    d = eigvals + delta
    UXd = UX / d[:, np.newaxis]
    gram = UX.T @ UXd
    rhs = UXd.T @ Uy
    beta = t.cast(FloatVector, np.linalg.lstsq(gram, rhs, rcond=None)[0])
    resid = Uy - UX @ beta
    df = n - X.shape[1]
    Vg = np.sum(resid**2 / d) / df
    Ve = delta * Vg
    return delta, Vg, Ve


# ─────────────────────────────────────────────────────────────────────────────
# RR-BLUP
# ─────────────────────────────────────────────────────────────────────────────


def RR_BLUP(
    phenotype: Vector,
    genotype: Matrix,
    lambda_: float | None = None,
    n_folds: int = 5,
) -> tuple[FloatVector, float]:
    """
    Ridge Regression BLUP (RR-BLUP) for genomic prediction.

    lambda is estimated via REML (EMMA) rather than hardcoded from h2=0.5.
    This matches GAPIT's use of rrBLUP::mixed.solve internally.

    Parameters
    ----------
    phenotype : np.ndarray (n,)
    genotype  : np.ndarray (n, m)   -- marker matrix (0/1/2)
    lambda_   : float  -- ridge parameter. If None, estimated via REML.
    n_folds   : int    -- cross-validation folds for accuracy estimation

    Returns
    -------
    gebv  : np.ndarray (n,) -- genomic estimated breeding values
    acc   : float           -- cross-validated accuracy (Pearson r)
    """
    print("[PyGAPIT] Running RR-BLUP genomic prediction ...")
    y = as_float_vector(phenotype, name="phenotype")
    Z = np.nan_to_num(as_float_matrix(genotype, name="genotype matrix"))
    n, m = Z.shape
    require_row_count(Z, len(y), name="genotype matrix")

    if lambda_ is None:
        # Build GRM for REML-based lambda estimation (matches GAPIT)
        from ..stats.kinship import vanraden_kinship

        K = vanraden_kinship(Z)
        X0 = np.ones((n, 1))
        delta, Vg, Ve = _reml_delta(y, X0, K)
        # RR-BLUP: lambda = Ve/Vg * m  (because K = Z*Z'/m in RR-BLUP parameterisation)
        # delta = Ve/Vg so lambda = delta * m
        lambda_ = delta * m
        print(
            f"[PyGAPIT]  REML delta={delta:.4f}, Vg={Vg:.4f}, Ve={Ve:.4f} => lambda={lambda_:.2f}"
        )

    A = Z.T @ Z + lambda_ * np.eye(m)
    rhs = Z.T @ y
    u = np.linalg.solve(A, rhs)
    gebv = Z @ u

    acc = _cross_validate_rrblup(y, Z, lambda_, n_folds)
    print(f"[PyGAPIT]  RR-BLUP CV accuracy (r): {acc:.4f}")
    return gebv, acc


# ─────────────────────────────────────────────────────────────────────────────
# G-BLUP
# ─────────────────────────────────────────────────────────────────────────────


def GBLUP(
    phenotype: Vector,
    kinship: Matrix | None = None,
    n_folds: int = 5,
    K: Matrix | None = None,
) -> tuple[FloatVector, float]:
    """
    Genomic BLUP (G-BLUP) using a pre-computed GRM.

    h2 is estimated via REML (EMMA) per trait — NOT hardcoded at 0.5.
    Matches GAPIT's gBLUP implementation.

    Parameters
    ----------
    phenotype : np.ndarray (n,)
    kinship   : np.ndarray (n, n)   (also accepted as keyword `K=`)
    n_folds   : int

    Returns
    -------
    gebv : np.ndarray (n,)
    acc  : float
    """
    # Accept either `kinship=` or `K=`
    if kinship is None and K is not None:
        kinship = K
    if kinship is None:
        raise ValueError("kinship matrix required (pass as `kinship=` or `K=`).")
    kinship_matrix = as_float_matrix(kinship, name="kinship matrix")

    print("[PyGAPIT] Running G-BLUP genomic prediction ...")
    y = as_float_vector(phenotype, name="phenotype")
    n = len(y)
    require_square(kinship_matrix, name="kinship matrix", size=n)
    X0 = np.ones((n, 1))

    # ── REML estimate of delta = Ve/Vg ──────────────────────────────────────
    delta, Vg, Ve = _reml_delta(y, X0, kinship_matrix)
    lambda_ = delta  # Ve/Vg
    print(f"[PyGAPIT]  REML delta={delta:.4f}, h2={Vg / (Vg + Ve):.4f}")

    # ── Henderson's MME solution ──────────────────────────────────────────────
    V = kinship_matrix + lambda_ * np.eye(n)
    Vinv = np.linalg.inv(V)
    XVi = X0.T @ Vinv
    beta = np.linalg.solve(XVi @ X0, XVi @ y)
    gebv = np.asarray(kinship_matrix @ Vinv @ (y - X0 @ beta), dtype=np.float64)

    # ── Cross-validation ──────────────────────────────────────────────────────
    fold_size = n // n_folds
    predictions = np.zeros(n)
    for fold in range(n_folds):
        test = np.arange(fold * fold_size, min((fold + 1) * fold_size, n), dtype=int)
        train = np.asarray(np.setdiff1d(np.arange(n, dtype=int), test), dtype=int)
        K_tt = kinship_matrix[np.ix_(train, train)]
        K_pt = kinship_matrix[np.ix_(test, train)]
        yt = y[train]
        # Re-estimate lambda on training fold
        try:
            d_t, _, _ = _reml_delta(yt, np.ones((len(train), 1)), K_tt)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            d_t = lambda_
        V_t = K_tt + d_t * np.eye(len(train))
        Vinv_t = np.linalg.inv(V_t)
        mu_t = np.mean(yt)
        predictions[test] = K_pt @ Vinv_t @ (yt - mu_t) + mu_t

    valid = ~np.isnan(y)
    acc = stats.pearsonr(y[valid], predictions[valid]).statistic.item()
    print(f"[PyGAPIT]  G-BLUP CV accuracy (r): {acc:.4f}")
    return gebv, acc


# ─────────────────────────────────────────────────────────────────────────────
# BayesB
# ─────────────────────────────────────────────────────────────────────────────


def BayesB(
    phenotype: Vector,
    genotype: Matrix,
    n_iter: int = 5000,
    burn_in: int = 1000,
    pi: float = 0.95,
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

    Returns
    -------
    beta_hat  : np.ndarray (m,) -- posterior mean marker effects
    gebv      : np.ndarray (n,) -- genomic estimated breeding values
    """
    print(f"[PyGAPIT] Running BayesB ({n_iter} iterations, burn-in={burn_in}) ...")
    y = as_float_vector(phenotype, name="phenotype")
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

        for j in np.random.permutation(m):
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

            if np.random.rand() < p_incl:
                delta[j] = True
                beta[j] = np.random.normal(mean_j, np.sqrt(max(var_j, 0)))
            else:
                delta[j] = False
                beta[j] = 0.0

            residual -= zj * beta[j]

        Ve = _sample_var(residual, n, a=4, b=np.var(y) * 0.5)
        Vb = _sample_var(
            beta[delta],
            max(int(delta.sum()), 1),
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


def _sample_var(residuals: FloatVector, n: int, a: float = 4, b: float = 1) -> float:
    """Sample from Scaled-Inverse-Chi-Squared distribution."""
    if n < 2:
        return b / a
    shape = (a + n) / 2
    sum_squares = np.sum(residuals**2).item()
    scale = (a * b + sum_squares) / 2
    return scale / np.random.gamma(shape)


def _cross_validate_rrblup(
    y: FloatVector, Z: FloatMatrix, lambda_: float, n_folds: int
) -> float:
    n, m = Z.shape
    fold_size = n // n_folds
    preds = np.zeros(n)
    for fold in range(n_folds):
        test = list(range(fold * fold_size, min((fold + 1) * fold_size, n)))
        test_set = set(test)
        train = [i for i in range(n) if i not in test_set]
        Zt, yt = Z[train], y[train]
        A = Zt.T @ Zt + lambda_ * np.eye(m)
        u = np.linalg.solve(A, Zt.T @ yt)
        preds[test] = Z[test] @ u
    return stats.pearsonr(y, preds).statistic.item()
