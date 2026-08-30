"""
EMMA - Efficient Mixed Model Association
Direct Python translation of GAPIT's GAPIT.emma.R and GAPIT.EMMAxP3D.R

Mathematical model:
    y = X*beta + u + e
    u ~ N(0, K * sigma2_g)
    e ~ N(0, I * sigma2_e)
    delta = sigma2_e / sigma2_g

Core trick (P3D): Estimate delta ONCE from null model, then fix it
for all SNP tests using spectral decomposition for speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import t as t_dist

from .._typing import FloatMatrix, FloatVector
from ..io.formats import impute_missing


@dataclass
class EMMAResult:
    """Output from variance component estimation."""

    reml: float
    delta: float  # sigma2_e / sigma2_g
    ve: float  # residual variance
    vg: float  # genetic variance
    h2: float  # narrow-sense heritability


@dataclass
class GWASResult:
    """Per-SNP GWAS output."""

    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    stats: FloatVector
    vg: float
    ve: float
    h2: float


def _eigen_R_wo_Z(K: FloatMatrix, X: FloatMatrix) -> tuple[FloatVector, FloatMatrix]:
    """
    Spectral decomposition of the residual projection matrix (no Z).
    Equivalent to emma.eigen.R.wo.Z in R.
    Projects K onto null space of X, returns eigenvalues/vectors.
    """
    n, q = X.shape
    # S = I - X(X'X)^-1 X'  (projection onto null space of X)
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = np.eye(n) - X @ XtX_inv @ X.T
    # Symmetric matrix for eigen: S(K + I)S
    SHS = S @ (K + np.eye(n)) @ S
    eigvals, eigvecs = np.linalg.eigh(SHS)
    # Keep n-q non-trivial components (remove q near-zero eigenvalues)
    eigvals = eigvals[q:][::-1]
    eigvecs = eigvecs[:, q:][:, ::-1]
    return eigvals - 1.0, eigvecs  # subtract 1 added by I


def _eigen_L_wo_Z(K: FloatMatrix) -> tuple[FloatVector, FloatMatrix]:
    """
    Eigendecomposition of K for the full log-likelihood.
    Equivalent to emma.eigen.L.wo.Z in R.
    """
    eigvals, eigvecs = np.linalg.eigh(K)
    return eigvals[::-1], eigvecs[:, ::-1]


def _reml_ll(log_delta: float, lambda_R: FloatVector, etas: FloatVector) -> float:
    """
    REML log-likelihood as a function of log(delta).
    Equation from Kang et al. (2008) Genetics.
    """
    nq = len(etas)
    delta = math.exp(log_delta)
    denom = lambda_R + delta
    sse = float(np.sum(etas**2 / denom))
    log_scale = math.log(nq / (2 * math.pi)) - 1.0 - math.log(sse)
    log_denom = float(np.sum(np.log(denom)))
    return 0.5 * (nq * log_scale - log_denom)


def _reml_dll(log_delta: float, lambda_R: FloatVector, etas: FloatVector) -> float:
    """Derivative of REML log-likelihood w.r.t. log(delta)."""
    nq = len(etas)
    delta = math.exp(log_delta)
    etasq = etas**2
    denom = lambda_R + delta
    weighted_sq = float(np.sum(etasq / denom**2))
    weighted = float(np.sum(etasq / denom))
    inv_sum = float(np.sum(1.0 / denom))
    return 0.5 * delta * (nq * weighted_sq / weighted - inv_sum)


def emma_remle(
    y: FloatVector,
    X: FloatMatrix,
    K: FloatMatrix,
    ngrids: int = 100,
    llim: float = -10.0,
    ulim: float = 10.0,
    esp: float = 1e-10,
) -> EMMAResult:
    """
    REML variance component estimation via EMMA algorithm.
    Translates emma.REMLE() from GAPIT's emma.R.

    Parameters
    ----------
    y : (n,) observed phenotype
    X : (n, q) fixed-effects design matrix (intercept + covariates)
    K : (n, n) genomic kinship matrix
    ngrids : number of grid points for initial search
    llim, ulim : log-delta search bounds
    esp : convergence tolerance

    Returns
    -------
    EMMAResult with delta, ve, vg, h2, reml
    """
    n = len(y)
    q = X.shape[1]

    # Check singularity
    if np.linalg.matrix_rank(X.T @ X) < q:
        return EMMAResult(reml=0.0, delta=1.0, ve=0.0, vg=0.0, h2=0.0)

    # Spectral decomposition
    lambda_R, U_R = _eigen_R_wo_Z(K, X)
    # Rotate phenotype into eigenbasis
    etas = U_R.T @ y  # (n-q,) rotated residuals

    # Grid search over log(delta)
    log_deltas = np.linspace(llim, ulim, ngrids + 1)
    dlls = np.array([_reml_dll(ld, lambda_R, etas) for ld in log_deltas])

    opt_log_deltas: list[float] = []
    opt_lls: list[float] = []

    # Boundary cases
    if dlls[0] < esp:
        opt_log_deltas.append(llim)
        opt_lls.append(_reml_ll(llim, lambda_R, etas))
    if dlls[-2] > -esp:
        opt_log_deltas.append(ulim)
        opt_lls.append(_reml_ll(ulim, lambda_R, etas))

    # Find sign changes (local maxima of LL)
    for i in range(len(log_deltas) - 1):
        if dlls[i] * dlls[i + 1] < 0 and dlls[i] > 0 and dlls[i + 1] < 0:
            try:
                root = brentq(
                    _reml_dll,
                    log_deltas[i],
                    log_deltas[i + 1],
                    args=(lambda_R, etas),
                    xtol=esp,
                    full_output=False,
                )
                opt_log_deltas.append(root)
                opt_lls.append(_reml_ll(root, lambda_R, etas))
            except (ValueError, RuntimeError, FloatingPointError):
                root = None

    if not opt_log_deltas:
        # Fallback: take grid maximum
        best_idx = np.argmax([_reml_ll(ld, lambda_R, etas) for ld in log_deltas])
        opt_log_deltas = [log_deltas[best_idx]]
        opt_lls = [_reml_ll(log_deltas[best_idx], lambda_R, etas)]

    best_idx = int(np.argmax(opt_lls))
    best_delta = float(np.exp(opt_log_deltas[best_idx]))
    best_ll = opt_lls[best_idx]

    # Recover variance components
    nq = n - q
    denom = lambda_R + best_delta
    sse = float(np.sum(etas**2 / denom))
    vg = sse / nq
    ve = vg * best_delta
    h2 = vg / (vg + ve) if (vg + ve) > 0 else 0.0

    return EMMAResult(reml=best_ll, delta=best_delta, ve=ve, vg=vg, h2=h2)


def emmax_p3d(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    ngrids: int = 100,
    llim: float = -10.0,
    ulim: float = 10.0,
    snp_impute: str = "middle",
) -> GWASResult:
    """
    EMMAxP3D: genome-wide association using EMMA with P3D approximation.
    Translates GAPIT.EMMAxP3D.R.

    P3D (Population Parameters Previously Determined):
    1. Estimate delta from the null model (no SNP) ONCE
    2. Fix delta for all m SNP tests → fast O(m*n) instead of O(m*n^3)

    Parameters
    ----------
    y  : (n,) phenotype
    X0 : (n, q) covariate matrix (intercept + PCs)
    GD : (n, m) genotype matrix, 0/1/2 coded
    K  : (n, n) kinship matrix
    snp_impute : missing-genotype policy; GAPIT defaults to ``"middle"``

    Returns
    -------
    GWASResult with p_values, effects, se, stats, vg, ve, h2
    """
    GD = impute_missing(GD, method=snp_impute)
    n, m = GD.shape
    q0 = X0.shape[1]

    # ── Step 1: Estimate delta from null model (P3D) ──────────────────────
    remle = emma_remle(y, X0, K, ngrids=ngrids, llim=llim, ulim=ulim)
    delta = remle.delta
    vg = remle.vg
    ve = remle.ve
    h2 = remle.h2

    # ── Step 2: Build transformed system ─────────────────────────────────
    # Eigendecompose kinship: K = U * diag(lambda) * U'
    lambda_L, U_L = _eigen_L_wo_Z(K)
    lambda_L = np.maximum(lambda_L, 0)  # numerical stability

    # Rotation matrix: U * diag(1/sqrt(lambda + delta))
    scale = 1.0 / np.sqrt(lambda_L + delta)
    transformed_basis = U_L * scale
    # Apply transformation: yt = scale * U' * y,  Xt0 = scale * U' * X0
    Uty = transformed_basis.T @ y  # (n,)
    UtX0 = transformed_basis.T @ X0  # (n, q0)
    UtGD = transformed_basis.T @ GD  # (n, m)

    # ── Step 3: Test each SNP ─────────────────────────────────────────────
    q1 = q0 + 1
    p_values = np.ones(m)
    effects = np.full(m, np.nan)
    se_arr = np.full(m, np.nan)
    stats_arr = np.full(m, np.nan)
    df = n - q1

    for i in range(m):
        raw_snp = GD[:, i]
        observed = np.isfinite(raw_snp)
        if np.count_nonzero(observed) <= q1 or np.nanstd(raw_snp) < 1e-8:
            p_values[i] = 1.0
            continue
        if not np.all(observed):
            observed_count = int(np.count_nonzero(observed))
            covariance = K[np.ix_(observed, observed)] + delta * np.eye(observed_count)
            precision = np.linalg.pinv(covariance)
            marker_design = np.column_stack([X0[observed], raw_snp[observed]])
            information = marker_design.T @ precision @ marker_design
            information_inverse = np.linalg.pinv(information)
            beta = information_inverse @ marker_design.T @ precision @ y[observed]
            se = np.sqrt(information_inverse[q0, q0] * vg)
            if se < 1e-12:
                p_values[i] = 1.0
                continue
            t_stat = beta[q0] / se
            p_values[i] = float(2.0 * t_dist.sf(abs(t_stat), observed_count - q1))
            effects[i] = beta[q0]
            se_arr[i] = se
            stats_arr[i] = t_stat
            continue
        snp = UtGD[:, i]

        # Build design matrix with SNP
        Xt = np.column_stack([UtX0, snp])  # (n, q1)
        # OLS in transformed space: beta = (Xt'Xt)^-1 Xt'yt
        try:
            XtX = Xt.T @ Xt
            Xty = Xt.T @ Uty
            beta, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
        except np.linalg.LinAlgError:
            p_values[i] = 1.0
            continue

        # Standard error and t-statistic for the SNP coefficient (last element)
        try:
            iXX = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            p_values[i] = 1.0
            continue

        se = np.sqrt(iXX[q0, q0] * vg)
        if se < 1e-12:
            p_values[i] = 1.0
            continue

        t_stat = beta[q0] / se
        p_val = 2.0 * t_dist.sf(abs(t_stat), df)

        p_values[i] = min(max(p_val, 0.0), 1.0)
        effects[i] = beta[q0]
        se_arr[i] = se
        stats_arr[i] = t_stat

    return GWASResult(
        p_values=p_values,
        effects=effects,
        se=se_arr,
        stats=stats_arr,
        vg=vg,
        ve=ve,
        h2=h2,
    )
