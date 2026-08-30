"""
GLM - General Linear Model for GWAS.
Translates FarmCPU.LM() from GAPIT.FarmCPU.R and the base GLM model.

Model:  y = X0*beta0 + alpha*s_i + e
        e ~ N(0, I*sigma^2)

No kinship. Population structure controlled by PCs in X0.
Fast OLS per SNP via vectorized numpy operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist

from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    as_float_matrix,
    as_float_vector,
    readonly_copy,
    require_row_count,
)


@dataclass(frozen=True, slots=True)
class GLMResult:
    p_values: FloatVector  # p-values for each SNP
    effects: FloatVector  # effect size estimates
    se: FloatVector  # standard errors
    t_stats: FloatVector  # t-statistics
    r2_full: float  # R² of null model

    def __post_init__(self) -> None:
        for field in ("p_values", "effects", "se", "t_stats"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def _ols_vectorized(
    y: FloatVector, X0: FloatMatrix, GD: FloatMatrix
) -> tuple[FloatVector, FloatVector, FloatVector, FloatVector]:
    """
    Vectorized OLS test for all m SNPs simultaneously.
    Uses the partitioned regression trick:
      test SNP effect after projecting out X0 covariates.

    Returns (effects, se, t_stats, p_values)
    """
    n, q0 = X0.shape
    m = GD.shape[1]
    df = n - q0 - 1

    # ── Project y and GD onto null space of X0 ───────────────────────────
    # This is numerically equivalent to regressing out X0 first
    # y_res = y - X0 * (X0'X0)^-1 X0' y
    # g_res = GD - X0 * (X0'X0)^-1 X0' GD
    try:
        XtX_inv = np.linalg.pinv(X0.T @ X0)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X0.T @ X0 + np.eye(q0) * 1e-10)

    H = X0 @ XtX_inv @ X0.T  # hat matrix for X0
    y_res = y - H @ y  # (n,) residual of y after X0
    G_res = GD - H @ GD  # (n, m) residual of each SNP after X0

    # ── SNP effect: alpha = (g_res' y_res) / (g_res' g_res) ──────────────
    g_ss = np.sum(G_res**2, axis=0)  # (m,) sum of squares
    valid = g_ss > 1e-10  # skip monomorphic

    effects = np.zeros(m)
    se = np.ones(m)
    t_stats = np.zeros(m)
    p_values = np.ones(m)

    if valid.any():
        g_valid = G_res[:, valid]  # (n, m_valid)
        g_ss_v = g_ss[valid]  # (m_valid,)

        alpha = (g_valid.T @ y_res) / g_ss_v  # (m_valid,)

        # Residuals of full model
        y_hat_snp = g_valid * alpha[np.newaxis, :]  # (n, m_valid)
        e_full = y_res[:, np.newaxis] - y_hat_snp  # (n, m_valid)
        sse = np.sum(e_full**2, axis=0)  # (m_valid,)

        sigma2 = sse / df
        se_v = np.sqrt(sigma2 / g_ss_v)  # (m_valid,)
        se_v = np.where(se_v < 1e-12, 1e-12, se_v)

        t_v = alpha / se_v  # (m_valid,)
        p_v = 2.0 * t_dist.sf(np.abs(t_v), df)
        p_v = np.clip(p_v, 0.0, 1.0)

        effects[valid] = alpha
        se[valid] = se_v
        t_stats[valid] = t_v
        p_values[valid] = p_v

    return effects, se, t_stats, p_values


def glm_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
) -> GLMResult:
    """
    GLM genome-wide association scan.
    Translates FarmCPU.LM() from GAPIT.FarmCPU.R

    Parameters
    ----------
    y  : (n,) phenotype (no missing values)
    X0 : (n, q) covariate matrix — intercept + PCs + user CVs
    GD : (n, m) genotype matrix, 0/1/2 coded

    Returns
    -------
    GLMResult with p_values, effects, se, t_stats for all m SNPs
    """
    y = as_float_vector(y, name="phenotype")
    X0 = as_float_matrix(X0, name="covariate matrix")
    GD = as_float_matrix(GD, name="genotype matrix")
    n = len(y)
    require_row_count(X0, n, name="covariate matrix")
    require_row_count(GD, n, name="genotype matrix")
    if X0.shape[1] == 0:
        raise ValueError("covariate matrix must contain at least one column")
    if n <= X0.shape[1] + 1:
        raise ValueError("GLM requires more observations than fitted parameters")

    # Null model R²
    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    try:
        beta0 = np.linalg.lstsq(X0, y, rcond=None)[0]
        y_hat0 = X0 @ beta0
        ss_res0 = np.sum((y - y_hat0) ** 2)
        r2_null = 1.0 - ss_res0 / ss_tot if ss_tot > 0 else 0.0
    except np.linalg.LinAlgError:
        r2_null = 0.0

    effects, se, t_stats, p_values = _ols_vectorized(y, X0, GD)

    return GLMResult(
        p_values=p_values,
        effects=effects,
        se=se,
        t_stats=t_stats,
        r2_full=r2_null,
    )


def glm_scan_with_cofactors(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    cofactor_indices: IntVector | None,
) -> GLMResult:
    """
    GLM scan including pseudo-QTN cofactors as additional fixed effects.
    Used inside BLINK and FarmCPU iterations.

    Builds X_extended = [X0 | GD[:, cofactor_indices]] and runs GLM.
    """
    if cofactor_indices is not None and len(cofactor_indices) > 0:
        cofactors = GD[:, cofactor_indices]
        X_ext = np.column_stack([X0, cofactors])
    else:
        X_ext = X0

    return glm_gwas(y, X_ext, GD)
