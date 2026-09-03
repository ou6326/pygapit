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
from scipy.special import stdtr

from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    as_float_matrix,
    as_float_vector,
    readonly_copy,
    require_row_count,
)

_MARKER_BATCH_SIZE = 4096
_MARKER_BATCH_TARGET_BYTES = 32 * 1024**2
_REWARD_MAX_BASE_CONDITION = np.finfo(np.float64).eps ** -0.25
_FLOAT64_BYTES = 8


def _marker_batch_size(n_individuals: int) -> int:
    """Bound a marker-work matrix while retaining large BLAS batches."""
    memory_limited_size = max(
        1,
        _MARKER_BATCH_TARGET_BYTES // (n_individuals * _FLOAT64_BYTES),
    )
    return min(_MARKER_BATCH_SIZE, memory_limited_size)


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

    null_solver = XtX_inv @ X0.T
    y_res = y - X0 @ (null_solver @ y)
    y_res_ss = y_res @ y_res

    effects = np.zeros(m)
    se = np.ones(m)
    t_stats = np.zeros(m)
    p_values = np.ones(m)

    batch_size = _marker_batch_size(n)
    for batch_start in range(0, m, batch_size):
        batch_stop = min(batch_start + batch_size, m)
        marker_values = GD[:, batch_start:batch_stop]
        residualized = marker_values - X0 @ (null_solver @ marker_values)
        residualized_ss: FloatVector = np.einsum("ij,ij->j", residualized, residualized)
        valid = residualized_ss > 1e-10
        if not valid.any():
            continue

        residualized_y = residualized.T @ y_res
        valid_ss = residualized_ss[valid]
        valid_residualized_y = residualized_y[valid]
        alpha = valid_residualized_y / valid_ss
        sse = y_res_ss - valid_residualized_y**2 / valid_ss
        sigma2 = np.maximum(sse, 0.0) / df
        se_v = np.sqrt(sigma2 / valid_ss)
        se_v = np.where(se_v < 1e-12, 1e-12, se_v)

        t_v = alpha / se_v
        p_v: FloatVector = np.asarray(
            2.0 * stdtr(df, -np.abs(t_v)),
            dtype=np.float64,
        )
        np.clip(p_v, 0.0, 1.0, out=p_v)

        batch_rows = np.flatnonzero(valid) + batch_start
        effects[batch_rows] = alpha
        se[batch_rows] = se_v
        t_stats[batch_rows] = t_v
        p_values[batch_rows] = p_v

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
    X_ext: FloatMatrix
    if cofactor_indices is not None and len(cofactor_indices) > 0:
        cofactors = GD[:, cofactor_indices]
        X_ext = np.column_stack([X0, cofactors])
    else:
        X_ext = X0

    return glm_gwas(y, X_ext, GD)


def reward_substitute_cofactor_statistics(
    result: GLMResult,
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    qtns: IntVector,
) -> GLMResult:
    """Restore pseudo-QTN statistics using GAPIT 3.5's SUB reward rule."""
    if len(qtns) == 0:
        return result

    base_design: FloatMatrix = np.column_stack([X0, GD[:, qtns]])
    base_design_pinv = np.linalg.pinv(base_design)
    n = len(y)
    cofactor_count = len(qtns)
    cofactor_p = np.full((GD.shape[1], cofactor_count), np.nan, dtype=np.float64)
    start = X0.shape[1]
    beta = base_design_pinv @ y
    residual = y - base_design @ beta
    covariance_factor = base_design_pinv @ base_design_pinv.T

    base_condition = np.linalg.cond(base_design)
    if not np.isfinite(base_condition) or base_condition > _REWARD_MAX_BASE_CONDITION:
        # Rank-deficient augmented designs do not obey the full-rank block
        # inverse update below, while ill-conditioned ones amplify its rounding
        # error. Preserve their Moore-Penrose solution exactly.
        for marker in range(GD.shape[1]):
            marker_values = GD[:, marker]
            residualized = marker_values - base_design @ (
                base_design_pinv @ marker_values
            )
            if residualized @ residualized < 1e-8:
                continue
            design: FloatMatrix = np.column_stack([base_design, marker_values])
            degrees_of_freedom = n - design.shape[1]
            design_pinv = np.linalg.pinv(design)
            marker_beta = design_pinv @ y
            marker_residual = y - design @ marker_beta
            mse = (marker_residual @ marker_residual) / degrees_of_freedom
            marker_covariance = design_pinv @ design_pinv.T * mse
            standard_errors = np.sqrt(np.maximum(np.diag(marker_covariance), 0.0))
            statistics = marker_beta / standard_errors
            p_values = np.asarray(
                2.0 * stdtr(degrees_of_freedom, -np.abs(statistics)),
                dtype=np.float64,
            )
            cofactor_p[marker] = p_values[start : start + cofactor_count]
    else:
        # Frisch-Waugh-Lovell plus the block inverse of [B | g]'[B | g]
        # gives every substitute-marker fit from one pseudoinverse of B.
        # Work in batches so the temporary residualized genotype matrix stays
        # bounded for large marker sets.
        degrees_of_freedom = n - base_design.shape[1] - 1
        cofactor_beta = beta[start : start + cofactor_count]
        cofactor_variance = np.diag(covariance_factor)[start : start + cofactor_count]
        batch_size = _marker_batch_size(n)
        base_residual_ss = residual @ residual
        for batch_start in range(0, GD.shape[1], batch_size):
            batch_stop = min(batch_start + batch_size, GD.shape[1])
            marker_values = GD[:, batch_start:batch_stop]
            projection_coefficients = base_design_pinv @ marker_values
            residualized = marker_values - base_design @ projection_coefficients
            residualized_ss: FloatVector = np.einsum(
                "ij,ij->j", residualized, residualized
            )
            valid = residualized_ss >= 1e-8
            if not valid.any():
                continue

            valid_ss = residualized_ss[valid]
            residualized_y = residualized.T @ residual
            valid_residualized_y = residualized_y[valid]
            marker_effects = valid_residualized_y / valid_ss
            residual_ss = base_residual_ss - (valid_residualized_y**2 / valid_ss)
            mse: FloatVector = np.maximum(residual_ss, 0.0) / degrees_of_freedom

            cofactor_projection = projection_coefficients[
                start : start + cofactor_count, valid
            ]
            substitute_effects = cofactor_beta[:, np.newaxis] - (
                cofactor_projection * marker_effects[np.newaxis, :]
            )
            substitute_variances = (
                cofactor_variance[:, np.newaxis]
                + cofactor_projection**2 / valid_ss[np.newaxis, :]
            )
            substitute_se = np.sqrt(
                np.maximum(substitute_variances * mse[np.newaxis, :], 0.0)
            )
            substitute_statistics: FloatMatrix = substitute_effects / substitute_se
            substitute_p = np.asarray(
                2.0 * stdtr(degrees_of_freedom, -np.abs(substitute_statistics)),
                dtype=np.float64,
            )
            batch_rows = np.flatnonzero(valid) + batch_start
            cofactor_p[batch_rows] = substitute_p.T

    # GAPIT's min(..., na.rm=TRUE) returns Inf when every substitute is
    # unavailable. Normalize that invalid p-value to the equivalent
    # non-significant value 1.0 while preserving GAPIT's finite rewards.
    reward_p = np.asarray(
        [
            np.min(column[np.isfinite(column)]) if np.isfinite(column).any() else 1.0
            for column in cofactor_p.T
        ],
        dtype=np.float64,
    )
    degrees_of_freedom = n - base_design.shape[1]
    mse = (residual @ residual) / degrees_of_freedom
    covariance = covariance_factor * mse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistics = beta / standard_errors

    p_values = result.p_values.copy()
    effects = result.effects.copy()
    se = result.se.copy()
    t_stats = result.t_stats.copy()
    p_values[qtns] = reward_p
    effects[qtns] = beta[start : start + cofactor_count]
    se[qtns] = standard_errors[start : start + cofactor_count]
    t_stats[qtns] = statistics[start : start + cofactor_count]
    return GLMResult(p_values, effects, se, t_stats, result.r2_full)
