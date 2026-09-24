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

from .._resources import (
    DEFAULT_MARKER_WORKSPACE_MIB,
    marker_batch_size,
    validate_marker_workspace_mib,
)
from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    as_float_matrix,
    as_float_vector,
    readonly_copy,
    require_row_count,
)
from ..io.storage import GenotypeStore, GenotypeView, as_genotype_store

_MARKER_BATCH_SIZE = 4096
_REWARD_MAX_BASE_CONDITION = np.finfo(np.float64).eps ** -0.25


def _marker_batch_size(
    n_individuals: int,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
) -> int:
    """Bound a marker-work matrix while retaining large BLAS batches."""
    return marker_batch_size(
        n_individuals,
        marker_workspace_mib,
        max_markers=_MARKER_BATCH_SIZE,
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
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix | GenotypeStore,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
) -> tuple[FloatVector, FloatVector, FloatVector, FloatVector]:
    """
    Vectorized OLS test for all m SNPs simultaneously.
    Uses the partitioned regression trick:
      test SNP effect after projecting out X0 covariates.

    Returns (effects, se, t_stats, p_values)
    """
    n, q0 = X0.shape
    genotype = as_genotype_store(GD)
    m = genotype.shape[1]
    df = n - q0 - 1

    # ── Project y and GD onto null space of X0 ───────────────────────────
    # This is numerically equivalent to regressing out X0 first
    # y_res = y - X0 * (X0'X0)^-1 X0' y
    # g_res = GD - X0 * (X0'X0)^-1 X0' GD
    # Work with X directly instead of pinv(X'X), whose condition number is
    # squared and can turn constant markers into numerical false positives.
    null_solver = np.linalg.pinv(X0)
    y_res = y - X0 @ (null_solver @ y)
    y_res_ss = y_res @ y_res

    effects = np.zeros(m)
    se = np.ones(m)
    t_stats = np.zeros(m)
    p_values = np.ones(m)

    batch_size = _marker_batch_size(n, marker_workspace_mib)
    for batch_start in range(0, m, batch_size):
        batch_stop = min(batch_start + batch_size, m)
        marker_values = genotype.read_markers(slice(batch_start, batch_stop))
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
    GD: FloatMatrix | GenotypeStore,
    *,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
) -> GLMResult:
    """
    GLM genome-wide association scan.
    Translates FarmCPU.LM() from GAPIT.FarmCPU.R

    Parameters
    ----------
    y  : (n,) phenotype (no missing values)
    X0 : (n, q) covariate matrix — intercept + PCs + user CVs
    GD : (n, m) genotype matrix or chunk-readable store, 0/1/2 coded
    marker_workspace_mib : target size of one temporary marker-work matrix

    Returns
    -------
    GLMResult with p_values, effects, se, t_stats for all m SNPs
    """
    marker_workspace_mib = validate_marker_workspace_mib(marker_workspace_mib)
    y = as_float_vector(y, name="phenotype")
    X0 = as_float_matrix(X0, name="covariate matrix")
    genotype = as_genotype_store(GD)
    n = len(y)
    require_row_count(X0, n, name="covariate matrix")
    if genotype.shape[0] != n:
        raise ValueError(f"genotype matrix must have {n} rows; got {genotype.shape[0]}")
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

    effects, se, t_stats, p_values = _ols_vectorized(
        y,
        X0,
        genotype,
        marker_workspace_mib,
    )

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
    GD: FloatMatrix | GenotypeStore,
    cofactor_indices: IntVector | None,
    *,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
) -> GLMResult:
    """
    GLM scan including pseudo-QTN cofactors as additional fixed effects.
    Used inside BLINK and FarmCPU iterations.

    Builds X_extended = [X0 | GD[:, cofactor_indices]] and runs GLM.
    """
    genotype = as_genotype_store(GD)
    X_ext: FloatMatrix
    if cofactor_indices is not None and len(cofactor_indices) > 0:
        cofactors = _read_selected_markers(
            genotype,
            cofactor_indices,
            marker_workspace_mib,
        )
        X_ext = np.column_stack([X0, cofactors])
    else:
        X_ext = X0

    return glm_gwas(
        y,
        X_ext,
        genotype,
        marker_workspace_mib=marker_workspace_mib,
    )


def reward_substitute_cofactor_statistics(
    result: GLMResult,
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix | GenotypeStore,
    qtns: IntVector,
    *,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
) -> GLMResult:
    """Restore pseudo-QTN statistics using GAPIT 3.5's SUB reward rule."""
    if len(qtns) == 0:
        return result

    marker_workspace_mib = validate_marker_workspace_mib(marker_workspace_mib)
    genotype = as_genotype_store(GD)
    cofactor_values = _read_selected_markers(
        genotype,
        qtns,
        marker_workspace_mib,
    )
    base_design: FloatMatrix = np.column_stack([X0, cofactor_values])
    base_design_pinv = np.linalg.pinv(base_design)
    n = len(y)
    cofactor_count = len(qtns)
    cofactor_p = np.full(
        (genotype.shape[1], cofactor_count),
        np.nan,
        dtype=np.float64,
    )
    start = X0.shape[1]
    beta = base_design_pinv @ y
    residual = y - base_design @ beta
    covariance_factor = base_design_pinv @ base_design_pinv.T

    base_condition = np.linalg.cond(base_design)
    if not np.isfinite(base_condition) or base_condition > _REWARD_MAX_BASE_CONDITION:
        # Rank-deficient augmented designs do not obey the full-rank block
        # inverse update below, while ill-conditioned ones amplify its rounding
        # error. Preserve their Moore-Penrose solution exactly.
        _fill_rank_deficient_reward_p_values(
            genotype,
            y,
            base_design,
            base_design_pinv,
            start,
            cofactor_count,
            marker_workspace_mib,
            cofactor_p,
        )
    else:
        # Frisch-Waugh-Lovell plus the block inverse of [B | g]'[B | g]
        # gives every substitute-marker fit from one pseudoinverse of B.
        # Work in batches so the temporary residualized genotype matrix stays
        # bounded for large marker sets.
        _fill_full_rank_reward_p_values(
            genotype,
            residual,
            base_design,
            base_design_pinv,
            np.diag(covariance_factor),
            beta,
            start,
            cofactor_count,
            marker_workspace_mib,
            cofactor_p,
        )

    reward_p = _minimum_available_p_values(cofactor_p)
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


def _fill_rank_deficient_reward_p_values(
    genotype: GenotypeStore,
    y: FloatVector,
    base_design: FloatMatrix,
    base_design_pinv: FloatMatrix,
    cofactor_start: int,
    cofactor_count: int,
    marker_workspace_mib: float,
    cofactor_p: FloatMatrix,
) -> None:
    """Evaluate substitutes by exact pseudoinverse for unstable designs."""
    n = len(y)
    batch_size = _marker_batch_size(n, marker_workspace_mib)
    for batch_start in range(0, genotype.shape[1], batch_size):
        batch_stop = min(batch_start + batch_size, genotype.shape[1])
        marker_batch = genotype.read_markers(slice(batch_start, batch_stop))
        for batch_offset in range(marker_batch.shape[1]):
            marker_values = marker_batch[:, batch_offset]
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
            covariance = design_pinv @ design_pinv.T * mse
            standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            statistics = marker_beta / standard_errors
            p_values = np.asarray(
                2.0 * stdtr(degrees_of_freedom, -np.abs(statistics)),
                dtype=np.float64,
            )
            marker = batch_start + batch_offset
            cofactor_p[marker] = p_values[
                cofactor_start : cofactor_start + cofactor_count
            ]


def _fill_full_rank_reward_p_values(
    genotype: GenotypeStore,
    residual: FloatVector,
    base_design: FloatMatrix,
    base_design_pinv: FloatMatrix,
    covariance_diagonal: FloatVector,
    beta: FloatVector,
    cofactor_start: int,
    cofactor_count: int,
    marker_workspace_mib: float,
    cofactor_p: FloatMatrix,
) -> None:
    """Apply the bounded Frisch-Waugh-Lovell substitute-marker update."""
    n = len(residual)
    degrees_of_freedom = n - base_design.shape[1] - 1
    cofactor_beta = beta[cofactor_start : cofactor_start + cofactor_count]
    cofactor_variance = covariance_diagonal[
        cofactor_start : cofactor_start + cofactor_count
    ]
    batch_size = _marker_batch_size(n, marker_workspace_mib)
    base_residual_ss = residual @ residual
    for batch_start in range(0, genotype.shape[1], batch_size):
        batch_stop = min(batch_start + batch_size, genotype.shape[1])
        marker_values = genotype.read_markers(slice(batch_start, batch_stop))
        projection = base_design_pinv @ marker_values
        residualized = marker_values - base_design @ projection
        residualized_ss: FloatVector = np.einsum("ij,ij->j", residualized, residualized)
        valid = residualized_ss >= 1e-8
        if not valid.any():
            continue
        valid_ss = residualized_ss[valid]
        residualized_y = (residualized.T @ residual)[valid]
        marker_effects = residualized_y / valid_ss
        residual_ss = base_residual_ss - residualized_y**2 / valid_ss
        mse: FloatVector = np.maximum(residual_ss, 0.0) / degrees_of_freedom
        cofactor_projection = projection[
            cofactor_start : cofactor_start + cofactor_count, valid
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
        statistics: FloatMatrix = substitute_effects / substitute_se
        substitute_p = np.asarray(
            2.0 * stdtr(degrees_of_freedom, -np.abs(statistics)),
            dtype=np.float64,
        )
        cofactor_p[np.flatnonzero(valid) + batch_start] = substitute_p.T


def _minimum_available_p_values(cofactor_p: FloatMatrix) -> FloatVector:
    """Return each cofactor's best finite substitute-marker p-value."""
    result: FloatVector = np.asarray(
        [
            np.min(column[np.isfinite(column)]) if np.isfinite(column).any() else 1.0
            for column in cofactor_p.T
        ],
        dtype=np.float64,
    )
    return result


def _read_selected_markers(
    genotype: GenotypeStore,
    marker_indices: IntVector,
    marker_workspace_mib: float,
) -> FloatMatrix:
    """Read an arbitrary marker selection through bounded logical batches."""
    selected = np.empty(
        (genotype.shape[0], len(marker_indices)),
        dtype=np.float64,
    )
    batch_size = _marker_batch_size(genotype.shape[0], marker_workspace_mib)
    for start in range(0, len(marker_indices), batch_size):
        stop = min(start + batch_size, len(marker_indices))
        view = GenotypeView(
            genotype,
            marker_indices=marker_indices[start:stop],
        )
        selected[:, start:stop] = view.read_markers(slice(None))
    return selected
