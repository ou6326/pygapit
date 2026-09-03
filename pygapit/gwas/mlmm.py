"""
MLMM - Multiple Loci Mixed Model.
Translates GAPIT.mlmm.R / GAPIT.mlmm_cof.R

Algorithm (stepwise forward/backward selection):
  1. Run MLM scan with fixed kinship K
  2. Add most significant marker as fixed cofactor
  3. Re-run MLM conditioned on all cofactors
  4. Repeat up to max_steps times
  5. Build a backward path by removing the least significant cofactor
  6. Select the optimal forward/backward model by extended BIC (extBIC)

Key difference from FarmCPU:
  K stays FIXED throughout (all-marker kinship).
  Cofactors are added as fixed effects TO the model WITH K.
  This creates partial confounding between cofactors and K,
  which FarmCPU solves by separating FEM and REM.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from .._typing import FloatMatrix, FloatVector, IntVector, readonly_copy
from ..stats.emma import GWASResult, emma_remle


@dataclass(frozen=True, slots=True)
class MLMMResult:
    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    stats: FloatVector
    selected_qtns: IntVector
    vg: float
    ve: float
    h2: float
    n_steps: int
    method: str = "MLMM"

    def __post_init__(self) -> None:
        for field in ("p_values", "effects", "se", "stats", "selected_qtns"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def _normalize_kinship(K: FloatMatrix) -> FloatMatrix:
    """Apply the GAPIT MLMM kinship scaling without changing its structure."""
    if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
        raise ValueError("MLMM kinship matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalue_scale = np.max([np.max(np.abs(eigenvalues)), 1.0])
    psd_tolerance = 1e-8 * eigenvalue_scale
    if np.min(eigenvalues) < -psd_tolerance:
        raise ValueError("MLMM kinship matrix must be positive semidefinite")
    n = K.shape[0]
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    scale = np.sum(centering * K)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("MLMM kinship matrix must have positive centered variation")
    normalized = (n - 1) * K / scale
    return t.cast(FloatMatrix, (normalized + normalized.T) / 2.0)


def _profile_ml_log_likelihood(
    y: FloatVector,
    X: FloatMatrix,
    eigenvalues: FloatVector,
    eigenvectors: FloatMatrix,
) -> np.float64:
    """Profile the ordinary ML likelihood used by GAPIT's extBIC."""
    n = len(y)
    rotated_y = eigenvectors.T @ y
    rotated_X = eigenvectors.T @ X

    def log_likelihood(log_delta: float) -> np.float64:
        delta = t.cast(np.float64, np.exp(log_delta))
        denominator = eigenvalues + delta
        weights = 1.0 / np.sqrt(denominator)
        weighted_X = rotated_X * weights[:, np.newaxis]
        weighted_y = rotated_y * weights
        beta, *_ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)
        residual = rotated_y - rotated_X @ beta
        sse = np.sum(residual**2 / denominator)
        return t.cast(
            np.float64,
            0.5
            * (
                n * (np.log(n / (2.0 * np.pi)) - 1.0 - np.log(sse))
                - np.sum(np.log(denominator))
            ),
        )

    def objective(value: float) -> np.float64:
        return -log_likelihood(value)

    optimum = minimize_scalar(
        objective,
        bounds=(-10.0, 10.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    candidates = [
        log_likelihood(-10.0),
        log_likelihood(10.0),
        log_likelihood(optimum.x),
    ]
    return np.max(candidates)


def _ext_bic(
    log_lik: float,
    n: int,
    n_fixed: int,
    m: int,
    n_selected: int,
) -> np.float64:
    """
    Extended BIC for multi-locus model selection.
    Uses the exact combinatorial penalty for choosing ``n_selected`` markers
    from ``m`` candidates. GAPIT's covariate branch incorrectly includes
    ordinary covariates in this combination count; pyGAPIT deliberately counts
    selected markers only.
    """
    log_combinations = (
        gammaln(m + 1) - gammaln(n_selected + 1) - gammaln(m - n_selected + 1)
    )
    return t.cast(
        np.float64,
        -2.0 * log_lik + (n_fixed + 1) * np.log(n) + 2.0 * log_combinations,
    )


def _restore_cofactor_statistics(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    cofactors: list[int],
    result: GWASResult,
    ngrids: int,
) -> GWASResult:
    """Replace collinear scan placeholders with joint GLS cofactor statistics."""
    if not cofactors:
        return result
    design: FloatMatrix = np.column_stack([X0] + [GD[:, index] for index in cofactors])
    remle = emma_remle(y, design, K, ngrids=ngrids)
    covariance = K + remle.delta * np.eye(len(y))
    precision = np.linalg.pinv(covariance)
    information_inverse = np.linalg.pinv(design.T @ precision @ design)
    beta = information_inverse @ design.T @ precision @ y
    standard_errors: FloatVector = np.sqrt(
        np.maximum(np.diag(information_inverse) * remle.vg, 0.0)
    )
    statistics = beta / standard_errors
    degrees_of_freedom = len(y) - design.shape[1]
    p_values = t.cast(
        FloatVector,
        2.0 * t_dist.sf(np.abs(statistics), degrees_of_freedom),
    )
    result_p_values = result.p_values.copy()
    result_effects = result.effects.copy()
    result_se = result.se.copy()
    result_stats = result.stats.copy()
    offset = X0.shape[1]
    for position, marker in enumerate(cofactors):
        coefficient = offset + position
        result_p_values[marker] = p_values[coefficient]
        result_effects[marker] = beta[coefficient]
        result_se[marker] = standard_errors[coefficient]
        result_stats[marker] = statistics[coefficient]
    return GWASResult(
        p_values=result_p_values,
        effects=result_effects,
        se=result_se,
        stats=result_stats,
        vg=result.vg,
        ve=result.ve,
        h2=result.h2,
    )


def _least_significant_cofactor(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    cofactors: list[int],
    ngrids: int,
) -> int:
    """Return the marker with GAPIT's smallest absolute joint GLS t statistic."""
    design: FloatMatrix = np.column_stack([X0] + [GD[:, index] for index in cofactors])
    remle = emma_remle(y, design, K, ngrids=ngrids)
    covariance = K + remle.delta * np.eye(len(y))
    precision = np.linalg.pinv(covariance)
    information_inverse = np.linalg.pinv(design.T @ precision @ design)
    beta = information_inverse @ design.T @ precision @ y
    standard_errors: FloatVector = np.sqrt(
        np.maximum(np.diag(information_inverse) * remle.vg, 0.0)
    )
    marker_statistics = np.abs(beta[X0.shape[1] :] / standard_errors[X0.shape[1] :])
    return cofactors[int(np.nanargmin(marker_statistics))]


def _conditioned_marker_scan(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    cofactors: list[int],
    ngrids: int,
) -> GWASResult:
    """Run the RSS/F marker scan used by GAPIT's MLMM implementation."""
    design: FloatMatrix = (
        np.column_stack([X0] + [GD[:, index] for index in cofactors])
        if cofactors
        else X0
    )
    remle = emma_remle(y, design, K, ngrids=ngrids)
    covariance = remle.vg * K + remle.ve * np.eye(len(y))
    cholesky = np.linalg.cholesky(covariance)
    transformed_y = np.linalg.solve(cholesky, y)
    transformed_design = np.linalg.solve(cholesky, design)
    design_beta, *_ = np.linalg.lstsq(transformed_design, transformed_y, rcond=None)
    residual = transformed_y - transformed_design @ design_beta
    projection = np.eye(len(y)) - transformed_design @ np.linalg.pinv(
        transformed_design
    )
    transformed_genotypes = projection @ np.linalg.solve(cholesky, GD)
    null_rss = np.sum(residual**2)
    degrees_of_freedom = len(y) - design.shape[1] - 1

    p_values = np.ones(GD.shape[1])
    effects = np.full(GD.shape[1], np.nan)
    standard_errors = np.full(GD.shape[1], np.nan)
    statistics = np.full(GD.shape[1], np.nan)
    for marker in range(GD.shape[1]):
        if marker in cofactors:
            continue
        genotype = transformed_genotypes[:, marker]
        genotype_sum_squares = np.sum(genotype**2)
        if genotype_sum_squares < 1e-12:
            continue
        effect = np.sum(genotype * residual) / genotype_sum_squares
        marker_rss = np.sum((residual - genotype * effect) ** 2)
        f_statistic = t.cast(
            np.float64,
            np.maximum(
                (null_rss / marker_rss - 1.0) * degrees_of_freedom,
                0.0,
            ),
        )
        statistic = np.sign(effect) * np.sqrt(f_statistic)
        p_values[marker] = f_dist.sf(f_statistic, 1, degrees_of_freedom)
        effects[marker] = effect
        statistics[marker] = statistic
        if f_statistic > 0.0:
            standard_errors[marker] = abs(effect) / np.sqrt(f_statistic)

    return GWASResult(
        p_values=p_values,
        effects=effects,
        se=standard_errors,
        stats=statistics,
        vg=remle.vg,
        ve=remle.ve,
        h2=remle.h2,
    )


def mlmm_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    max_steps: int = 10,
    ngrids: int = 100,
) -> MLMMResult:
    """
    MLMM genome-wide association.
    Translates mlmm() from GAPIT.mlmm.R

    Parameters
    ----------
    y          : (n,) phenotype
    X0         : (n, q) covariate matrix
    GD         : (n, m) genotype matrix
    K          : (n, n) kinship matrix (fixed throughout)
    max_steps  : maximum forward selection steps (maxsteps=10 in R)

    Returns
    -------
    MLMMResult with final p-values and selected QTN indices
    """
    n, m = GD.shape
    K = _normalize_kinship(K)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    model_candidates: list[tuple[np.float64, list[int]]] = []

    def record_model(selected: list[int]) -> None:
        design: FloatMatrix = (
            np.column_stack([X0] + [GD[:, index] for index in selected])
            if selected
            else X0
        )
        log_likelihood = _profile_ml_log_likelihood(
            y, design, eigenvalues, eigenvectors
        )
        model_candidates.append((
            _ext_bic(
                log_likelihood,
                n,
                design.shape[1],
                m,
                len(selected),
            ),
            selected.copy(),
        ))

    cofactors: list[int] = []
    record_model(cofactors)

    # ── Initial scan without cofactors ───────────────────────────────────
    result = _conditioned_marker_scan(y, X0, GD, K, cofactors, ngrids)

    # GAPIT's maxsteps counts the null model.
    for _step in range(max(max_steps - 1, 0)):
        p_values = result.p_values.copy()
        p_values[cofactors] = np.nan
        if not np.isfinite(p_values).any():
            break
        cofactors.append(int(np.nanargmin(p_values)))
        try:
            result = _conditioned_marker_scan(y, X0, GD, K, cofactors, ngrids)
            record_model(cofactors)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            cofactors.pop()
            break
        if result.h2 < 0.01:
            break

    # ── Backward elimination ─────────────────────────────────────────────
    # GAPIT builds a complete backward path from the final forward model.
    backward_cofactors = cofactors.copy()
    while len(backward_cofactors) > 1:
        dropped = _least_significant_cofactor(y, X0, GD, K, backward_cofactors, ngrids)
        backward_cofactors.remove(dropped)
        record_model(backward_cofactors)

    def criterion(candidate: tuple[np.float64, list[int]]) -> np.float64:
        return candidate[0]

    _, best_cofactors = min(model_candidates, key=criterion)

    # ── Final scan with best cofactor set ────────────────────────────────
    final_result = _conditioned_marker_scan(y, X0, GD, K, best_cofactors, ngrids)
    final_result = _restore_cofactor_statistics(
        y, X0, GD, K, best_cofactors, final_result, ngrids
    )

    return MLMMResult(
        p_values=final_result.p_values,
        effects=final_result.effects,
        se=final_result.se,
        stats=final_result.stats,
        selected_qtns=np.array(best_cofactors, dtype=int),
        vg=final_result.vg,
        ve=final_result.ve,
        h2=final_result.h2,
        n_steps=len(best_cofactors),
        method="MLMM",
    )
