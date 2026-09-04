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

import typing as t
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.optimize import brentq
from scipy.stats import t as t_dist

from .._typing import (
    FloatMatrix,
    FloatVector,
    as_float_matrix,
    as_float_vector,
    readonly_copy,
    require_row_count,
    require_square,
)
from ..io.formats import impute_missing


@dataclass(frozen=True, slots=True)
class EMMAResult:
    """Output from variance component estimation."""

    reml: float
    delta: float  # sigma2_e / sigma2_g
    ve: float  # residual variance
    vg: float  # genetic variance
    h2: float  # narrow-sense heritability


@dataclass(frozen=True, slots=True)
class GWASResult:
    """Per-SNP GWAS output."""

    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    stats: FloatVector
    vg: float
    ve: float
    h2: float

    def __post_init__(self) -> None:
        for field in ("p_values", "effects", "se", "stats"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


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


def _real_eigendecomposition(
    matrix: FloatMatrix,
) -> tuple[FloatVector, FloatMatrix]:
    """Match R's general ``eigen`` ordering for the native-incidence path."""
    values, vectors = np.linalg.eig(matrix)
    if (
        np.max(np.abs(values.imag), initial=0.0) > 1e-8
        or np.max(np.abs(vectors.imag), initial=0.0) > 1e-8
    ):
        raise np.linalg.LinAlgError("incidence eigendecomposition is not real")
    real_values = values.real
    real_vectors = vectors.real
    order = np.argsort(np.abs(real_values))[::-1]
    return real_values[order], real_vectors[:, order]


def _eigen_L_w_Z(Z: FloatMatrix, K: FloatMatrix) -> tuple[FloatVector, FloatMatrix]:
    """Translate GAPIT's ``emma.eigen.L.w.Z`` decomposition."""
    values, vectors = _real_eigendecomposition(K @ (Z.T @ Z))
    basis, _ = np.linalg.qr(Z @ vectors, mode="complete")
    return values, basis


def _eigen_R_w_Z(
    Z: FloatMatrix, K: FloatMatrix, X: FloatMatrix
) -> tuple[FloatVector, FloatMatrix]:
    """Translate GAPIT's ``emma.eigen.R.w.Z`` decomposition."""
    n, t_random = Z.shape
    q_fixed = X.shape[1]
    random_rank = t_random - q_fixed
    if random_rank <= 0:
        raise ValueError(
            "incidence matrix must have more columns than the fixed-effect design"
        )

    projection_coefficients, *_ = np.linalg.lstsq(X, Z, rcond=None)
    residualized_Z = Z - X @ projection_coefficients
    values: FloatVector | None = None
    random_basis: FloatMatrix | None = None
    # High compression permits an equivalent factorization in group space:
    # RKR' = (R K^1/2)(R K^1/2)'.  Use it only when the smaller problem is
    # materially cheaper and has enough well-resolved positive eigenvalues.
    # Multi-scale crossover benchmarks show a stable group-space advantage
    # through half-size incidence problems.  Beyond that point the faster path
    # depends on sample size and the active BLAS implementation.
    if t_random * 2 <= n:
        symmetric_kinship = (K + K.T) / 2.0
        kinship_values, kinship_vectors = np.linalg.eigh(symmetric_kinship)
        kinship_scale = np.max([np.max(np.abs(kinship_values)), 1.0])
        kinship_tolerance = np.finfo(np.float64).eps * t_random * kinship_scale
        positive = kinship_values > kinship_tolerance
        if np.count_nonzero(positive) >= random_rank:
            kinship_factor: FloatMatrix = kinship_vectors[:, positive] * np.sqrt(
                kinship_values[positive]
            )
            observation_factor = residualized_Z @ kinship_factor
            group_covariance = observation_factor.T @ observation_factor
            group_covariance = (group_covariance + group_covariance.T) / 2.0
            group_values, group_vectors = np.linalg.eigh(group_covariance)
            group_values = group_values[::-1][:random_rank]
            group_scale = np.max([np.max(np.abs(group_values)), 1.0])
            group_tolerance = np.finfo(np.float64).eps * n * group_scale
            if np.min(group_values) > group_tolerance:
                values = group_values
                group_vectors = group_vectors[:, ::-1][:, :random_rank]
                random_basis = observation_factor @ group_vectors
                random_basis /= np.sqrt(values)[np.newaxis, :]
    if values is None or random_basis is None:
        # Rank-deficient and weakly compressed cases retain the established
        # observation-space path.
        # AB and BA share their nonzero eigenvalues; use the symmetric
        # observation-space form to avoid platform-dependent complex eigenvectors.
        residual_covariance = residualized_Z @ K @ residualized_Z.T
        residual_covariance = (residual_covariance + residual_covariance.T) / 2.0
        if random_rank * 2 < n:
            values, random_basis = scipy_eigh(
                residual_covariance,
                subset_by_index=(n - random_rank, n - 1),
            )
        else:
            values, random_basis = np.linalg.eigh(residual_covariance)
        values = values[::-1]
        random_basis = random_basis[:, ::-1]
        values = values[:random_rank]
        random_basis = random_basis[:, :random_rank]
    fixed_basis, _ = np.linalg.qr(X, mode="reduced")
    # random_basis comes from the eigendecomposition of the covariance after
    # projecting Z out of X, so it is already orthonormal and orthogonal to
    # fixed_basis.  Re-factorizing their concatenation repeats an O(n t^2) QR
    # for every compression candidate.  Numerical zero eigenvalues are the
    # exception: their arbitrary eigenvectors can leave the residualized range,
    # so retain the established QR fallback for rank-deficient spectra.
    model_basis: FloatMatrix = np.column_stack([random_basis, fixed_basis])
    value_scale = np.max([np.max(np.abs(values)), 1.0])
    value_tolerance = np.finfo(np.float64).eps * n * value_scale
    if np.min(np.abs(values)) <= value_tolerance:
        model_basis, _ = np.linalg.qr(model_basis, mode="reduced")
    return values, model_basis


def _reml_ll(log_delta: float, lambda_R: FloatVector, etas: FloatVector) -> np.float64:
    """
    REML log-likelihood as a function of log(delta).
    Equation from Kang et al. (2008) Genetics.
    """
    nq = len(etas)
    delta = t.cast(np.float64, np.exp(log_delta))
    denom = lambda_R + delta
    sse = np.sum(etas**2 / denom)
    log_scale = t.cast(np.float64, np.log(nq / (2 * np.pi)) - 1.0 - np.log(sse))
    log_denom = t.cast(np.float64, np.sum(np.log(denom)))
    return np.float64(0.5) * (nq * log_scale - log_denom)


def _reml_dll(log_delta: float, lambda_R: FloatVector, etas: FloatVector) -> np.float64:
    """Derivative of REML log-likelihood w.r.t. log(delta)."""
    nq = len(etas)
    delta = t.cast(np.float64, np.exp(log_delta))
    etasq = etas**2
    denom = lambda_R + delta
    weighted_sq = np.sum(etasq / denom**2)
    weighted = np.sum(etasq / denom)
    inv_sum = np.sum(1.0 / denom)
    return np.float64(0.5) * delta * (nq * weighted_sq / weighted - inv_sum)


def _reml_ll_w_Z(
    log_delta: float,
    lambda_R: FloatVector,
    etas: FloatVector,
    residual_rank: int,
) -> np.float64:
    """REML log-likelihood for GAPIT's native incidence-matrix path."""
    delta = t.cast(np.float64, np.exp(log_delta))
    etas1 = etas[: len(lambda_R)]
    etas2_sq = np.sum(etas[len(lambda_R) :] ** 2)
    nq = len(etas)
    denom = lambda_R + delta
    sse = np.sum(etas1**2 / denom) + etas2_sq / delta
    log_scale = t.cast(np.float64, np.log(nq / (2 * np.pi)) - 1.0 - np.log(sse))
    log_denom = t.cast(
        np.float64,
        np.sum(np.log(denom)) + residual_rank * np.log(delta),
    )
    return np.float64(0.5) * (nq * log_scale - log_denom)


def _reml_dll_w_Z(
    log_delta: float,
    lambda_R: FloatVector,
    etas: FloatVector,
    residual_rank: int,
) -> np.float64:
    """Derivative of native-incidence REML with respect to log(delta)."""
    delta = t.cast(np.float64, np.exp(log_delta))
    etas1_sq = etas[: len(lambda_R)] ** 2
    etas2_sq = np.sum(etas[len(lambda_R) :] ** 2)
    nq = len(etas)
    denom = lambda_R + delta
    weighted_sq = np.sum(etas1_sq / denom**2) + etas2_sq / delta**2
    weighted = np.sum(etas1_sq / denom) + etas2_sq / delta
    inverse_sum = np.sum(1.0 / denom) + residual_rank / delta
    return np.float64(0.5) * delta * (nq * weighted_sq / weighted - inverse_sum)


def emma_remle(
    y: FloatVector,
    X: FloatMatrix,
    K: FloatMatrix,
    ngrids: int = 100,
    llim: float = -10.0,
    ulim: float = 10.0,
    esp: float = 1e-10,
    Z: FloatMatrix | None = None,
) -> EMMAResult:
    """
    REML variance component estimation via EMMA algorithm.
    Translates emma.REMLE() from GAPIT's emma.R.

    Parameters
    ----------
    y : (n,) observed phenotype
    X : (n, q) fixed-effects design matrix (intercept + covariates)
    K : genomic kinship matrix; (n, n) without Z or (t, t) with Z
    ngrids : number of grid points for initial search
    llim, ulim : log-delta search bounds
    esp : convergence tolerance
    Z : optional (n, t) incidence matrix for random effects

    Returns
    -------
    EMMAResult with delta, ve, vg, h2, reml
    """
    y = as_float_vector(y, name="phenotype")
    X = as_float_matrix(X, name="design matrix")
    K = as_float_matrix(K, name="kinship matrix")
    n = len(y)
    require_row_count(X, n, name="design matrix")
    q = X.shape[1]
    if q == 0:
        raise ValueError("design matrix must contain at least one column")
    if n <= q:
        raise ValueError("REML requires more observations than fixed effects")

    incidence: FloatMatrix | None = None
    if Z is None:
        require_square(K, name="kinship matrix", size=n)
    else:
        incidence = as_float_matrix(Z, name="incidence matrix")
        require_row_count(incidence, n, name="incidence matrix")
        require_square(K, name="kinship matrix", size=incidence.shape[1])
        if incidence.shape[1] <= q:
            raise ValueError(
                "incidence matrix must have more columns than the fixed-effect design"
            )

    if np.linalg.matrix_rank(X) < q:
        raise ValueError("design matrix must have linearly independent columns")

    # Spectral decomposition
    etas: FloatVector
    if incidence is None:
        lambda_R, U_R = _eigen_R_wo_Z(K, X)
        residual_rank = 0
        etas = U_R.T @ y
    else:
        lambda_R, model_basis = _eigen_R_w_Z(incidence, K, X)
        residual_rank = n - incidence.shape[1]
        model_coordinates = model_basis.T @ y
        random_coordinates = model_coordinates[: len(lambda_R)]
        if residual_rank == 0:
            etas = random_coordinates
        else:
            residual_sum_squares = max(
                y @ y - model_coordinates @ model_coordinates,
                0.0,
            )
            residual_coordinates = np.zeros(residual_rank, dtype=np.float64)
            residual_coordinates[0] = np.sqrt(residual_sum_squares)
            etas = np.concatenate([random_coordinates, residual_coordinates])

    def ll_at(log_delta: float) -> np.float64:
        if incidence is None:
            return _reml_ll(log_delta, lambda_R, etas)
        return _reml_ll_w_Z(log_delta, lambda_R, etas, residual_rank)

    def dll_at(log_delta: float) -> np.float64:
        if incidence is None:
            return _reml_dll(log_delta, lambda_R, etas)
        return _reml_dll_w_Z(log_delta, lambda_R, etas, residual_rank)

    # Grid search over log(delta)
    log_deltas = np.linspace(llim, ulim, ngrids + 1)
    dlls = np.array([dll_at(ld) for ld in log_deltas])

    opt_log_deltas: list[float] = []
    opt_lls: list[float] = []

    # Boundary cases
    if dlls[0] < esp:
        opt_log_deltas.append(llim)
        opt_lls.append(ll_at(llim))
    if dlls[-2] > -esp:
        opt_log_deltas.append(ulim)
        opt_lls.append(ll_at(ulim))

    # Find sign changes (local maxima of LL)
    for i in range(len(log_deltas) - 1):
        if dlls[i] * dlls[i + 1] < 0 and dlls[i] > 0 and dlls[i + 1] < 0:
            try:
                root = brentq(
                    dll_at,
                    log_deltas[i],
                    log_deltas[i + 1],
                    xtol=esp,
                    full_output=False,
                )
                opt_log_deltas.append(root)
                opt_lls.append(ll_at(root))
            except (ValueError, RuntimeError, FloatingPointError):
                root = None

    if not opt_log_deltas:
        # Fallback: take grid maximum
        best_idx = np.argmax([ll_at(ld) for ld in log_deltas])
        opt_log_deltas = [log_deltas[best_idx]]
        opt_lls = [ll_at(log_deltas[best_idx])]

    best_idx = int(np.argmax(opt_lls))
    best_delta = t.cast(np.float64, np.exp(opt_log_deltas[best_idx]))
    best_ll = opt_lls[best_idx]

    # Recover variance components
    nq = n - q
    denom = lambda_R + best_delta
    if incidence is None:
        sse = np.sum(etas**2 / denom)
    else:
        etas1 = etas[: len(lambda_R)]
        etas2_sq = np.sum(etas[len(lambda_R) :] ** 2)
        sse = np.sum(etas1**2 / denom) + etas2_sq / best_delta
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
    Z: FloatMatrix | None = None,
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
    K  : kinship matrix; (n, n) without Z or (t, t) with Z
    snp_impute : missing-genotype policy; GAPIT defaults to ``"middle"``
    Z  : optional (n, t) incidence matrix for random effects

    Returns
    -------
    GWASResult with p_values, effects, se, stats, vg, ve, h2
    """
    y = as_float_vector(y, name="phenotype")
    X0 = as_float_matrix(X0, name="covariate matrix")
    GD = as_float_matrix(GD, name="genotype matrix")
    K = as_float_matrix(K, name="kinship matrix")
    n = len(y)
    require_row_count(X0, n, name="covariate matrix")
    require_row_count(GD, n, name="genotype matrix")
    incidence: FloatMatrix | None = None
    if Z is None:
        require_square(K, name="kinship matrix", size=n)
    else:
        incidence = as_float_matrix(Z, name="incidence matrix")
        require_row_count(incidence, n, name="incidence matrix")
        require_square(K, name="kinship matrix", size=incidence.shape[1])
    GD = impute_missing(GD, method=snp_impute)
    n, m = GD.shape
    q0 = X0.shape[1]

    # ── Step 1: Estimate delta from null model (P3D) ──────────────────────
    remle = emma_remle(y, X0, K, ngrids=ngrids, llim=llim, ulim=ulim, Z=incidence)
    delta = remle.delta
    vg = remle.vg
    ve = remle.ve
    h2 = remle.h2

    # ── Step 2: Build transformed system ─────────────────────────────────
    # Eigendecompose kinship: K = U * diag(lambda) * U'
    if incidence is None:
        lambda_L, U_L = _eigen_L_wo_Z(K)
    else:
        lambda_L, U_L = _eigen_L_w_Z(incidence, K)
    lambda_L = np.maximum(lambda_L, 0)  # numerical stability

    # Rotation matrix: U * diag(1/sqrt(lambda + delta))
    scale = 1.0 / np.sqrt(lambda_L + delta)
    if incidence is not None:
        scale = np.concatenate([
            scale,
            np.full(n - incidence.shape[1], 1.0 / np.sqrt(delta)),
        ])
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

    se: np.float64
    t_stat: np.float64
    for i in range(m):
        raw_snp = GD[:, i]
        observed = np.isfinite(raw_snp)
        if np.count_nonzero(observed) <= q1 or np.nanstd(raw_snp) < 1e-8:
            p_values[i] = 1.0
            continue
        if not np.all(observed):
            observed_count = int(np.count_nonzero(observed))
            random_covariance: FloatMatrix
            if incidence is None:
                random_covariance = K[np.ix_(observed, observed)]
            else:
                observed_incidence: FloatMatrix = incidence[observed]
                random_covariance = observed_incidence @ K @ observed_incidence.T
            covariance: FloatMatrix = random_covariance + delta * np.eye(observed_count)
            precision: FloatMatrix = np.linalg.pinv(covariance)
            marker_design: FloatMatrix = np.column_stack([
                X0[observed],
                raw_snp[observed],
            ])
            information: FloatMatrix = marker_design.T @ precision @ marker_design
            information_inverse: FloatMatrix = np.linalg.pinv(information)
            beta: FloatVector = (
                information_inverse @ marker_design.T @ precision @ y[observed]
            )
            se = np.sqrt(information_inverse[q0, q0] * vg)
            if se < 1e-12:
                p_values[i] = 1.0
                continue
            t_stat = beta[q0] / se
            p_values[i] = 2.0 * t_dist.sf(abs(t_stat), observed_count - q1)
            effects[i] = beta[q0]
            se_arr[i] = se
            stats_arr[i] = t_stat
            continue
        snp = UtGD[:, i]

        # Build design matrix with SNP
        Xt: FloatMatrix = np.column_stack([UtX0, snp])  # (n, q1)
        # OLS in transformed space: beta = (Xt'Xt)^-1 Xt'yt
        try:
            XtX: FloatMatrix = Xt.T @ Xt
            Xty: FloatVector = Xt.T @ Uty
            beta, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
        except np.linalg.LinAlgError:
            p_values[i] = 1.0
            continue

        # Standard error and t-statistic for the SNP coefficient (last element)
        try:
            iXX: FloatMatrix = np.linalg.inv(XtX)
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
