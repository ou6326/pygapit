"""
Genomic Selection / Prediction methods.
Translates GAPIT.GS.R, GAPIT.EMMAxP3D.R (BLUP section), GAPIT.SUPER.GS.R

Methods:
  gBLUP  - Genomic BLUP: uses genome-wide kinship (polygenic traits)
  cBLUP  - Compressed BLUP: uses CMLM compression (large datasets)
  sBLUP  - SUPER BLUP: uses pseudo-QTN kinship (oligogenic traits)

Henderson's Mixed Model Equations (MME):
  [X'X        X'Z      ] [beta]   [X'y]
  [Z'X   Z'Z + delta*K^-1] [u  ] = [Z'y]

BLUP = û = K Z' V^-1 (y - X*beta)
BLUE = X*beta (fixed effects only)
PEV  = diag(C22) where C22 is the (2,2) block of the MME inverse
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    StrVector,
    Vector,
    as_float_matrix,
    as_float_vector,
    as_str_vector,
    readonly_copy,
    require_length,
    require_row_count,
    require_square,
)
from ..stats.emma import EMMAResult, emma_remle
from ..stats.kinship import vanraden_kinship


@dataclass(frozen=True, slots=True)
class GBLUPResult:
    """Genomic prediction output per individual."""

    taxa: StrVector
    blue: FloatVector  # BLUE (fixed effects prediction)
    blup: FloatVector  # BLUP (total genomic breeding value)
    pev: FloatVector  # prediction error variance
    gebv: FloatVector  # genomic estimated breeding value
    prediction: FloatVector  # blue + blup
    vg: float  # genetic variance
    ve: float  # residual variance
    h2: float  # heritability
    method: str = "gBLUP"

    def __post_init__(self) -> None:
        for field in ("taxa", "blue", "blup", "pev", "gebv", "prediction"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class SUPERSelectionResult:
    """Pseudo-QTN sets evaluated by the SUPER selection stage."""

    qtn_indices: IntVector
    candidate_counts: IntVector
    reml: FloatVector

    def __post_init__(self) -> None:
        for field in ("qtn_indices", "candidate_counts", "reml"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def _emma_blup(
    y: FloatVector,
    X: FloatMatrix,
    K: FloatMatrix,
    delta: float,
    vg: float,
    ve: float,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """
    Solve GAPIT's EMMA-transformed mixed model for BLUE, BLUP, and PEV.

    The spectral form remains well defined for the singular relationship
    matrices produced by centered markers, unlike regularizing ``K`` before
    inversion in a direct Henderson system.
    """
    n = len(y)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    transformed_basis = eigenvectors * (1.0 / np.sqrt(eigenvalues + delta))
    transformed_y = transformed_basis.T @ y
    transformed_x = transformed_basis.T @ X
    information = transformed_x.T @ transformed_x
    score = transformed_x.T @ transformed_y
    try:
        information_inverse = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        information_inverse = np.linalg.pinv(information)
    beta = information_inverse @ score

    residual = y - X @ beta
    transformed_residual = transformed_basis.T @ residual
    u = K @ transformed_basis @ transformed_residual

    c11 = vg * information_inverse
    c21 = -K @ transformed_basis @ transformed_x @ c11
    k_inverse = np.linalg.pinv(K)
    try:
        random_block = np.linalg.inv(np.eye(n) / ve + k_inverse / vg)
    except np.linalg.LinAlgError:
        random_block = np.linalg.pinv(np.eye(n) / ve + k_inverse / vg)
    correction = c21 @ transformed_x.T @ transformed_basis.T @ K
    pev = np.diag(random_block - correction)

    return beta, u, pev


def _emma_blup_with_incidence(
    y: FloatVector,
    X: FloatMatrix,
    K: FloatMatrix,
    Z: FloatMatrix,
    vg: float,
    ve: float,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """Solve GAPIT's native incidence-matrix BLUE, BLUP, and PEV system."""
    genetic_covariance = vg * K
    covariance = Z @ genetic_covariance @ Z.T + ve * np.eye(len(y))
    solve_rhs: FloatMatrix = np.column_stack([y, X])
    solved = np.linalg.solve(covariance, solve_rhs)
    precision_y = solved[:, 0]
    precision_x = solved[:, 1:]
    information = X.T @ precision_x
    try:
        information_inverse = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        information_inverse = np.linalg.pinv(information)
    beta = information_inverse @ X.T @ precision_y
    covariance_projection = genetic_covariance @ Z.T
    random_effect = covariance_projection @ (precision_y - precision_x @ beta)

    kinship_inverse = np.linalg.pinv(K, hermitian=True)
    random_information = Z.T @ Z / ve + kinship_inverse / vg
    try:
        conditional_covariance = np.linalg.inv(random_information)
    except np.linalg.LinAlgError:
        conditional_covariance = np.linalg.pinv(random_information)
    fixed_effect_basis = covariance_projection @ precision_x
    fixed_effect_correction = (
        fixed_effect_basis @ information_inverse @ fixed_effect_basis.T
    )
    pev = np.diag(conditional_covariance + fixed_effect_correction)
    return beta, random_effect, pev


def gblup(
    y: FloatVector,
    X0: FloatMatrix,
    K: FloatMatrix,
    taxa: StrVector | None = None,
    ngrids: int = 100,
) -> GBLUPResult:
    """
    Genomic BLUP prediction.
    Translates the BLUP computation in GAPIT.EMMAxP3D.R and GAPIT.GS.R

    Parameters
    ----------
    y    : (n,) phenotype (training set)
    X0   : (n, q) covariate matrix (intercept + PCs)
    K    : (n, n) kinship matrix
    taxa : (n,) individual IDs

    Returns
    -------
    GBLUPResult with BLUP, BLUE, PEV per individual
    """
    y = as_float_vector(y, name="phenotype")
    X0 = as_float_matrix(X0, name="covariate matrix")
    K = as_float_matrix(K, name="kinship matrix")
    n = len(y)
    require_row_count(X0, n, name="covariate matrix")
    require_square(K, name="kinship matrix", size=n)
    if taxa is None:
        taxa = np.arange(n).astype(str)
    else:
        taxa = as_str_vector(taxa, name="taxa")
        require_length(taxa, n, name="taxa")

    # ── Estimate variance components via REML ────────────────────────────
    remle = emma_remle(y, X0, K, ngrids=ngrids)
    delta = remle.delta
    vg = remle.vg
    ve = remle.ve
    h2 = vg / (vg + ve) if (vg + ve) > 0 else 0.0

    # ── Solve in the same EMMA spectral space used by GAPIT ───────────────
    beta, u, pev = _emma_blup(y, X0, K, delta, vg, ve)

    # ── Compute BLUE and prediction ───────────────────────────────────────
    blue = X0 @ beta  # BLUE: fixed-effects prediction
    gebv = u  # genomic estimated breeding value
    blup = gebv  # total BLUP = random effects
    prediction = blue + blup  # phenotype prediction

    return GBLUPResult(
        taxa=taxa,
        blue=blue,
        blup=blup,
        pev=pev,
        gebv=gebv,
        prediction=prediction,
        vg=vg,
        ve=ve,
        h2=h2,
        method="gBLUP",
    )


def predict_new(
    K_train_train: FloatMatrix,
    K_new_train: FloatMatrix,
    blup_train: FloatVector,
) -> FloatVector:
    """
    Predict GEBV for new (un-phenotyped) individuals.
    Translates GAPIT.GS.R: UO = t(KWO) %*% solve(KW) %*% UW

    Parameters
    ----------
    K_train_train : (n_train, n_train) kinship among training individuals
    K_new_train   : (n_new, n_train) kinship between new and training individuals
    blup_train    : (n_train,) BLUPs from training set

    Returns
    -------
    (n_new,) predicted GEBVs for new individuals
    """
    K_train_train = as_float_matrix(K_train_train, name="training kinship matrix")
    K_new_train = as_float_matrix(K_new_train, name="new-to-training kinship matrix")
    blup_train = as_float_vector(blup_train, name="training BLUP")
    n_train = len(blup_train)
    require_square(K_train_train, name="training kinship matrix", size=n_train)
    if K_new_train.shape[1] != n_train:
        raise ValueError(
            "new-to-training kinship matrix must have one column per training BLUP"
        )

    try:
        K_inv = np.linalg.solve(
            K_train_train + np.eye(len(K_train_train)) * 1e-8,
            np.eye(len(K_train_train)),
        )
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K_train_train)

    return np.asarray(K_new_train @ K_inv @ blup_train, dtype=float)


def cblup(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    taxa: StrVector | None = None,
    group_to: int | None = None,
    ngrids: int = 100,
) -> GBLUPResult:
    """
    Compressed BLUP (cBLUP).
    Uses CMLM-compressed kinship for prediction.
    Translates GAPIT's cBLUP path.

    Faster than gBLUP for large n. Uses optimal group kinship.
    """
    from ..gwas.mlm import (
        _fit_reml_for_groups,
        _kinship_cluster_tree,
        compress_kinship,
    )

    y = as_float_vector(y, name="phenotype")
    X0 = as_float_matrix(X0, name="covariate matrix")
    GD = as_float_matrix(GD, name="genotype matrix")
    n = len(y)
    require_row_count(X0, n, name="covariate matrix")
    require_row_count(GD, n, name="genotype matrix")
    K_full = vanraden_kinship(GD)

    if group_to is None:
        group_to = n

    # Find optimal compression
    candidates = np.unique(
        np.round(np.linspace(1, group_to, min(15, group_to))).astype(int)
    )
    cluster_tree: FloatMatrix | None = None
    if np.any((candidates > 1) & (candidates < n)):
        try:
            cluster_tree = _kinship_cluster_tree(K_full)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            cluster_tree = None
    best_reml = -np.inf
    best_candidate_fit: EMMAResult | None = None
    best_K_c = K_full.copy()
    best_Z = np.eye(n)

    for g in candidates:
        compression_failed = False
        try:
            K_c, Z = compress_kinship(
                K_full,
                int(g),
                cluster_tree=cluster_tree,
            )
            fit = _fit_reml_for_groups(y, X0, K_c, Z)
            if fit.reml > best_reml:
                best_reml = fit.reml
                best_candidate_fit = fit
                best_K_c = K_c.copy()
                best_Z = Z.copy()
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            compression_failed = True
        if compression_failed:
            continue

    # Compression selection uses EMMA's standard 100-point grid.  Reuse that
    # complete fit for the default final resolution; a custom final grid still
    # requires the established refit so its numerical behavior is unchanged.
    if ngrids == 100 and best_candidate_fit is not None:
        remle = best_candidate_fit
    else:
        remle = emma_remle(y, X0, best_K_c, ngrids=ngrids, Z=best_Z)
    beta, group_blup, group_pev = _emma_blup_with_incidence(
        y,
        X0,
        best_K_c,
        best_Z,
        remle.vg,
        remle.ve,
    )
    if taxa is None:
        taxa = np.arange(n).astype(str)
    else:
        taxa = as_str_vector(taxa, name="taxa")
        require_length(taxa, n, name="taxa")
    blue = X0 @ beta
    blup = best_Z @ group_blup
    pev = best_Z @ group_pev
    return GBLUPResult(
        taxa=taxa,
        blue=blue,
        blup=blup,
        pev=pev,
        gebv=blup,
        prediction=blue + blup,
        vg=remle.vg,
        ve=remle.ve,
        h2=remle.h2,
        method="cBLUP",
    )


def select_super_qtns(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    chromosomes: Vector,
    positions: FloatVector,
    p_values: FloatVector,
    *,
    bin_size: int = 10_000,
    candidate_counts: Sequence[int] | None = None,
    ngrids: int = 100,
) -> SUPERSelectionResult:
    """Select a pseudo-QTN kinship by binning markers and maximizing REML."""
    y = as_float_vector(y, name="phenotype")
    X0 = as_float_matrix(X0, name="covariate matrix")
    GD = as_float_matrix(GD, name="genotype matrix")
    chromosomes = as_str_vector(chromosomes, name="marker chromosomes")
    positions = as_float_vector(positions, name="marker positions")
    p_values = as_float_vector(p_values, name="marker p-values")
    n = len(y)
    marker_count = GD.shape[1]
    require_row_count(X0, n, name="covariate matrix")
    require_row_count(GD, n, name="genotype matrix")
    for values, name in (
        (chromosomes, "marker chromosomes"),
        (positions, "marker positions"),
        (p_values, "marker p-values"),
    ):
        require_length(values, marker_count, name=name)
    if bin_size <= 0:
        raise ValueError("SUPER bin_size must be positive")
    if not np.isfinite(positions).all():
        raise ValueError("SUPER marker positions must be finite")

    finite_markers = np.flatnonzero(np.isfinite(p_values))
    if len(finite_markers) == 0:
        raise ValueError("SUPER selection requires at least one finite marker p-value")
    if np.any((p_values[finite_markers] < 0.0) | (p_values[finite_markers] > 1.0)):
        raise ValueError("SUPER marker p-values must be between 0 and 1")

    representatives: dict[tuple[str, int], int] = {}
    for marker_value in finite_markers:
        marker = marker_value.item()
        key = (chromosomes[marker], int(np.floor(positions[marker] / bin_size)))
        current = representatives.get(key)
        if current is None or p_values[marker] < p_values[current]:
            representatives[key] = marker
    ranked = np.asarray(tuple(representatives.values()), dtype=np.int_)
    ranked = ranked[np.lexsort((ranked, p_values[ranked]))]

    requested_counts = (
        tuple(range(10, 101, 10))
        if candidate_counts is None
        else tuple(candidate_counts)
    )
    if not requested_counts:
        raise ValueError("SUPER candidate_counts must contain at least one value")
    if any(type(count) is not int for count in requested_counts):
        raise TypeError("SUPER candidate_counts must contain integers")
    if any(count <= 0 for count in requested_counts):
        raise ValueError("SUPER candidate_counts must be positive")
    counts = np.unique(
        np.minimum(np.asarray(requested_counts, dtype=np.int_), len(ranked))
    )

    fitted_counts: list[int] = []
    fitted_reml: list[float] = []
    fitted_qtns: list[IntVector] = []
    for count in counts:
        qtns = ranked[:count]
        varying = np.var(GD[:, qtns], axis=0) > 0.0
        qtns = qtns[varying]
        if len(qtns) == 0:
            continue
        try:
            pseudo_kinship = vanraden_kinship(GD[:, qtns])
            fit = emma_remle(y, X0, pseudo_kinship, ngrids=ngrids)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        fitted_counts.append(int(count))
        fitted_reml.append(fit.reml)
        fitted_qtns.append(np.sort(qtns))

    if not fitted_qtns:
        raise ValueError("SUPER selection could not fit any pseudo-QTN candidate set")
    best = int(np.argmax(fitted_reml))
    return SUPERSelectionResult(
        qtn_indices=np.asarray(fitted_qtns[best], dtype=np.int_),
        candidate_counts=np.asarray(fitted_counts, dtype=np.int_),
        reml=np.asarray(fitted_reml, dtype=np.float64),
    )


def sblup(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    qtn_indices: Vector,
    taxa: StrVector | None = None,
    ngrids: int = 100,
) -> GBLUPResult:
    """
    SUPER BLUP (sBLUP).
    Uses pseudo-QTN kinship for prediction — better for oligogenic traits.
    Translates GAPIT.SUPER.GS.R

    Parameters
    ----------
    qtn_indices : non-empty indices of pseudo-QTNs identified by
        SUPER/FarmCPU/BLINK GWAS
    """
    y = as_float_vector(y, name="phenotype")
    X0 = as_float_matrix(X0, name="covariate matrix")
    GD = as_float_matrix(GD, name="genotype matrix")
    n = len(y)
    require_row_count(X0, n, name="covariate matrix")
    require_row_count(GD, n, name="genotype matrix")

    indices = np.asarray(qtn_indices)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("sBLUP requires at least one pseudo-QTN index")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("sBLUP pseudo-QTN indices must be integers")
    integer_indices = indices.astype(np.intp, copy=False)
    if np.any(integer_indices < 0) or np.any(integer_indices >= GD.shape[1]):
        raise ValueError("sBLUP pseudo-QTN index is outside the genotype matrix")

    unique_indices = np.unique(integer_indices)
    K_pseudo = vanraden_kinship(GD[:, unique_indices])

    result = gblup(y, X0, K_pseudo, taxa=taxa, ngrids=ngrids)
    return replace(result, method="sBLUP")
