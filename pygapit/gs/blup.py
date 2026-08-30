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

from dataclasses import dataclass

import numpy as np

from .._typing import FloatMatrix, FloatVector, StrVector, Vector
from ..stats.emma import emma_remle
from ..stats.kinship import vanraden_kinship


@dataclass
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
    n = len(y)
    if taxa is None:
        taxa = np.arange(n).astype(str)

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
    from ..gwas.mlm import compress_kinship, reml_for_groups

    n = len(y)
    K_full = vanraden_kinship(GD)

    if group_to is None:
        group_to = n

    # Find optimal compression
    candidates = np.unique(
        np.round(np.linspace(1, group_to, min(15, group_to))).astype(int)
    )
    best_reml = -np.inf
    best_K_eff = K_full.copy()

    for g in candidates:
        compression_failed = False
        try:
            K_c, Z = compress_kinship(K_full, int(g))
            K_eff = Z @ K_c @ Z.T + np.eye(n) * 1e-6
            reml = reml_for_groups(y, X0, K_c, Z)
            if reml > best_reml:
                best_reml = reml
                best_K_eff = K_eff.copy()
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            compression_failed = True
        if compression_failed:
            continue

    result = gblup(y, X0, best_K_eff, taxa=taxa, ngrids=ngrids)
    result.method = "cBLUP"
    return result


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

    K_pseudo += np.eye(len(y)) * 1e-6

    result = gblup(y, X0, K_pseudo, taxa=taxa, ngrids=ngrids)
    result.method = "sBLUP"
    return result
