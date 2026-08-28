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
from scipy.linalg import solve

from ..stats.emma import emma_remle
from ..stats.kinship import vanraden_kinship


@dataclass
class GBLUPResult:
    """Genomic prediction output per individual."""

    taxa: np.ndarray
    blue: np.ndarray  # BLUE (fixed effects prediction)
    blup: np.ndarray  # BLUP (total genomic breeding value)
    pev: np.ndarray  # prediction error variance
    gebv: np.ndarray  # genomic estimated breeding value
    prediction: np.ndarray  # blue + blup
    vg: float  # genetic variance
    ve: float  # residual variance
    h2: float  # heritability
    method: str = "gBLUP"


def _henderson_mme(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Henderson's Mixed Model Equations.
    Translates the MME solver from GAPIT.EMMAxP3D.R (BLUP computation section)

    System:
      [X'X        X'Z      ] [beta]   [X'y]
      [Z'X   Z'Z + delta*K^-1] [u  ] = [Z'y]

    With Z = I (complete data, no missing), this simplifies to:
      [X'X      X'  ] [beta]   [X'y]
      [X    I+delta*K^-1] [u  ] = [y  ]

    Returns (beta, u, C_uu) where C_uu = inverse of (I + delta*K^-1) block
    """
    n = len(y)
    q = X.shape[1]

    # K^-1: use pseudo-inverse for numerical stability
    try:
        K_inv = np.linalg.inv(K + np.eye(n) * 1e-8)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K)

    # Build MME coefficient matrix (2n+q) × (q+n)
    XtX = X.T @ X  # (q, q)
    XtZ = X.T  # (q, n) since Z = I
    ZtX = X  # (n, q)
    ZtZ_plus = np.eye(n) + delta * K_inv  # (n, n)

    top = np.hstack([XtX, XtZ])  # (q, q+n)
    bot = np.hstack([ZtX, ZtZ_plus])  # (n, q+n)
    C = np.vstack([top, bot])  # (q+n, q+n)

    rhs = np.concatenate([X.T @ y, y])  # (q+n,)

    try:
        sol = solve(C, rhs, assume_a="sym")
    except (ValueError, np.linalg.LinAlgError):
        sol, _, _, _ = np.linalg.lstsq(C, rhs, rcond=None)

    beta = sol[:q]
    u = sol[q:]

    # PEV = diagonal of C^-1 block for u
    try:
        C_inv = np.linalg.inv(C)
        C_uu = C_inv[q:, q:]
        pev = np.diag(C_uu)
    except np.linalg.LinAlgError:
        pev = np.full(n, np.nan)

    return beta, u, pev


def gblup(
    y: np.ndarray,
    X0: np.ndarray,
    K: np.ndarray,
    taxa: np.ndarray | None = None,
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

    # ── Solve Henderson's MME ─────────────────────────────────────────────
    beta, u, pev = _henderson_mme(y, X0, K, delta)

    # ── Compute BLUE and prediction ───────────────────────────────────────
    blue = X0 @ beta  # BLUE: fixed-effects prediction
    gebv = u  # genomic estimated breeding value
    blup = gebv  # total BLUP = random effects
    prediction = blue + blup  # phenotype prediction

    # Scale PEV by vg
    pev_scaled = pev * vg

    return GBLUPResult(
        taxa=taxa,
        blue=blue,
        blup=blup,
        pev=pev_scaled,
        gebv=gebv,
        prediction=prediction,
        vg=vg,
        ve=ve,
        h2=h2,
        method="gBLUP",
    )


def predict_new(
    K_train_train: np.ndarray,
    K_new_train: np.ndarray,
    blup_train: np.ndarray,
) -> np.ndarray:
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
    y: np.ndarray,
    X0: np.ndarray,
    GD: np.ndarray,
    taxa: np.ndarray | None = None,
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
    y: np.ndarray,
    X0: np.ndarray,
    GD: np.ndarray,
    qtn_indices: np.ndarray | None,
    taxa: np.ndarray | None = None,
    ngrids: int = 100,
) -> GBLUPResult:
    """
    SUPER BLUP (sBLUP).
    Uses pseudo-QTN kinship for prediction — better for oligogenic traits.
    Translates GAPIT.SUPER.GS.R

    Parameters
    ----------
    qtn_indices : indices of QTNs identified by SUPER/FarmCPU/BLINK GWAS
    """
    if qtn_indices is not None and len(qtn_indices) > 0:
        K_pseudo = vanraden_kinship(GD[:, qtn_indices])
    else:
        K_pseudo = vanraden_kinship(GD)

    K_pseudo += np.eye(len(y)) * 1e-6

    result = gblup(y, X0, K_pseudo, taxa=taxa, ngrids=ngrids)
    result.method = "sBLUP"
    return result
