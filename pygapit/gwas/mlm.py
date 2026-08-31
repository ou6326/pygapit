"""
MLM and CMLM - Mixed Linear Model and Compressed MLM for GWAS.
Translates GAPIT.Main.R (MLM section) and GAPIT.Compress.R

MLM model:
    y = X*beta + u + e
    u ~ N(0, K*sigma2_g),  e ~ N(0, I*sigma2_e)

CMLM: compress n individuals into g groups via hierarchical clustering,
use group-level kinship instead of individual kinship.
Optimal g selected by maximum REML log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .._typing import FloatMatrix, FloatVector, readonly_copy
from ..stats.emma import emma_remle, emmax_p3d


@dataclass(frozen=True, slots=True)
class MLMResult:
    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    stats: FloatVector
    vg: float
    ve: float
    h2: float
    method: str = "MLM"

    def __post_init__(self) -> None:
        for field in ("p_values", "effects", "se", "stats"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def mlm_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    ngrids: int = 100,
) -> MLMResult:
    """
    MLM genome-wide association using EMMA + P3D.
    Translates GAPIT.Main.R MLM path + GAPIT.EMMAxP3D.R

    Parameters
    ----------
    y  : (n,) phenotype vector
    X0 : (n, q) covariate matrix (intercept + PCs)
    GD : (n, m) genotype matrix, 0/1/2 coded
    K  : (n, n) kinship matrix (VanRaden or user-supplied)

    Returns
    -------
    MLMResult with p_values, effects, vg, ve, h2
    """
    result = emmax_p3d(y, X0, GD, K, ngrids=ngrids)
    return MLMResult(
        p_values=result.p_values,
        effects=result.effects,
        se=result.se,
        stats=result.stats,
        vg=result.vg,
        ve=result.ve,
        h2=result.h2,
        method="MLM",
    )


# ── CMLM: Compressed Mixed Linear Model ──────────────────────────────────


def compress_kinship(
    K: FloatMatrix,
    n_groups: int,
) -> tuple[FloatMatrix, FloatMatrix]:
    """
    Compress n individuals into g groups and compute group kinship.
    Translates GAPIT.Compress.R and GAPIT.ZmatrixCompress.R

    Returns
    -------
    K_c : (g, g) group kinship matrix
    Z   : (n, g) incidence matrix mapping individuals to groups
    """
    n = K.shape[0]
    n_groups = min(n_groups, n)

    if n_groups == n:
        return K.copy(), np.eye(n)

    if n_groups <= 1:
        # All in one group → GLM equivalent
        Z = np.ones((n, 1)) / n
        K_c = np.array([[K.mean()]])
        return K_c, Z

    try:
        # GAPIT.Compress calls R's dist(K), i.e. Euclidean distance between
        # complete kinship-profile rows rather than 1 - pairwise kinship.
        condensed = pdist(K, metric="euclidean")
        Z_link = linkage(condensed, method="average")
        labels = fcluster(Z_link, n_groups, criterion="maxclust")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        # Fallback: random grouping
        labels = np.tile(np.arange(n_groups), int(np.ceil(n / n_groups)))[:n] + 1

    # Build Z matrix: n × g
    g = len(np.unique(labels))
    Z = np.zeros((n, g))
    for i, lbl in enumerate(labels):
        Z[i, lbl - 1] = 1.0
    # Normalize columns so each column sums to group size
    col_sums = Z.sum(axis=0)
    col_sums[col_sums == 0] = 1.0

    # Group kinship: K_c = (Z'KZ) / (group_sizes outer product)
    ZtK = Z.T @ K  # (g, n)
    K_c = ZtK @ Z  # (g, g)
    # Normalize by group sizes
    outer = np.outer(col_sums, col_sums)
    K_c = K_c / outer

    return K_c, Z


def reml_for_groups(
    y: FloatVector,
    X0: FloatMatrix,
    K_c: FloatMatrix,
    Z: FloatMatrix,
) -> float:
    """
    Compute REML log-likelihood for a given compression.
    Used to select optimal group number in CMLM.
    Translates GAPIT's group optimization by REML.
    """
    result = emma_remle(y, X0, K_c, Z=Z)
    return result.reml


def cmlm_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    group_from: int = 1,
    group_to: int | None = None,
    ngrids: int = 100,
) -> MLMResult:
    """
    CMLM genome-wide association.
    Translates GAPIT.Main.R CMLM path + GAPIT.Compress.R + GAPIT.ZmatrixCompress.R

    Selects optimal compression by maximizing REML log-likelihood.

    Parameters
    ----------
    group_from : minimum number of groups to try
    group_to   : maximum number of groups (default = n)

    Notes
    -----
    A native-incidence REML fit requires more random-effect groups than fixed
    effects. GAPIT silently changes smaller group counts to one and effectively
    switches models; pyGAPIT instead excludes invalid search candidates and
    rejects a range containing no valid CMLM fit.
    """
    n = len(y)
    if group_to is None:
        group_to = n

    # Clamp range
    group_from = max(1, group_from)
    group_to = min(n, group_to)
    minimum_groups = X0.shape[1] + 1
    if group_to < minimum_groups:
        raise ValueError(
            "CMLM group_to must be greater than the number of fixed effects "
            f"({X0.shape[1]})"
        )
    group_from = max(group_from, minimum_groups)

    # Try a range of group counts, pick best REML
    candidates = np.unique(
        np.round(
            np.linspace(group_from, group_to, min(20, group_to - group_from + 1))
        ).astype(int)
    )

    best_reml = -np.inf
    best_K_c = K.copy()
    best_Z = np.eye(n)
    best_n_groups = n
    fitted_candidate = False

    for g in candidates:
        compression_failed = False
        try:
            K_c, Z = compress_kinship(K, int(g))
            reml = reml_for_groups(y, X0, K_c, Z)
            if reml > best_reml:
                fitted_candidate = True
                best_reml = reml
                best_K_c = K_c.copy()
                best_Z = Z.copy()
                best_n_groups = g
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            compression_failed = True
        if compression_failed:
            continue

    if not fitted_candidate:
        raise RuntimeError("CMLM failed to fit every requested compression level")

    # Run EMMAX-P3D with optimal compressed kinship
    result = emmax_p3d(y, X0, GD, best_K_c, ngrids=ngrids, Z=best_Z)
    return MLMResult(
        p_values=result.p_values,
        effects=result.effects,
        se=result.se,
        stats=result.stats,
        vg=result.vg,
        ve=result.ve,
        h2=result.h2,
        method=f"CMLM(g={best_n_groups})",
    )
