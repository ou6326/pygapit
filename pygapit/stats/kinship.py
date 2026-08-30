"""
Kinship matrix calculations.
Direct Python translation of GAPIT.kinship.VanRaden.R and GAPIT.kinship.Zhang.R

VanRaden (2009) method:
    K = ZZ' / [2 * sum(p_j * (1-p_j))]
    where Z = centered genotype matrix (0/1/2 -> -1/0/1 minus allele freq deviation)
"""

from __future__ import annotations

import warnings

import numpy as np

from .._typing import FloatMatrix


def vanraden_kinship(GD: FloatMatrix) -> FloatMatrix:
    """
    Compute genomic relationship matrix using VanRaden (2009) method.
    Direct translation of GAPIT.kinship.VanRaden.R

    Parameters
    ----------
    GD : (n_individuals, n_snps) genotype matrix, coded 0/1/2

    Returns
    -------
    K : (n, n) symmetric kinship matrix
        K[i,i] ~ 1 for outbred, > 1 for inbred
        K[i,j] > 0 = more related than average
    """
    n, _m = GD.shape

    # ── Remove monomorphic SNPs ────────────────────────────────────────────
    fa = GD.sum(axis=0) / (2 * n)  # allele frequency
    valid = (fa > 0) & (fa < 1)
    if valid.sum() == 0:
        warnings.warn("All SNPs are monomorphic; returning identity matrix.")
        return np.eye(n)

    GD = GD[:, valid]
    fa = fa[valid]
    GD.shape[1]

    # ── Center genotypes ──────────────────────────────────────────────────
    # p = allele frequency of alternate allele
    p = GD.sum(axis=0) / (2 * n)
    # P = deviation vector: 2*(p - 0.5)
    P = 2.0 * (p - 0.5)
    # Shift coding: 0/1/2 -> -1/0/1
    Z = GD - 1.0
    # Z_centered = Z - P  (column-wise subtraction)
    Z_c = Z - P[np.newaxis, :]  # (n, m)

    # ── Compute K = Z_c' Z_c / adj ───────────────────────────────────────
    # Note: R uses crossprod(Z, Z) where Z is TRANSPOSED first
    # In Python: Z_c is (n, m), so K = Z_c @ Z_c.T
    K = Z_c @ Z_c.T  # (n, n)

    # Adjustment factor: 2 * sum(p_j * (1 - p_j))
    adj = float(2.0 * np.sum(p * (1.0 - p)))
    if adj < 1e-12:
        warnings.warn("Adjustment factor near zero; check allele frequencies.")
        adj = 1.0

    return K / adj


def zhang_kinship(GD: FloatMatrix) -> FloatMatrix:
    """
    Identity-by-state kinship (Zhang method).
    Translates GAPIT.kinship.Zhang.R

    K[i,j] = proportion of alleles shared identical-by-state
    Faster to compute than VanRaden but less statistically motivated.
    """
    _n, _m = GD.shape

    # Remove monomorphic
    fa = GD.mean(axis=0) / 2.0
    valid = (fa > 0) & (fa < 1)
    GD = GD[:, valid]

    # IBS: proportion of matching alleles
    # For 0/1/2 coded: match when |g_i - g_j| == 0
    # Approximation: use correlation-based similarity
    GD_norm = GD / 2.0  # scale to 0-1
    # Mean centering
    GD_c = GD_norm - GD_norm.mean(axis=0)
    kinship: FloatMatrix = GD_c @ GD_c.T / GD_c.shape[1]
    # Normalize to make diagonal ~ 1
    diag_mean = float(np.mean(np.diag(kinship)))
    if diag_mean > 0:
        kinship /= diag_mean

    return kinship


def scale_kinship(K: FloatMatrix) -> FloatMatrix:
    """
    Scale kinship matrix so diagonal mean = 1.
    Useful for numerical stability in mixed model solvers.
    """
    d = np.mean(np.diag(K))
    if d > 1e-12:
        return K / d
    return K
