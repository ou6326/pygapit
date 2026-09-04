"""
Kinship matrix calculations.
Direct Python translation of GAPIT.kinship.VanRaden.R and GAPIT.kinship.Zhang.R

VanRaden (2009) method:
    K = ZZ' / [2 * sum(p_j * (1-p_j))]
    where Z = centered genotype matrix (0/1/2 -> -1/0/1 minus allele freq deviation)
"""

from __future__ import annotations

import typing as t
import warnings

import numpy as np

from .._typing import FloatMatrix, as_float_matrix, require_square


def vanraden_factor(GD: FloatMatrix) -> FloatMatrix:
    """Return a factor whose cross-product is the VanRaden kinship matrix."""
    GD = as_float_matrix(GD, name="genotype matrix")
    n = GD.shape[0]
    allele_frequencies = GD.sum(axis=0) / (2.0 * n)
    valid = (allele_frequencies > 0.0) & (allele_frequencies < 1.0)
    if not valid.any():
        warnings.warn("All SNPs are monomorphic; returning identity factor.")
        return np.eye(n)

    selected = GD[:, valid]
    frequencies = allele_frequencies[valid]
    centered = selected - 2.0 * frequencies
    adjustment = 2.0 * np.sum(frequencies * (1.0 - frequencies))
    if adjustment < 1e-12:
        warnings.warn("Adjustment factor near zero; check allele frequencies.")
        adjustment = 1.0
    return centered / t.cast(np.float64, np.sqrt(adjustment))


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
    GD = as_float_matrix(GD, name="genotype matrix")
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
    adj = 2.0 * np.sum(p * (1.0 - p))
    if adj < 1e-12:
        warnings.warn("Adjustment factor near zero; check allele frequencies.")
        adj = 1.0

    return K / adj


def zhang_kinship(GD: FloatMatrix) -> FloatMatrix:
    """
    Compute the Zhang relationship matrix used by GAPIT 3.5.
    Translates GAPIT.kinship.Zhang.R
    """
    GD = as_float_matrix(GD, name="genotype matrix")
    n, _m = GD.shape

    # Remove invariant markers, matching the R implementation.
    fa = GD.sum(axis=0) / (2.0 * n)
    valid = (fa > 0) & (fa < 1)
    GD = GD[:, valid]
    if GD.shape[1] == 0:
        warnings.warn("All SNPs are monomorphic; returning identity matrix.")
        return np.eye(n)

    heterozygosity = 1.0 - np.abs(GD - 1.0)
    individual_heterozygosity = heterozygosity.sum(axis=1) / (2.0 * GD.shape[1])
    inbreeding = 1.0 - np.min(individual_heterozygosity)
    top = 1.0 + inbreeding

    centered = GD - GD.mean(axis=0)
    kinship: FloatMatrix = centered @ centered.T
    diagonal = np.diag(kinship).copy()
    diagonal_min = np.min(diagonal)
    diagonal_max = np.max(diagonal)
    floor = np.min(kinship)
    scale = diagonal_max - floor
    if scale <= 1e-12:
        warnings.warn(
            "Zhang kinship has no usable variation; returning identity matrix."
        )
        return np.eye(n)

    scaled_kinship: FloatMatrix = top * (kinship - floor) / scale
    kinship = scaled_kinship
    adjusted_diagonal_min = top * (diagonal_min - floor) / scale
    diagonal_mask = np.eye(n, dtype=bool)
    off_diagonal_mask = ~diagonal_mask

    if adjusted_diagonal_min < 1.0:
        denominator = (top + 1.0 - adjusted_diagonal_min) * 0.5
        kinship[diagonal_mask] = (
            kinship[diagonal_mask] - adjusted_diagonal_min + 1.0
        ) / denominator
        if adjusted_diagonal_min > 1e-12:
            kinship[off_diagonal_mask] /= adjusted_diagonal_min

    if n > 1:
        off_diagonal_max = np.max(kinship[off_diagonal_mask])
        if off_diagonal_max > top:
            kinship[off_diagonal_mask] *= top / off_diagonal_max

    return kinship


def scale_kinship(K: FloatMatrix) -> FloatMatrix:
    """
    Scale kinship matrix so diagonal mean = 1.
    Useful for numerical stability in mixed model solvers.
    """
    K = as_float_matrix(K, name="kinship matrix")
    require_square(K, name="kinship matrix")
    d = np.mean(np.diag(K))
    if d > 1e-12:
        return K / d
    return K
