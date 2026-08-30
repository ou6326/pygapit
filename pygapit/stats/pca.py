"""
Principal Component Analysis for population structure control.
Translates GAPIT.PCA.R

PCA on the genotype matrix controls for population structure
by including the top k PCs as fixed-effect covariates in the
GWAS/GS model (Q matrix approach, Price et al. 2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import (
    FloatMatrix,
    FloatVector,
    as_float_matrix,
    readonly_copy,
    require_row_count,
)


@dataclass(frozen=True, slots=True)
class PCAResult:
    scores: FloatMatrix  # (n, k) PC scores per individual
    loadings: FloatMatrix  # (m, k) SNP loadings
    var_explained: FloatVector  # proportion variance explained
    eigenvalues: FloatVector

    def __post_init__(self) -> None:
        for field in ("scores", "loadings", "var_explained", "eigenvalues"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def compute_pca(
    GD: FloatMatrix,
    n_components: int = 3,
    maf_filter: float = 0.05,
) -> PCAResult:
    """
    Compute principal components from genotype matrix.
    Translates GAPIT.PCA.R

    Parameters
    ----------
    GD : (n_individuals, n_snps) genotype matrix, 0/1/2 coded
    n_components : number of PCs to return (PCA.total parameter in GAPIT)
    maf_filter : minimum MAF for SNPs used in PCA

    Returns
    -------
    PCAResult with scores, loadings, variance explained
    """
    GD = as_float_matrix(GD, name="genotype matrix")
    n, _m = GD.shape
    if n < 2:
        raise ValueError("PCA requires at least two individuals")
    if n_components < 0:
        raise ValueError("n_components must be non-negative")

    # ── MAF filter ────────────────────────────────────────────────────────
    freq = GD.sum(axis=0) / (2 * n)
    maf = np.minimum(freq, 1.0 - freq)
    valid_snps = maf >= maf_filter
    if valid_snps.sum() < n_components:
        valid_snps = maf > 0  # relax if too few pass filter

    GD_filtered = GD[:, valid_snps]

    # ── Center each SNP as in R's prcomp(scale. = FALSE) ────────────────
    col_means = GD_filtered.mean(axis=0)
    GD_centered = GD_filtered - col_means

    # ── SVD (efficient for tall matrices) ────────────────────────────────
    k = min(n_components, min(GD_centered.shape))
    # Use truncated SVD via numpy
    # G = U * S * V^T, scores = U * S
    U, singular_values, Vt = np.linalg.svd(GD_centered, full_matrices=False)
    singular_values_array: FloatVector = singular_values
    eigenvalues_all: FloatVector = singular_values_array**2 / (n - 1)
    U = U[:, :k]
    S = singular_values_array[:k]
    Vt = Vt[:k, :]

    scores = U * S  # (n, k) — PC scores (same as R's prcomp$x)
    loadings = Vt.T  # (m, k)
    eigenvalues = S**2 / (n - 1)
    total_var = float(np.sum(eigenvalues_all))
    var_explained: FloatVector = (
        eigenvalues / total_var if total_var > 0 else eigenvalues / eigenvalues.sum()
    )

    return PCAResult(
        scores=scores,
        loadings=loadings,
        var_explained=var_explained,
        eigenvalues=eigenvalues,
    )


def build_covariate_matrix(
    pca_result: PCAResult,
    n_pcs: int,
    extra_covariates: FloatMatrix | None = None,
) -> FloatMatrix:
    """
    Build the fixed-effect design matrix X0 = [1 | PC1 | ... | PCk | CVs].
    This is the X0 matrix used in all GAPIT GWAS/GS models.

    Parameters
    ----------
    pca_result : PCAResult from compute_pca
    n_pcs : number of PCs to include
    extra_covariates : (n, p) additional fixed effects (user CV matrix)

    Returns
    -------
    X0 : (n, 1 + n_pcs + p) design matrix
    """
    n = pca_result.scores.shape[0]
    k = min(n_pcs, pca_result.scores.shape[1])

    # Always include intercept
    X0 = np.ones((n, 1))

    if k > 0:
        X0 = np.column_stack([X0, pca_result.scores[:, :k]])

    if extra_covariates is not None:
        cv = as_float_matrix(extra_covariates, name="extra covariates")
        require_row_count(cv, n, name="extra covariates")
        X0 = np.column_stack([X0, cv])

    return X0
