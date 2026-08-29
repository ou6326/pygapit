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
from numpy.typing import NDArray


@dataclass
class PCAResult:
    scores: np.ndarray  # (n, k) PC scores per individual
    loadings: np.ndarray  # (m, k) SNP loadings
    var_explained: np.ndarray  # (k,) proportion variance explained
    eigenvalues: np.ndarray  # (k,) eigenvalues


def compute_pca(
    GD: np.ndarray,
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
    GD = np.asarray(GD, dtype=float)
    n, _m = GD.shape

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
    singular_values_array: NDArray[np.float64] = np.asarray(
        singular_values, dtype=np.float64
    )
    eigenvalues_all: NDArray[np.float64] = singular_values_array**2 / (n - 1)
    U = U[:, :k]
    S = singular_values_array[:k]
    Vt = Vt[:k, :]

    scores = U * S  # (n, k) — PC scores (same as R's prcomp$x)
    loadings = Vt.T  # (m, k)
    eigenvalues = S**2 / (n - 1)
    total_var = float(np.sum(eigenvalues_all))
    var_explained = np.asarray(
        eigenvalues / total_var if total_var > 0 else eigenvalues / eigenvalues.sum(),
        dtype=float,
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
    extra_covariates: np.ndarray | None = None,
) -> np.ndarray:
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
        cv = np.asarray(extra_covariates, dtype=float)
        if cv.ndim == 1:
            cv = cv.reshape(-1, 1)
        X0 = np.column_stack([X0, cv])

    return X0
