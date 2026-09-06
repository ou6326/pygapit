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
from scipy.linalg import eigh as scipy_eigh

from .._resources import (
    DEFAULT_MARKER_WORKSPACE_MIB,
    iter_marker_slices,
    validate_marker_workspace_mib,
)
from .._typing import (
    FloatMatrix,
    FloatVector,
    as_float_matrix,
    readonly_copy,
    require_row_count,
)
from ..io.storage import GenotypeStore, as_genotype_store


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
    GD: FloatMatrix | GenotypeStore,
    n_components: int = 3,
    maf_filter: float = 0.05,
    *,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
) -> PCAResult:
    """
    Compute principal components from genotype matrix.
    Translates GAPIT.PCA.R

    Parameters
    ----------
    GD : genotype matrix or GenotypeStore, with shape (n_individuals, n_snps)
        Marker values are 0/1/2 coded. Stores are read in bounded marker blocks.
    n_components : number of PCs to return (PCA.total parameter in GAPIT)
    maf_filter : minimum MAF for SNPs used in PCA
    marker_workspace_mib : target MiB for one centered marker batch

    Returns
    -------
    PCAResult with scores, loadings, variance explained
    """
    marker_workspace_mib = validate_marker_workspace_mib(marker_workspace_mib)
    genotype = as_genotype_store(GD)
    n, total_marker_count = genotype.shape
    if n < 2:
        raise ValueError("PCA requires at least two individuals")
    if n_components < 0:
        raise ValueError("n_components must be non-negative")

    # ── MAF filter ────────────────────────────────────────────────────────
    freq = np.empty(total_marker_count, dtype=np.float64)
    for marker_slice in iter_marker_slices(
        n,
        total_marker_count,
        marker_workspace_mib,
    ):
        block = genotype.read_markers(marker_slice)
        freq[marker_slice] = np.sum(block, axis=0) / (2.0 * n)
        del block
    maf = np.minimum(freq, 1.0 - freq)
    valid_snps = maf >= maf_filter
    if valid_snps.sum() < n_components:
        valid_snps = maf > 0  # relax if too few pass filter

    # ── Leading singular triplets through the smaller Gram matrix ───────
    marker_count = int(np.sum(valid_snps))
    k = min(n_components, n, marker_count)
    if k == 0:
        scores = np.empty((n, 0), dtype=np.float64)
        loadings = np.empty((marker_count, 0), dtype=np.float64)
        eigenvalues = np.empty(0, dtype=np.float64)
        total_var = np.float64(0.0)
    elif n <= marker_count:
        gram: FloatMatrix = np.zeros((n, n), dtype=np.float64)
        for marker_slice in iter_marker_slices(
            n,
            total_marker_count,
            marker_workspace_mib,
        ):
            batch_valid = valid_snps[marker_slice]
            if not batch_valid.any():
                continue
            centered = genotype.read_markers(marker_slice)[:, batch_valid]
            centered -= 2.0 * freq[marker_slice][batch_valid]
            gram += centered @ centered.T
            del centered
        wide_total_sum: np.float64 = np.trace(gram)
        total_var = wide_total_sum / (n - 1)
        gram_values, left_vectors = scipy_eigh(
            gram,
            subset_by_index=(n - k, n - 1),
            check_finite=False,
        )
        gram_values = np.maximum(gram_values[::-1], 0.0)
        left_vectors = left_vectors[:, ::-1]
        singular_values: FloatVector = np.sqrt(gram_values)
        scores = left_vectors * singular_values
        loadings = np.zeros((marker_count, k), dtype=np.float64)
        nonzero = singular_values > np.finfo(np.float64).eps * singular_values[0]
        safe_singular_values = np.where(nonzero, singular_values, 1.0)
        loading_start = 0
        for marker_slice in iter_marker_slices(
            n,
            total_marker_count,
            marker_workspace_mib,
        ):
            batch_valid = valid_snps[marker_slice]
            batch_marker_count = int(np.sum(batch_valid))
            if batch_marker_count == 0:
                continue
            centered = genotype.read_markers(marker_slice)[:, batch_valid]
            centered -= 2.0 * freq[marker_slice][batch_valid]
            batch_loadings = (centered.T @ left_vectors) / safe_singular_values
            batch_loadings[:, ~nonzero] = 0.0
            loadings[loading_start : loading_start + batch_marker_count] = (
                batch_loadings
            )
            loading_start += batch_marker_count
            del centered, batch_loadings
        eigenvalues = gram_values / (n - 1)
    else:
        GD_centered: FloatMatrix = np.empty((n, marker_count), dtype=np.float64)
        centered_start = 0
        for marker_slice in iter_marker_slices(
            n,
            total_marker_count,
            marker_workspace_mib,
        ):
            batch_valid = valid_snps[marker_slice]
            batch_marker_count = int(np.sum(batch_valid))
            if batch_marker_count == 0:
                continue
            centered = genotype.read_markers(marker_slice)[:, batch_valid]
            centered -= 2.0 * freq[marker_slice][batch_valid]
            GD_centered[:, centered_start : centered_start + batch_marker_count] = (
                centered
            )
            centered_start += batch_marker_count
            del centered
        tall_total_sum: np.float64 = np.einsum(
            "ij,ij->",
            GD_centered,
            GD_centered,
        )
        total_var = tall_total_sum / (n - 1)
        gram = GD_centered.T @ GD_centered
        gram_values, loadings = scipy_eigh(
            gram,
            subset_by_index=(marker_count - k, marker_count - 1),
            check_finite=False,
        )
        gram_values = np.maximum(gram_values[::-1], 0.0)
        loadings = loadings[:, ::-1]
        scores = GD_centered @ loadings
        eigenvalues = gram_values / (n - 1)

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
