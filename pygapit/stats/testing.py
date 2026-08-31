"""
Statistical testing utilities.
Translates GAPIT.Perform.BH.FDR.Multiple.Correction.Procedure.R and GAPIT.FDR.TypeI.R
"""

from __future__ import annotations

import typing as t

import numpy as np
from scipy.stats import chi2

from .._typing import BoolVector, FloatVector, Vector

SignificanceResult = dict[str, float | int | BoolVector | FloatVector]


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Genome-wide Bonferroni threshold: alpha / m."""
    return alpha / n_tests


def benjamini_hochberg(p_values: FloatVector, alpha: float = 0.05) -> FloatVector:
    """
    Benjamini-Hochberg FDR correction.
    Translates GAPIT.Perform.BH.FDR.Multiple.Correction.Procedure.R

    Returns array of adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranks = np.arange(1, n + 1)
    # BH step-up: adjusted = p * n / rank
    adjusted = p_values[order] * n / ranks
    # Enforce monotonicity (cumulative minimum from right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # Restore original order
    result = np.empty(n)
    result[order] = adjusted
    return result


def genomic_inflation_factor(p_values: FloatVector) -> float:
    """
    Genomic inflation factor lambda.
    lambda = median(chi2_observed) / 0.4549
    lambda ~ 1.0 = well-controlled
    lambda > 1.1 = inflation (population structure not fully controlled)
    """
    valid = p_values[(p_values > 0) & (p_values < 1) & ~np.isnan(p_values)]
    if len(valid) == 0:
        return 1.0
    chi2_obs = t.cast(FloatVector, chi2.ppf(1.0 - valid, df=1))
    lam = np.median(chi2_obs) / 0.4549  # 0.4549 = median of chi2(1)
    return lam.item()


def get_significant_snps(
    p_values: FloatVector,
    snp_names: Vector,
    chromosomes: Vector,
    positions: Vector,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> SignificanceResult:
    """
    Identify significant SNPs after multiple testing correction.

    Parameters
    ----------
    p_values : array of p-values
    snp_names, chromosomes, positions : SNP annotation arrays
    method : 'bonferroni' or 'fdr_bh'
    alpha : significance level

    Returns
    -------
    dict with 'threshold', 'significant_mask', 'adj_pvalues'
    """
    m = len(p_values)

    if method == "bonferroni":
        threshold = bonferroni_threshold(m, alpha)
        sig_mask = p_values <= threshold
        adj_pvalues = np.minimum(p_values * m, 1.0)
    elif method == "fdr_bh":
        adj_pvalues = benjamini_hochberg(p_values, alpha)
        threshold = alpha
        sig_mask = adj_pvalues <= threshold
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bonferroni' or 'fdr_bh'")

    return {
        "threshold": threshold,
        "significant_mask": sig_mask,
        "adj_pvalues": adj_pvalues,
        "n_significant": int(sig_mask.sum()),
    }
