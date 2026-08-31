"""
PyGAPIT LD Utilities - Linkage Disequilibrium calculations.

Bug fixes (v1.0.1):
  - LD_matrix: constant/invariant SNPs now return r²=NaN instead of crashing
    or producing undefined values; diagonal forced to 1.0; result clipped to [0,1].
  - LD_decay:  same NaN-guard for constant columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .._typing import FloatMatrix, FloatVector, Matrix


def _safe_r2(a: FloatVector, b: FloatVector) -> float:
    """Pearson r² with guard for constant/invariant inputs.

    Returns NaN if either column has zero variance (undefined correlation).
    """
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = np.asarray(stats.pearsonr(a, b)[0], dtype=float).item()
    return np.clip(r**2, 0.0, 1.0)  # clip floating-point noise


def LD_matrix(genotype: Matrix, max_snps: int = 500) -> FloatMatrix:
    """
    Compute pairwise LD (r²) matrix for up to *max_snps* SNPs.

    Parameters
    ----------
    genotype : np.ndarray (n, m)
    max_snps : int  -- trim to this many if m > max_snps

    Returns
    -------
    r2_matrix : np.ndarray (m', m')
        Values in [0, 1]; NaN for invariant SNP pairs (undefined).
        Diagonal is always 1.0.
    """
    G = np.nan_to_num(genotype[:, :max_snps].astype(float))
    m = G.shape[1]
    r2 = np.full((m, m), np.nan)  # default NaN; filled below
    np.fill_diagonal(r2, 1.0)  # diagonal always 1

    for i in range(m):
        if np.std(G[:, i]) < 1e-12:  # invariant SNP — leave row/col as NaN
            continue
        for j in range(i + 1, m):
            v = _safe_r2(G[:, i], G[:, j])
            r2[i, j] = r2[j, i] = v

    return r2


def LD_decay(
    genotype: Matrix,
    snp_info: pd.DataFrame,
    chrom: int | None = None,
    max_dist: int = 1_000_000,
) -> pd.DataFrame:
    """
    Compute LD decay (r² vs physical distance).

    Parameters
    ----------
    genotype  : np.ndarray (n, m)
    snp_info  : pd.DataFrame with Chromosome, Position columns
    chrom     : int -- restrict to this chromosome (None = first found)
    max_dist  : int -- max bp distance

    Returns
    -------
    pd.DataFrame with columns: distance, r2
        Pairs involving invariant SNPs are silently dropped (r² undefined).
    """
    if chrom is None:
        chrom = snp_info["Chromosome"].iloc[0]
    mask = snp_info["Chromosome"] == chrom
    idx = np.where(mask)[0]
    G = np.nan_to_num(genotype[:, idx].astype(float))
    pos = snp_info["Position"].values[idx]

    records: list[dict[str, float | int]] = []
    m = len(idx)
    for i in range(m):
        if np.std(G[:, i]) < 1e-12:
            continue
        for j in range(i + 1, m):
            d = abs(int(float(str(pos[j]))) - int(float(str(pos[i]))))
            if d > max_dist:
                break
            v = _safe_r2(G[:, i], G[:, j])
            if not np.isnan(v):
                records.append({"distance": d, "r2": v})

    return pd.DataFrame(records)
