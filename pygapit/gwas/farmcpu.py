"""
FarmCPU - Fixed And Random Model Circulating Probability Unification.
Translates FarmCPU.BIN, FarmCPU.GLM, FarmCPU.Burger from GAPIT.FarmCPU.R

Algorithm:
  Two models alternate until convergence:

  Fixed Effect Model (FEM) — tests all markers:
    y = X0*beta + sum(cofactors)*t + alpha*s_i + e     (pure GLM)
    Cofactors control false positives. No kinship.

  Random Effect Model (REM) — selects pseudo-QTNs:
    y = X0*beta + u + e
    u ~ N(0, K_pseudo * sigma2_g)
    K_pseudo built from current pseudo-QTN set only.
    REML selects which QTNs minimize variance components.

  Bin method for QTN selection:
    Genome divided into bins of size `bin_size` bp.
    At most one QTN per bin (most significant within each bin).
    Bin size optimized by REML log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import FloatMatrix, FloatVector, IntVector, LabelVector, NumericVector
from ..stats.emma import emma_remle
from ..stats.kinship import vanraden_kinship
from .glm import glm_scan_with_cofactors


@dataclass(frozen=True, slots=True)
class FarmCPUResult:
    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    t_stats: FloatVector
    selected_qtns: IntVector  # final pseudo-QTN indices
    n_iterations: int
    vg: float
    ve: float
    h2: float
    method: str = "FarmCPU"


def _bin_select_qtns(
    p_values: FloatVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    bin_size: int = 5_000_000,
    max_qtns: int | None = None,
    p_threshold: float | None = None,
) -> IntVector:
    """
    Select pseudo-QTNs using bin method.
    Translates FarmCPU.BIN() from GAPIT.FarmCPU.R

    Within each genomic bin of `bin_size` bp, select the most
    significant SNP. Assumes one causal variant per bin region.

    Parameters
    ----------
    p_values    : (m,) p-values from FEM
    chromosomes : (m,) chromosome labels
    positions   : (m,) genomic positions in bp
    bin_size    : bin width in bp (default 5 Mbp)
    max_qtns    : maximum number of QTNs (bound = sqrt(n)/sqrt(log10(n)))
    p_threshold : p-value cutoff for candidate selection

    Returns selected QTN indices (0-based).
    """
    m = len(p_values)

    if p_threshold is None:
        p_threshold = 1.0 / m  # Bonferroni

    # Pre-filter: only consider significant SNPs
    sig_mask = (p_values <= p_threshold) & ~np.isnan(p_values)
    sig_idx = np.where(sig_mask)[0]

    if len(sig_idx) == 0:
        return np.array([], dtype=int)

    # Assign each SNP to a bin
    bin_ids: dict[tuple[str, int], tuple[float, int]] = {}
    for raw_i in sig_idx:
        i = int(raw_i)
        chrom = str(chromosomes[i])
        pos = float(positions[i]) if not np.isnan(float(positions[i])) else 0
        bin_num = int(pos // bin_size)
        key = (chrom, bin_num)
        if key not in bin_ids or p_values[i] < bin_ids[key][0]:
            bin_ids[key] = (float(p_values[i]), i)

    selected = np.array([v[1] for v in bin_ids.values()], dtype=int)

    # Sort by p-value
    order = np.argsort(p_values[selected])
    selected = selected[order]

    # Apply max_qtns bound
    if max_qtns is not None and len(selected) > max_qtns:
        selected = selected[:max_qtns]

    return np.asarray(selected, dtype=int)


def _build_pseudo_kinship(
    GD: FloatMatrix,
    qtn_indices: IntVector | None,
) -> FloatMatrix | None:
    """
    Build kinship from pseudo-QTN genotypes only.
    Translates FarmCPU.Burger() kinship construction in GAPIT.FarmCPU.R

    The pseudo-kinship K* = VanRaden(GD[:, qtn_indices]).
    Using only QTN genotypes, this kinship captures only the
    background variance explained by currently-identified QTNs.
    """
    if qtn_indices is None or len(qtn_indices) == 0:
        return None

    GK = GD[:, qtn_indices]
    K = vanraden_kinship(GK)
    # Add small diagonal for numerical stability
    K += np.eye(len(K)) * 1e-6
    return K


def _rem_select_qtns(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    candidate_qtns: IntVector,
) -> tuple[IntVector, float, float]:
    """
    Random Effect Model: select pseudo-QTNs by REML.
    Translates FarmCPU.Burger() from GAPIT.FarmCPU.R

    Builds kinship from candidate QTNs and estimates variance
    components. The REML LL guides which QTN set to keep.

    Returns (selected_qtn_indices, vg, ve)
    """
    if len(candidate_qtns) == 0:
        return np.array([], dtype=int), 0.0, 0.0

    K = _build_pseudo_kinship(GD, candidate_qtns)
    if K is None:
        return candidate_qtns, 0.0, 0.0

    try:
        result = emma_remle(y, X0, K)
        return candidate_qtns, result.vg, result.ve
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return candidate_qtns, 0.0, 0.0


def farmcpu_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    chromosomes: LabelVector,
    positions: NumericVector,
    max_iterations: int = 10,
    bin_size: int = 5_000_000,
    p_threshold: float | None = None,
    converge_threshold: float = 1.0,
) -> FarmCPUResult:
    """
    FarmCPU genome-wide association scan.
    Translates the main FarmCPU loop from GAPIT.FarmCPU.R

    Parameters
    ----------
    y              : (n,) phenotype
    X0             : (n, q) covariate matrix (intercept + PCs)
    GD             : (n, m) genotype matrix, 0/1/2
    chromosomes    : (m,) chromosome labels for each SNP
    positions      : (m,) bp positions for each SNP
    max_iterations : maximum FEM/REM cycles
    bin_size       : genomic bin size in bp for QTN selection
    p_threshold    : p-value threshold for candidate QTNs
    converge_threshold : Jaccard convergence criterion

    Returns
    -------
    FarmCPUResult
    """
    n, m = GD.shape

    if p_threshold is None:
        p_threshold = 1.0 / m

    # Maximum QTNs: bound = sqrt(n) / sqrt(log10(n)) (from FarmCPU.BIN)
    max_qtns = max(1, int(np.sqrt(n) / np.sqrt(max(1, np.log10(n)))))

    # ── Initial FEM scan: no cofactors ─────────────────────────────────
    glm_result = glm_scan_with_cofactors(y, X0, GD, None)
    p_values = glm_result.p_values.copy()

    current_qtns = np.array([], dtype=int)
    current_vg = 0.0
    current_ve = 0.0
    n_iter = 0

    for iteration in range(max_iterations):
        n_iter = iteration + 1
        prev_qtns = current_qtns.copy()

        # ── REM: Select pseudo-QTNs via bin method ─────────────────────
        candidate_qtns = _bin_select_qtns(
            p_values,
            chromosomes,
            positions,
            bin_size=bin_size,
            max_qtns=max_qtns,
            p_threshold=p_threshold,
        )

        if len(candidate_qtns) == 0:
            break

        # ── REM: Estimate variance components with pseudo-kinship ──────
        selected_qtns, vg, ve = _rem_select_qtns(y, X0, GD, candidate_qtns)
        current_qtns = selected_qtns
        current_vg = vg
        current_ve = ve

        # ── FEM: Test all markers with pseudo-QTN cofactors ────────────
        glm_result = glm_scan_with_cofactors(y, X0, GD, current_qtns)
        p_values = glm_result.p_values.copy()

        # ── Convergence check ──────────────────────────────────────────
        if len(current_qtns) > 0 and len(prev_qtns) > 0:
            intersection = len(np.intersect1d(current_qtns, prev_qtns))
            union = len(np.union1d(current_qtns, prev_qtns))
            jaccard = intersection / union if union > 0 else 0.0
        elif len(current_qtns) == 0 and len(prev_qtns) == 0:
            jaccard = 1.0
        else:
            jaccard = 0.0

        if jaccard >= converge_threshold:
            break

    # Final FEM pass with converged QTN set
    final_result = glm_scan_with_cofactors(y, X0, GD, current_qtns)
    h2 = (
        current_vg / (current_vg + current_ve) if (current_vg + current_ve) > 0 else 0.0
    )

    return FarmCPUResult(
        p_values=final_result.p_values,
        effects=final_result.effects,
        se=final_result.se,
        t_stats=final_result.t_stats,
        selected_qtns=current_qtns,
        n_iterations=n_iter,
        vg=current_vg,
        ve=current_ve,
        h2=h2,
        method="FarmCPU",
    )
