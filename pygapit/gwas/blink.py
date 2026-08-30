"""
BLINK - Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway.
Translates Blink() and Blink.BICselection() from GAPIT.Blink.R

Algorithm (two iterating GLMs):
  Loop until convergence:
    GLM-1 (cofactor selection):
      - Sort markers by p-value
      - Apply LD pruning: remove markers in LD (|r| > LD_threshold) with top marker
      - Apply BIC selection: add cofactors while BIC decreases
    GLM-2 (marker testing):
      - Test all m markers with current cofactor set as fixed effects
      - Get new p-values

Key advantages over FarmCPU:
  - No kinship matrix required → GLM speed
  - LD-based pruning instead of bin approach → handles clustered QTNs
  - BIC replaces expensive REML for cofactor selection
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import FloatMatrix, FloatVector, IntVector
from .glm import glm_gwas, glm_scan_with_cofactors


@dataclass(frozen=True, slots=True)
class BLINKResult:
    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    t_stats: FloatVector
    selected_qtns: IntVector  # indices of pseudo-QTN cofactors
    n_iterations: int
    method: str = "BLINK"


def _compute_bic(
    y: FloatVector,
    X0: FloatMatrix,
    cofactor_indices: list[int],
    GD: FloatMatrix,
) -> float:
    """
    Compute BIC for model with given cofactors.
    BIC = -2*logL + k*log(n)
    where k = number of parameters, n = sample size.

    Translates Blink.BICselection() from GAPIT.Blink.R
    """
    n = len(y)
    if cofactor_indices:
        cof_mat = GD[:, np.array(cofactor_indices)]
        X = np.column_stack([X0, cof_mat])
    else:
        X = X0

    try:
        beta, _residuals, _rank, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        sse = np.sum((y - y_hat) ** 2)
    except np.linalg.LinAlgError:
        return np.inf

    if sse <= 0:
        sse = 1e-12

    k = X.shape[1]  # number of parameters
    # Log-likelihood under Gaussian errors
    log_lik = -0.5 * n * (np.log(2 * np.pi * sse / n) + 1)
    bic = -2.0 * log_lik + k * np.log(n)
    return float(bic)


def _ld_prune(
    candidate_indices: IntVector,
    GD: FloatMatrix,
    ld_threshold: float = 0.7,
) -> IntVector:
    """
    LD-based pruning of candidate SNPs.
    Translates Blink.LDRemove() from GAPIT.Blink.R

    Starting from the most significant marker, removes all markers
    in LD (|r| > threshold) with it. Repeats for next remaining marker.

    Parameters
    ----------
    candidate_indices : sorted candidate indices (best first by p-value)
    GD               : genotype matrix
    ld_threshold     : absolute-correlation threshold for LD pruning (default 0.7)

    Returns pruned set of indices (still ordered by significance).
    """
    if len(candidate_indices) == 0:
        return candidate_indices

    kept: list[int] = []
    remaining = [int(idx) for idx in candidate_indices]

    while remaining:
        ref = remaining[0]
        kept.append(ref)
        remaining.pop(0)

        if not remaining:
            break

        # Compute r² between ref SNP and all remaining
        g_ref = GD[:, ref].astype(float)
        g_ref_std = g_ref.std()
        if g_ref_std < 1e-8:
            continue

        to_remove: list[int] = []
        for idx in remaining:
            g_cand = GD[:, idx].astype(float)
            g_cand_std = g_cand.std()
            if g_cand_std < 1e-8:
                to_remove.append(idx)
                continue
            # Pearson r between ref and candidate
            cov = np.mean((g_ref - g_ref.mean()) * (g_cand - g_cand.mean()))
            r = cov / (g_ref_std * g_cand_std)
            if abs(r) > ld_threshold:
                to_remove.append(idx)

        for idx in to_remove:
            remaining.remove(idx)

    return np.array(kept, dtype=int)


def _bic_select_cofactors(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    candidates: IntVector,
) -> IntVector:
    """
    Greedily add cofactors from candidates while BIC decreases.
    Translates Blink.BICselection() from GAPIT.Blink.R

    Returns indices of selected cofactors.
    """
    current_cofactors: list[int] = []
    current_bic = _compute_bic(y, X0, [], GD)

    for idx in candidates:
        if idx in current_cofactors:
            continue
        test_cofactors = current_cofactors + [int(idx)]
        new_bic = _compute_bic(y, X0, test_cofactors, GD)
        if new_bic < current_bic - 1e-6:  # BIC improvement
            current_cofactors = test_cofactors
            current_bic = new_bic

    return np.array(current_cofactors, dtype=int)


def blink_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    max_iterations: int = 10,
    ld_threshold: float = 0.7,
    p_threshold: float | None = None,
    converge_threshold: float = 1.0,
) -> BLINKResult:
    """
    BLINK genome-wide association scan.
    Translates Blink() from GAPIT.Blink.R

    Parameters
    ----------
    y              : (n,) phenotype
    X0             : (n, q) covariate matrix (intercept + PCs)
    GD             : (n, m) genotype matrix, 0/1/2 coded
    max_iterations : maximum number of BLINK iterations (maxLoop in R)
    ld_threshold   : absolute-correlation threshold for LD pruning (LD in R)
    p_threshold    : p-value threshold to pre-select candidates
                     (default: Bonferroni = 1/m)
    converge_threshold : Jaccard similarity for convergence check

    Returns
    -------
    BLINKResult with final p_values and selected QTN indices
    """
    _n, m = GD.shape

    if p_threshold is None:
        p_threshold = 1.0 / m  # Bonferroni

    # ── Iteration 0: initial GLM scan (no cofactors) ─────────────────────
    glm_result = glm_gwas(y, X0, GD)
    p_values = glm_result.p_values.copy()

    current_qtns = np.array([], dtype=int)
    np.array([-1], dtype=int)  # sentinel to trigger first iter
    n_iter = 0

    for iteration in range(max_iterations):
        n_iter = iteration + 1

        # ── GLM-1: Select cofactors ───────────────────────────────────────
        # Step 1: get significant candidates from current p-values
        sig_mask = (p_values <= p_threshold) & ~np.isnan(p_values)
        candidate_idx = np.where(sig_mask)[0]

        if len(candidate_idx) == 0:
            # No significant hits → converged with empty QTN set
            break

        # Sort candidates by p-value (best first)
        candidate_order = np.argsort(p_values[candidate_idx])
        candidates_sorted = candidate_idx[candidate_order]

        # Step 2: LD pruning
        candidates_pruned = _ld_prune(candidates_sorted, GD, ld_threshold)

        if len(candidates_pruned) == 0:
            break

        # Step 3: BIC selection
        new_qtns = _bic_select_cofactors(y, X0, GD, candidates_pruned)

        # ── Convergence check ─────────────────────────────────────────────
        # Jaccard similarity between current and previous QTN sets
        if len(new_qtns) > 0 and len(current_qtns) > 0:
            intersection = len(np.intersect1d(new_qtns, current_qtns))
            union = len(np.union1d(new_qtns, current_qtns))
            jaccard = intersection / union if union > 0 else 0.0
        elif len(new_qtns) == 0 and len(current_qtns) == 0:
            jaccard = 1.0
        else:
            jaccard = 0.0

        current_qtns = new_qtns

        # ── GLM-2: Test all markers with updated cofactors ───────────────
        glm_result = glm_scan_with_cofactors(y, X0, GD, current_qtns)
        p_values = glm_result.p_values.copy()

        if jaccard >= converge_threshold:
            break

    # Final scan with last cofactor set
    final_result = glm_scan_with_cofactors(y, X0, GD, current_qtns)

    return BLINKResult(
        p_values=final_result.p_values,
        effects=final_result.effects,
        se=final_result.se,
        t_stats=final_result.t_stats,
        selected_qtns=current_qtns,
        n_iterations=n_iter,
        method="BLINK",
    )
