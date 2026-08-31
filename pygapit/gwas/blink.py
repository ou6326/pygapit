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

import typing as t
from dataclasses import dataclass

import numpy as np

from .._typing import BoolVector, FloatMatrix, FloatVector, IntVector, readonly_copy
from .glm import (
    GLMResult,
    glm_gwas,
    glm_scan_with_cofactors,
    reward_substitute_cofactor_statistics,
)


@dataclass(frozen=True, slots=True)
class BLINKResult:
    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    t_stats: FloatVector
    selected_qtns: IntVector  # indices of pseudo-QTN cofactors
    n_iterations: int
    method: str = "BLINK"

    def __post_init__(self) -> None:
        for field in ("p_values", "effects", "se", "t_stats", "selected_qtns"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


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
    Select the candidate prefix minimizing GAPIT 3.5's naive BIC.

    Returns indices of selected cofactors.
    """
    if len(candidates) == 0:
        return candidates

    n = len(y)
    threshold = max(1, int(np.floor(n / np.log(n))))
    ordered = np.asarray(candidates[:threshold], dtype=int)
    bic = np.empty(len(ordered), dtype=np.float64)
    for prefix_size in range(1, len(ordered) + 1):
        design = np.column_stack([X0, GD[:, ordered[:prefix_size]]])
        beta = np.linalg.pinv(design) @ y
        residual = design @ beta - y
        variance = np.var(residual, ddof=1)
        if variance <= 0.0:
            variance = np.finfo(np.float64).tiny
        negative_twice_log_likelihood = (
            n * np.log(2.0 * np.pi)
            + n * np.log(variance)
            + (residual @ residual) / variance
        )
        penalty = (design.shape[1] - 1) * np.log(n)
        bic[prefix_size - 1] = negative_twice_log_likelihood + penalty

    best_prefix_size = int(np.argmin(bic)) + 1
    return ordered[:best_prefix_size]


def _candidate_mask(
    p_values: FloatVector,
    p_threshold: float,
    fdr_alpha: float | None,
) -> BoolVector:
    """Select candidates with GAPIT 3.5's FDR or a fixed p-value cutoff."""
    if fdr_alpha is not None:
        finite = p_values[np.isfinite(p_values)]
        if len(finite) == 0:
            return np.zeros(len(p_values), dtype=bool)
        sorted_p = np.sort(finite)
        distances = np.abs(fdr_alpha - sorted_p * len(p_values) / fdr_alpha)
        index_fdr = int(np.argmin(distances)) + 1
        fdr_cutoff = fdr_alpha * index_fdr / len(p_values)
        fdr_mask: BoolVector = (p_values < fdr_cutoff) & np.isfinite(p_values)
        return fdr_mask
    fixed_mask: BoolVector = (p_values <= p_threshold) & ~np.isnan(p_values)
    return fixed_mask


def _calibrate_no_qtn_p_values(p_values: FloatVector) -> FloatVector:
    """Apply GAPIT 3.5's BLINK fallback when no pseudo-QTN is selected."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p_glm_log = -np.log10(np.nanquantile(p_values, 0.05))
        bonferroni_comparison = p_glm_log / 1.3
        if not np.isfinite(bonferroni_comparison) or bonferroni_comparison <= 0.0:
            return p_values.copy()
        farmcpu_log = -np.log10(p_values) / bonferroni_comparison
        calibrated = t.cast(FloatVector, np.power(10.0, -farmcpu_log))
    calibrated[calibrated > 1.0] = 1.0
    return calibrated


def blink_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    max_iterations: int = 10,
    ld_threshold: float = 0.7,
    p_threshold: float | None = None,
    fdr_alpha: float | None = None,
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
    fdr_alpha      : GAPIT 3.5 FDR cutoff level for candidate selection. An
                     explicit p_threshold takes precedence.
    converge_threshold : Jaccard similarity for convergence check

    Returns
    -------
    BLINKResult with final p_values and selected QTN indices
    """
    _n, m = GD.shape

    if fdr_alpha is not None and not 0.0 < fdr_alpha <= 1.0:
        raise ValueError("fdr_alpha must be between 0 and 1")
    use_fdr = fdr_alpha is not None and p_threshold is None
    if p_threshold is None:
        p_threshold = 1.0 / m  # Bonferroni

    # ── Iteration 0: initial GLM scan (no cofactors) ─────────────────────
    glm_result = glm_gwas(y, X0, GD)
    p_values = glm_result.p_values.copy()

    current_qtns = np.array([], dtype=int)
    np.array([-1], dtype=int)  # sentinel to trigger first iter
    n_iter = 0
    no_qtn_p_values: FloatVector | None = None

    for iteration in range(max_iterations):
        n_iter = iteration + 1

        # ── GLM-1: Select cofactors ───────────────────────────────────────
        # Step 1: get significant candidates from current p-values
        sig_mask = _candidate_mask(
            p_values,
            p_threshold,
            fdr_alpha if use_fdr else None,
        )
        candidate_idx = np.where(sig_mask)[0]

        if len(candidate_idx) == 0:
            if len(current_qtns) == 0:
                no_qtn_p_values = _calibrate_no_qtn_p_values(p_values)
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
        if iteration > 0 and len(current_qtns) > 0:
            new_qtns = np.asarray(
                list(dict.fromkeys([*new_qtns, *current_qtns])), dtype=int
            )
            if len(candidates_sorted) > 1 and len(new_qtns) > 1:
                new_qtns = _bic_select_cofactors(y, X0, GD, new_qtns)

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
        glm_result = reward_substitute_cofactor_statistics(
            glm_scan_with_cofactors(y, X0, GD, current_qtns),
            y,
            X0,
            GD,
            current_qtns,
        )
        p_values = glm_result.p_values.copy()

        if jaccard >= converge_threshold:
            break

    # Final scan with last cofactor set
    final_result = reward_substitute_cofactor_statistics(
        glm_scan_with_cofactors(y, X0, GD, current_qtns),
        y,
        X0,
        GD,
        current_qtns,
    )
    if no_qtn_p_values is not None:
        final_result = GLMResult(
            no_qtn_p_values,
            final_result.effects,
            final_result.se,
            final_result.t_stats,
            final_result.r2_full,
        )

    return BLINKResult(
        p_values=final_result.p_values,
        effects=final_result.effects,
        se=final_result.se,
        t_stats=final_result.t_stats,
        selected_qtns=current_qtns,
        n_iterations=n_iter,
        method="BLINK",
    )
