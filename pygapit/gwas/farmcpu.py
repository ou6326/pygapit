"""
FarmCPU - Fixed And Random Model Circulating Probability Unification.
Translates GAPIT.FarmCPU.R's static-bin FarmCPU.BIN and FarmCPU.GLM path.

Static-bin/FEM loop:
  1. Run an initial FEM scan with no pseudo-QTN cofactors.
  2. Select the best significant marker from each genomic bin.
  3. Re-scan with those pseudo-QTNs as fixed FEM cofactors.
  4. Repeat until the selected set converges or reaches ``max_iterations``.

The exposed Python workflow implements GAPIT's ``method.bin='static'`` path.
It selects pseudo-QTNs from FEM p-values and does not run the ``Burger`` REML
optimization used only by GAPIT's ``method.bin='optimum'`` branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._resources import DEFAULT_MARKER_WORKSPACE_MIB, validate_marker_workspace_mib
from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    LabelVector,
    NumericVector,
    readonly_copy,
)
from ..io.storage import GenotypeStore, as_genotype_store
from .glm import glm_scan_with_cofactors, reward_substitute_cofactor_statistics


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

    def __post_init__(self) -> None:
        for field in ("p_values", "effects", "se", "t_stats", "selected_qtns"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


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
        pos = positions[i] if not np.isnan(positions[i]) else 0.0
        bin_num = int(pos // bin_size)
        key = (chrom, bin_num)
        if key not in bin_ids or p_values[i] < bin_ids[key][0]:
            bin_ids[key] = (p_values[i], i)

    selected = np.array([v[1] for v in bin_ids.values()], dtype=int)

    # Sort by p-value
    order = np.argsort(p_values[selected])
    selected = selected[order]

    # Apply max_qtns bound
    if max_qtns is not None and len(selected) > max_qtns:
        selected = selected[:max_qtns]

    return np.asarray(selected, dtype=int)


def farmcpu_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix | GenotypeStore,
    chromosomes: LabelVector,
    positions: NumericVector,
    max_iterations: int = 10,
    bin_size: int = 5_000_000,
    p_threshold: float | None = None,
    converge_threshold: float = 1.0,
    *,
    marker_workspace_mib: float = DEFAULT_MARKER_WORKSPACE_MIB,
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
    max_iterations : maximum static-bin/FEM cycles
    bin_size       : genomic bin size in bp for QTN selection
    p_threshold    : p-value threshold for candidate QTNs
    converge_threshold : Jaccard convergence criterion

    Returns
    -------
    FarmCPUResult
    """
    marker_workspace_mib = validate_marker_workspace_mib(marker_workspace_mib)
    genotype = as_genotype_store(GD)
    n, m = genotype.shape

    if p_threshold is None:
        p_threshold = 1.0 / m

    # GAPIT uses R's round() for the FarmCPU.BIN upper bound.
    max_qtns = max(1, round(np.sqrt(n) / np.sqrt(max(1, np.log10(n)))))

    # ── Initial FEM scan: no cofactors ─────────────────────────────────
    glm_result = glm_scan_with_cofactors(
        y,
        X0,
        genotype,
        None,
        marker_workspace_mib=marker_workspace_mib,
    )
    p_values = glm_result.p_values.copy()

    current_qtns = np.array([], dtype=int)
    n_iter = 0

    for iteration in range(max_iterations):
        n_iter = iteration + 1
        prev_qtns = current_qtns.copy()

        # ── Static bin: Select pseudo-QTNs from FEM p-values ───────────
        candidate_qtns = _bin_select_qtns(
            p_values,
            chromosomes,
            positions,
            bin_size=bin_size,
            max_qtns=max_qtns,
            p_threshold=p_threshold,
        )
        if iteration > 0 and len(prev_qtns) > 0:
            candidate_qtns = np.asarray(
                list(dict.fromkeys([*candidate_qtns, *prev_qtns])), dtype=int
            )

        if len(candidate_qtns) == 0:
            break

        current_qtns = candidate_qtns

        # ── FEM: Test all markers with pseudo-QTN cofactors ────────────
        glm_result = reward_substitute_cofactor_statistics(
            glm_scan_with_cofactors(
                y,
                X0,
                genotype,
                current_qtns,
                marker_workspace_mib=marker_workspace_mib,
            ),
            y,
            X0,
            genotype,
            current_qtns,
            marker_workspace_mib=marker_workspace_mib,
        )
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
    final_result = reward_substitute_cofactor_statistics(
        glm_scan_with_cofactors(
            y,
            X0,
            genotype,
            current_qtns,
            marker_workspace_mib=marker_workspace_mib,
        ),
        y,
        X0,
        genotype,
        current_qtns,
        marker_workspace_mib=marker_workspace_mib,
    )
    return FarmCPUResult(
        p_values=final_result.p_values,
        effects=final_result.effects,
        se=final_result.se,
        t_stats=final_result.t_stats,
        selected_qtns=current_qtns,
        n_iterations=n_iter,
        # GAPIT's static-bin FarmCPU wrapper returns no variance components.
        vg=0.0,
        ve=0.0,
        h2=0.0,
        method="FarmCPU",
    )
