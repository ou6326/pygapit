"""
MLMM - Multiple Loci Mixed Model.
Translates GAPIT.mlmm.R / GAPIT.mlmm_cof.R

Algorithm (stepwise forward/backward selection):
  1. Run MLM scan with fixed kinship K
  2. Add most significant marker as fixed cofactor
  3. Re-run MLM conditioned on all cofactors
  4. Repeat up to max_steps times
  5. Select optimal model by extended BIC (extBIC)
  6. Backward elimination: remove cofactors that become non-significant

Key difference from FarmCPU:
  K stays FIXED throughout (all-marker kinship).
  Cofactors are added as fixed effects TO the model WITH K.
  This creates partial confounding between cofactors and K,
  which FarmCPU solves by separating FEM and REM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import FloatMatrix, FloatVector, IntVector, readonly_copy
from ..stats.emma import emma_remle, emmax_p3d


@dataclass(frozen=True, slots=True)
class MLMMResult:
    p_values: FloatVector
    effects: FloatVector
    se: FloatVector
    stats: FloatVector
    selected_qtns: IntVector
    vg: float
    ve: float
    h2: float
    n_steps: int
    method: str = "MLMM"

    def __post_init__(self) -> None:
        for field in ("p_values", "effects", "se", "stats", "selected_qtns"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def _ext_bic(log_lik: float, n: int, k: int, m: int) -> float:
    """
    Extended BIC for multi-locus model selection.
    extBIC = -2*logL + k*log(n) + 2*k*log(m-1)
    where m = number of markers, k = number of parameters.
    Penalizes model complexity more heavily than standard BIC.
    Translates the opt_extBIC criterion from GAPIT.Bus.R
    """
    bic = -2.0 * log_lik + k * np.log(n) + 2.0 * k * np.log(max(m - 1, 1))
    return float(bic)


def mlmm_gwas(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    max_steps: int = 10,
    p_threshold: float = 1.2e-5,
    ngrids: int = 100,
) -> MLMMResult:
    """
    MLMM genome-wide association.
    Translates mlmm() from GAPIT.mlmm.R

    Parameters
    ----------
    y          : (n,) phenotype
    X0         : (n, q) covariate matrix
    GD         : (n, m) genotype matrix
    K          : (n, n) kinship matrix (fixed throughout)
    max_steps  : maximum forward selection steps (maxsteps=10 in R)
    p_threshold: threshold for adding cofactors (thresh=1.2e-5 in R)

    Returns
    -------
    MLMMResult with final p-values and selected QTN indices
    """
    n, m = GD.shape

    cofactors: list[int] = []  # list of selected SNP indices
    best_result = None  # best GLMResult from emmax_p3d
    best_bic = np.inf
    best_cofactors: list[int] = []
    best_vg, best_ve = 0.0, 0.0

    # ── Initial scan without cofactors ───────────────────────────────────
    result = emmax_p3d(y, X0, GD, K, ngrids=ngrids)
    current_vg = result.vg
    current_ve = result.ve
    best_result = result
    best_vg = current_vg
    best_ve = current_ve

    # Compute extBIC for null model
    null_reml = emma_remle(y, X0, K, ngrids=ngrids)
    current_bic = _ext_bic(null_reml.reml, n, X0.shape[1], m)
    best_bic = current_bic
    best_cofactors = []

    for _step in range(max_steps):
        # Find most significant SNP not already a cofactor
        p_vals = result.p_values.copy()
        p_vals[cofactors] = 1.0  # mask already-selected SNPs
        if np.nanmin(p_vals) > p_threshold:
            break

        new_snp = int(np.nanargmin(p_vals))
        cofactors.append(new_snp)

        # Build new X0 with cofactors added as fixed effects
        X_with_cof = np.column_stack([X0] + [GD[:, c] for c in cofactors])

        # Re-run EMMAX with extended X and same K
        try:
            result = emmax_p3d(y, X_with_cof, GD, K, ngrids=ngrids)
            current_vg = result.vg
            current_ve = result.ve
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            cofactors.pop()
            break

        # Compute REML LL for extBIC
        try:
            reml_result = emma_remle(y, X_with_cof, K, ngrids=ngrids)
            k_params = X_with_cof.shape[1]
            step_bic = _ext_bic(reml_result.reml, n, k_params, m)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            step_bic = best_bic + 1  # worse than current best

        if step_bic < best_bic - 1e-6:
            best_bic = step_bic
            best_result = result
            best_cofactors = cofactors.copy()
            best_vg = current_vg
            best_ve = current_ve

    # ── Backward elimination ─────────────────────────────────────────────
    # Remove cofactors that no longer contribute given the others
    if len(best_cofactors) > 1:
        improved = True
        while improved and len(best_cofactors) > 0:
            improved = False
            for c in list(best_cofactors):
                test_cofs = [x for x in best_cofactors if x != c]
                X_test = (
                    np.column_stack([X0] + [GD[:, cc] for cc in test_cofs])
                    if test_cofs
                    else X0
                )
                try:
                    reml_test = emma_remle(y, X_test, K, ngrids=ngrids)
                    k_test = X_test.shape[1]
                    bic_test = _ext_bic(reml_test.reml, n, k_test, m)
                    if bic_test < best_bic - 1e-6:
                        best_bic = bic_test
                        best_cofactors = test_cofs
                        improved = True
                        break
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    reml_test = None

    # ── Final scan with best cofactor set ────────────────────────────────
    if best_cofactors:
        X_final = np.column_stack([X0] + [GD[:, c] for c in best_cofactors])
    else:
        X_final = X0

    try:
        final_result = emmax_p3d(y, X_final, GD, K, ngrids=ngrids)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        final_result = best_result

    h2 = best_vg / (best_vg + best_ve) if (best_vg + best_ve) > 0 else 0.0

    return MLMMResult(
        p_values=final_result.p_values,
        effects=final_result.effects,
        se=final_result.se,
        stats=final_result.stats,
        selected_qtns=np.array(best_cofactors, dtype=int),
        vg=best_vg,
        ve=best_ve,
        h2=h2,
        n_steps=len(best_cofactors),
        method="MLMM",
    )
