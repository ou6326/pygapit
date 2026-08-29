"""Cross-language BLINK alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
from numpy.typing import NDArray

from pygapit.gwas.blink import _bic_select_cofactors, _ld_prune
from tests.cross_language.r_bridge import RBridge


def test_blink_ld_pruning_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
) -> None:
    """Compare ordered LD pruning using GAPIT's absolute-correlation cutoff."""
    duplicated = np.column_stack(
        [fixed_genotypes, fixed_genotypes[:, 0], 2.0 - fixed_genotypes[:, 1]]
    )
    candidates: NDArray[np.int_] = np.arange(duplicated.shape[1], dtype=np.int_)
    r_ld_remove = r_bridge.source_function(r_root, "GAPIT.Blink.R", "Blink.LDRemove")
    r_indices = (
        r_bridge.float_array(
            r_ld_remove(
                GDneo=r_bridge.matrix(duplicated),
                LD=0.7,
                Porder=r_bridge.float_vector(candidates.astype(float) + 1),
                orientation="col",
                bound=False,
            )
        ).astype(int)
        - 1
    )
    py_indices = _ld_prune(candidates, duplicated, ld_threshold=0.7)

    nt.assert_array_equal(py_indices, r_indices)


def test_blink_bic_selection_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
) -> None:
    """Compare prefix selection under BLINK's naive BIC criterion."""
    candidates = np.array([2, 0, 4, 1], dtype=int)
    phenotype_with_taxa = np.column_stack(
        [np.arange(len(fixed_phenotype), dtype=float), fixed_phenotype]
    )
    design = np.column_stack([np.ones(len(fixed_phenotype)), fixed_covariate])
    r_bic = r_bridge.source_function(r_root, "GAPIT.Blink.R", "Blink.BICselection")
    r_result = r_bic(
        Y=r_bridge.matrix(phenotype_with_taxa),
        Psort=r_bridge.float_vector(candidates.astype(float) + 1),
        CV=r_bridge.matrix(fixed_covariate[:, None]),
        GD=r_bridge.matrix(fixed_genotypes),
        orientation="col",
        BIC_method="naive",
    )
    r_indices = (
        r_bridge.float_array(r_bridge.component(r_result, "seqQTN")).astype(int) - 1
    )
    py_indices = _bic_select_cofactors(
        fixed_phenotype, design, fixed_genotypes, candidates
    )

    nt.assert_array_equal(py_indices, r_indices)
