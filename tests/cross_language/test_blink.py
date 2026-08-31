"""Cross-language BLINK alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
from numpy.typing import NDArray

from pygapit.gapit import GAPIT
from pygapit.gwas.blink import (
    _bic_select_cofactors,
    _candidate_mask,
    _ld_prune,
)
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


def test_blink_fdr_candidate_mask_matches_gapit_3_5(r_bridge: RBridge) -> None:
    """Compare GAPIT's data-dependent FDR cutoff used for pseudo-QTNs."""
    p_values = np.array([0.0001, 0.001, 0.01, 0.04, 0.5], dtype=np.float64)
    alpha = 0.05
    r_mask_function = r_bridge.function(
        "function(p, cutOff) {"
        " nm <- length(p); sp <- sort(p);"
        " spd <- abs(cutOff - sp * nm / cutOff);"
        " index_fdr <- grep(min(spd), spd)[1];"
        " FDRcutoff <- cutOff * index_fdr / nm;"
        " as.numeric(p < FDRcutoff)"
        "}"
    )
    r_mask = r_bridge.float_array(
        r_mask_function(r_bridge.float_vector(p_values), alpha)
    ).astype(bool)
    py_mask = _candidate_mask(
        p_values, p_threshold=1.0 / len(p_values), fdr_alpha=alpha
    )

    nt.assert_array_equal(py_mask, r_mask)


def test_blink_iterative_workflow_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare the complete regular-matrix BLINK iteration and final scan."""
    marker_count = fixed_genotypes.shape[1]
    phenotype = np.column_stack(
        [np.arange(len(fixed_phenotype), dtype=float), fixed_phenotype]
    )
    marker_map = np.column_stack(
        [
            np.arange(1, marker_count + 1, dtype=float),
            np.array([1, 1, 1, 2, 2, 2], dtype=float),
            np.array([100, 200, 1500, 100, 200, 1500], dtype=float),
        ]
    )
    r_bridge.source(r_root, "GAPIT.Specify.R")
    r_bridge.source_for_regular_matrices(r_root, "GAPIT.FarmCPU.R")
    r_bridge.source_for_regular_matrices(r_root, "GAPIT.Blink.R")
    r_blink = r_bridge.function("Blink")
    r_marker_map = r_bridge.function("as.data.frame")(
        r_bridge.matrix(marker_map, column_names=["SNP", "Chr", "Pos"])
    )
    r_result = r_blink(
        Y=r_bridge.matrix(phenotype, column_names=["Taxa", "Trait"]),
        GD=r_bridge.matrix(fixed_genotypes),
        GM=r_marker_map,
        file_output=False,
        maxLoop=5,
        LD=0.7,
        p_threshold=0.1,
        maf_threshold=0.0,
        converge=1.0,
    )
    phenotype_frame, genotype_frame, marker_frame = fixed_gapit_inputs
    py_result = GAPIT(
        Y=phenotype_frame,
        GD=genotype_frame,
        GM=marker_frame,
        model="BLINK",
        PCA_total=0,
        maf_threshold=0.0,
        maxLoop=5,
        LD=0.7,
        p_threshold=0.1,
        file_output=False,
    )

    r_gwas = r_bridge.float_array(r_bridge.component(r_result, "GWAS")).T
    r_qtns = (
        r_bridge.float_array(r_bridge.component(r_result, "seqQTN")).astype(int) - 1
    )
    r_effects = r_bridge.float_array(r_bridge.component(r_result, "Beta")).reshape(-1)
    assert not isinstance(py_result, dict)
    assert py_result.QTNs is not None
    assert py_result.GWAS is not None
    nt.assert_array_equal(np.asarray(py_result.GWAS["Chr"], dtype=float), r_gwas[:, 1])
    nt.assert_array_equal(py_result.GWAS["Pos"], r_gwas[:, 2])
    nt.assert_array_equal(np.sort(py_result.QTNs), np.sort(r_qtns))
    nt.assert_allclose(py_result.GWAS["P.value"], r_gwas[:, 3], rtol=1e-9, atol=1e-12)
    nt.assert_allclose(py_result.GWAS["effect"], r_effects, rtol=1e-9, atol=5e-7)
