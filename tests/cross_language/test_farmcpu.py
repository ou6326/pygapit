"""Cross-language FarmCPU alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
from numpy.typing import NDArray

from pygapit.gwas.farmcpu import _bin_select_qtns
from tests.cross_language.r_bridge import RBridge


def test_farmcpu_bin_selection_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
) -> None:
    """Compare the static-bin pseudo-QTN selection used by FarmCPU."""
    marker_count = fixed_genotypes.shape[1]
    p_values = np.array([0.08, 0.001, 0.03, 0.02, 0.004, 0.01], dtype=float)
    chromosomes = np.array([1, 1, 1, 1, 2, 2], dtype=float)
    positions = np.array([100, 200, 1500, 1600, 100, 1600], dtype=float)
    marker_map = np.column_stack(
        [np.arange(1, marker_count + 1, dtype=float), chromosomes, positions]
    )
    phenotype_with_taxa = np.column_stack(
        [np.arange(len(fixed_phenotype), dtype=float), fixed_phenotype]
    )

    r_bridge.source(r_root, "GAPIT.Specify.R")
    r_farmcpu_bin = r_bridge.source_function(r_root, "GAPIT.FarmCPU.R", "FarmCPU.BIN")
    r_result = r_farmcpu_bin(
        Y=r_bridge.matrix(phenotype_with_taxa),
        GDP=r_bridge.matrix(fixed_genotypes),
        GM=r_bridge.matrix(marker_map),
        P=r_bridge.float_vector(p_values),
        orientation="col",
        method="static",
        b=1000,
        s=1,
        theLoop=1,
    )
    r_indices = (
        r_bridge.float_array(r_bridge.component(r_result, "seqQTN")).astype(int) - 1
    )
    max_qtns = round(
        np.sqrt(len(fixed_phenotype)) / np.sqrt(np.log10(len(fixed_phenotype)))
    )
    py_indices = _bin_select_qtns(
        p_values,
        chromosomes,
        positions,
        bin_size=1000,
        max_qtns=max_qtns,
        p_threshold=1.0,
    )

    nt.assert_array_equal(py_indices, r_indices)
