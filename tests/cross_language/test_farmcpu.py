"""Cross-language FarmCPU alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
from numpy.typing import NDArray

from pygapit.gapit import GAPIT
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
    marker_map = np.column_stack([
        np.arange(1, marker_count + 1, dtype=float),
        chromosomes,
        positions,
    ])
    phenotype_with_taxa = np.column_stack([
        np.arange(len(fixed_phenotype), dtype=float),
        fixed_phenotype,
    ])

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


def test_farmcpu_iterative_workflow_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare the complete regular-matrix FarmCPU iteration and final scan."""
    marker_count = fixed_genotypes.shape[1]
    chromosomes = np.array([1, 1, 1, 2, 2, 2], dtype=float)
    positions = np.array([100, 200, 1500, 100, 200, 1500], dtype=float)
    phenotype = np.column_stack([
        np.arange(len(fixed_phenotype), dtype=float),
        fixed_phenotype,
    ])
    marker_map = np.column_stack([
        np.arange(1, marker_count + 1, dtype=float),
        chromosomes,
        positions,
    ])
    r_bridge.source(r_root, "GAPIT.Specify.R")
    r_bridge.source(r_root, "GAPIT.Power.R")
    r_bridge.source_for_regular_matrices(r_root, "GAPIT.FarmCPU.R")
    r_farmcpu = r_bridge.function("FarmCPU")
    r_marker_map = r_bridge.function("as.data.frame")(
        r_bridge.matrix(marker_map, column_names=["SNP", "Chr", "Pos"])
    )
    r_result = r_farmcpu(
        Y=r_bridge.matrix(phenotype, column_names=["Taxa", "Trait"]),
        GD=r_bridge.matrix(
            np.column_stack([
                np.arange(len(fixed_phenotype), dtype=float),
                fixed_genotypes,
            ])
        ),
        GM=r_marker_map,
        file_output=False,
        method_bin="static",
        bin_size=r_bridge.float_vector(np.array([1000.0, 1000.0, 1000.0])),
        bin_selection=r_bridge.float_vector(np.array([1.0])),
        maxLoop=5,
        p_threshold=0.1,
        QTN_threshold=0.1,
        maf_threshold=0.0,
        converge=1.0,
    )
    phenotype_frame, genotype_frame, marker_frame = fixed_gapit_inputs
    py_result = GAPIT(
        Y=phenotype_frame,
        GD=genotype_frame,
        GM=marker_frame,
        model="FARMCPU",
        PCA_total=0,
        maf_threshold=0.0,
        maxLoop=5,
        bin_size=1000,
        p_threshold=0.1,
        file_output=False,
    )

    r_gwas = r_bridge.float_array(r_bridge.component(r_result, "GWAS")).T
    r_qtns = (
        r_bridge.float_array(r_bridge.component(r_result, "seqQTN")).astype(int) - 1
    )
    assert not isinstance(py_result, dict)
    assert py_result.QTNs is not None
    assert py_result.GWAS is not None
    nt.assert_array_equal(np.asarray(py_result.GWAS["Chr"], dtype=float), r_gwas[:, 1])
    nt.assert_array_equal(py_result.GWAS["Pos"], r_gwas[:, 2])
    nt.assert_array_equal(np.sort(py_result.QTNs), np.sort(r_qtns))
    nt.assert_allclose(py_result.GWAS["P.value"], r_gwas[:, 3], rtol=1e-9, atol=1e-12)
    nt.assert_allclose(py_result.GWAS["effect"], r_gwas[:, 5], rtol=1e-9, atol=5e-7)
