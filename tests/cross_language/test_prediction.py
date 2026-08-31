"""Cross-language genomic-prediction alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
from numpy.typing import NDArray

from pygapit.gapit import GAPIT
from pygapit.gs.blup import gblup
from pygapit.stats.kinship import vanraden_kinship
from tests.cross_language.r_bridge import RBridge


def test_gblup_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare GAPIT's null-model BLUE, BLUP, PEV, and variance components."""
    design = np.column_stack([np.ones(len(fixed_phenotype)), fixed_covariate])
    covariates_with_taxa = np.column_stack(
        [np.arange(len(fixed_phenotype), dtype=float), fixed_covariate]
    )
    kinship = vanraden_kinship(fixed_genotypes)

    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, filename)
    r_emmax_p3d = r_bridge.source_function(r_root, "GAPIT.EMMAxP3D.R", "GAPIT.EMMAxP3D")
    r_null = r_bridge.evaluate("NULL")
    r_result = r_emmax_p3d(
        ys=r_bridge.matrix(fixed_phenotype[np.newaxis, :]),
        xs=r_bridge.matrix(fixed_genotypes[:, :1]),
        K=r_bridge.matrix(kinship),
        X0=r_bridge.matrix(design),
        CVI=r_bridge.matrix(covariates_with_taxa),
        file_from=1,
        file_to=1,
        file_fragment=1,
        fullGD=True,
        SNP_P3D=True,
        Timmer=r_null,
        Memory=r_null,
        optOnly=True,
    )
    py_result = gblup(fixed_phenotype, design, kinship)

    r_blue_parts = r_bridge.float_array(r_bridge.component(r_result, "BLUE"))
    r_blue = np.sum(r_blue_parts, axis=1)
    r_blup = r_bridge.float_array(r_bridge.component(r_result, "BLUP")).reshape(-1)
    r_pev = r_bridge.float_array(r_bridge.component(r_result, "PEV")).reshape(-1)

    nt.assert_allclose(py_result.blue, r_blue, rtol=2e-6, atol=1e-10)
    nt.assert_allclose(py_result.blup, r_blup, rtol=2e-6, atol=1e-10)
    nt.assert_allclose(py_result.pev, r_pev, rtol=2e-6, atol=1e-10)
    nt.assert_allclose(py_result.prediction, r_blue + r_blup, rtol=2e-6, atol=1e-10)

    phenotype_frame, genotype_frame, marker_frame = fixed_gapit_inputs
    taxa = phenotype_frame["Taxa"].astype(str).to_numpy()
    kinship_frame = pd.DataFrame(kinship, columns=taxa)
    kinship_frame.insert(0, "Taxa", taxa)
    covariate_frame = pd.DataFrame({"Taxa": taxa, "Covariate": fixed_covariate})
    workflow_result = GAPIT(
        Y=phenotype_frame,
        GD=genotype_frame,
        GM=marker_frame,
        KI=kinship_frame,
        CV=covariate_frame,
        model="MLM",
        PCA_total=0,
        maf_threshold=0.0,
        buspred=True,
        prediction_model="gBLUP",
        file_output=False,
    )

    assert not isinstance(workflow_result, dict)
    assert workflow_result.Pred is not None
    nt.assert_array_equal(workflow_result.Pred["Taxa"], taxa)
    nt.assert_allclose(workflow_result.Pred["BLUE"], r_blue, atol=5e-5)
    nt.assert_allclose(workflow_result.Pred["BLUP"], r_blup, atol=5e-5)
    nt.assert_allclose(workflow_result.Pred["PEV"], r_pev, atol=5e-7)
    nt.assert_allclose(workflow_result.Pred["Prediction"], r_blue + r_blup, atol=5e-5)
