"""Top-level GLM and MLM workflow alignment with GAPIT 3.5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
from numpy.typing import NDArray

from pygapit.gapit import GAPIT, GAPITResult
from tests.cross_language.r_bridge import RBridge
from tests.cross_language.workflow import (
    WorkflowInputs,
    assert_top_level_preparation,
    make_workflow_inputs,
    r_design_with_pca,
    r_scalar,
)


def _run_python_workflow(inputs: WorkflowInputs, model: str) -> GAPITResult:
    result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        CV=inputs.covariate,
        KI=inputs.kinship,
        model=model,
        trait="Trait",
        PCA_total=2,
        maf_threshold=0.0,
        file_output=False,
    )
    assert not isinstance(result, dict)
    assert result.GWAS is not None
    assert result.pca is not None
    assert result.kinship is not None
    return result


def test_top_level_glm_with_pca_cv_ki_and_missing_phenotype_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare public GLM orchestration with GAPIT's additive GLM kernel."""
    inputs = make_workflow_inputs(
        fixed_genotypes,
        fixed_phenotype,
        fixed_covariate,
        fixed_gapit_inputs,
    )
    design, r_scores = r_design_with_pca(r_bridge, r_root, inputs, pca_total=2)
    r_glm = r_bridge.source_function(r_root, "GAPIT.FarmCPU.R", "FarmCPU.LM")
    r_result = r_glm(
        r_bridge.float_vector(inputs.phenotype_values),
        w=r_bridge.matrix(design[:, 1:]),
        GDP=r_bridge.matrix(inputs.genotype_values),
        orientation="col",
        model="A",
        ncpus=1,
    )
    py_result = _run_python_workflow(inputs, "GLM")

    assert_top_level_preparation(py_result, inputs, r_scores)
    py_gwas = py_result.GWAS
    assert py_gwas is not None
    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "PF"))
    r_effects = r_bridge.float_array(r_bridge.component(r_result, "B")).reshape(-1)
    nt.assert_allclose(
        np.asarray(py_gwas["P.value"], dtype=np.float64),
        r_p_values,
        rtol=1e-10,
        atol=1e-12,
    )
    nt.assert_array_equal(
        np.asarray(py_gwas["effect"], dtype=np.float64),
        np.round(r_effects, 6),
    )


def test_top_level_mlm_with_pca_cv_ki_and_missing_phenotype_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare public MLM orchestration with GAPIT's EMMAX/P3D workflow."""
    inputs = make_workflow_inputs(
        fixed_genotypes,
        fixed_phenotype,
        fixed_covariate,
        fixed_gapit_inputs,
    )
    design, r_scores = r_design_with_pca(r_bridge, r_root, inputs, pca_total=2)
    covariates_with_taxa = np.column_stack(
        [np.arange(len(inputs.taxa), dtype=np.float64), design[:, 1:]]
    )
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, filename)
    r_mlm = r_bridge.source_function(r_root, "GAPIT.EMMAxP3D.R", "GAPIT.EMMAxP3D")
    r_null = r_bridge.evaluate("NULL")
    r_result = r_mlm(
        ys=r_bridge.matrix(inputs.phenotype_values[np.newaxis, :]),
        xs=r_bridge.matrix(inputs.genotype_values),
        K=r_bridge.matrix(inputs.kinship_values),
        X0=r_bridge.matrix(design),
        CVI=r_bridge.matrix(covariates_with_taxa),
        file_from=1,
        file_to=1,
        file_fragment=inputs.genotype_values.shape[1],
        fullGD=True,
        SNP_P3D=True,
        Timmer=r_null,
        Memory=r_null,
        optOnly=False,
    )
    py_result = _run_python_workflow(inputs, "MLM")

    assert_top_level_preparation(py_result, inputs, r_scores)
    py_gwas = py_result.GWAS
    assert py_gwas is not None
    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "ps")).reshape(-1)
    r_effects = r_bridge.float_array(
        r_bridge.component(r_result, "effect.est")
    ).reshape(-1)
    r_standard_errors = r_bridge.float_array(
        r_bridge.component(r_result, "stderr")
    ).reshape(-1)
    r_vg = r_scalar(r_bridge, r_result, "vgs")
    r_ve = r_scalar(r_bridge, r_result, "ves")

    nt.assert_allclose(
        np.asarray(py_gwas["P.value"], dtype=np.float64),
        r_p_values,
        rtol=2e-6,
        atol=1e-12,
    )
    nt.assert_array_equal(
        np.asarray(py_gwas["effect"], dtype=np.float64),
        np.round(r_effects, 6),
    )
    nt.assert_array_equal(
        np.asarray(py_gwas["se"], dtype=np.float64),
        np.round(r_standard_errors, 6),
    )
    nt.assert_allclose(py_result.vg, r_vg, rtol=2e-6, atol=1e-12)
    nt.assert_allclose(py_result.ve, r_ve, rtol=2e-6, atol=1e-12)
    nt.assert_allclose(py_result.h2, r_vg / (r_vg + r_ve), rtol=2e-6, atol=1e-12)
