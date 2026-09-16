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
    covariates_with_taxa = np.column_stack([
        np.arange(len(inputs.taxa), dtype=np.float64),
        design[:, 1:],
    ])
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


def test_top_level_glm_preprocessing_and_final_table_match_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Connect R imputation, MAF, PCA/K, scan, and final-table evidence."""
    phenotype, genotype, marker_map = (frame.copy() for frame in fixed_gapit_inputs)
    phenotype.loc[3, "Trait"] = np.nan
    genotype.iloc[0, 1] = np.nan
    genotype.iloc[:, -2] = 2.0
    genotype.iloc[:, -1] = 0.0
    genotype.iloc[0, -1] = 1.0

    raw_genotypes = genotype.iloc[:, 1:].to_numpy(dtype=np.float64)
    r_bridge.source(r_root, "GAPIT.Imputation.R")
    r_impute = r_bridge.function(
        "function(x) apply(x, 2, function(column) "
        "GAPIT.Imputation(column, impute='Middle', byRow=TRUE))"
    )
    imputed = r_bridge.float_array(r_impute(r_bridge.matrix(raw_genotypes)))

    valid_samples = phenotype["Trait"].notna().to_numpy()
    analysis_genotypes = imputed[valid_samples]
    analysis_phenotype = phenotype.loc[valid_samples, "Trait"].to_numpy(
        dtype=np.float64
    )
    allele_frequency = np.mean(analysis_genotypes, axis=0) / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    keep = maf >= 0.1
    filtered_genotypes = analysis_genotypes[:, keep]
    filtered_map = marker_map.loc[keep].reset_index(drop=True)
    retained_maf = maf[keep]

    r_pca = r_bridge.source_function(r_root, "GAPIT.PCA.R", "GAPIT.PCA")
    r_pca_result = r_pca(
        r_bridge.matrix(filtered_genotypes),
        r_bridge.float_vector(np.arange(len(analysis_phenotype), dtype=np.float64)),
        PC_number=2,
        file_output=False,
        PCA_total=2,
    )
    r_scores_with_taxa = r_bridge.float_array(r_bridge.component(r_pca_result, "PCs"))
    if r_scores_with_taxa.shape[0] != len(analysis_phenotype):
        r_scores_with_taxa = r_scores_with_taxa.T
    r_scores = r_scores_with_taxa[:, 1:3]
    design = np.column_stack([np.ones(len(analysis_phenotype)), r_scores])

    r_kinship = r_bridge.source_function(
        r_root,
        "GAPIT.kinship.VanRaden.R",
        "GAPIT.kinship.VanRaden",
    )
    expected_kinship = r_bridge.float_array(
        r_kinship(r_bridge.matrix(filtered_genotypes))
    )
    r_glm = r_bridge.source_function(r_root, "GAPIT.FarmCPU.R", "FarmCPU.LM")
    r_glm_result = r_glm(
        r_bridge.float_vector(analysis_phenotype),
        w=r_bridge.matrix(design[:, 1:]),
        GDP=r_bridge.matrix(filtered_genotypes),
        orientation="col",
        model="A",
        ncpus=1,
    )
    r_p_values = r_bridge.float_array(r_bridge.component(r_glm_result, "PF")).reshape(
        -1
    )
    r_effects = r_bridge.float_array(r_bridge.component(r_glm_result, "B")).reshape(-1)
    r_adjust = r_bridge.function("function(p) stats::p.adjust(p, method='BH')")
    r_adjusted = r_bridge.float_array(
        r_adjust(r_bridge.float_vector(r_p_values))
    ).reshape(-1)

    py_result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        trait="Trait",
        PCA_total=2,
        maf_threshold=0.1,
        SNP_impute="middle",
        file_output=False,
    )
    assert not isinstance(py_result, dict)
    assert py_result.GWAS is not None
    assert py_result.pca is not None
    assert py_result.kinship is not None

    nt.assert_array_equal(
        py_result.taxa,
        phenotype.loc[valid_samples, "Taxa"].to_numpy(dtype=np.str_),
    )
    nt.assert_allclose(py_result.kinship, expected_kinship, rtol=1e-12, atol=1e-12)
    for component in range(r_scores.shape[1]):
        py_scores = py_result.pca.scores[:, component]
        sign = np.sign(np.dot(py_scores, r_scores[:, component])) or 1.0
        nt.assert_allclose(
            py_scores, sign * r_scores[:, component], rtol=1e-12, atol=1e-12
        )

    chromosomes = filtered_map["Chromosome"].astype(str).to_numpy()
    positions = filtered_map["Position"].to_numpy(dtype=np.float64)
    output_order = np.lexsort((np.arange(len(filtered_map)), positions, chromosomes))
    gwas = py_result.GWAS
    nt.assert_array_equal(gwas["SNP"], filtered_map.loc[output_order, "SNP"])
    nt.assert_array_equal(gwas["Chr"].astype(str), chromosomes[output_order])
    nt.assert_array_equal(gwas["Pos"], positions[output_order])
    nt.assert_allclose(
        np.asarray(gwas["P.value"], dtype=np.float64),
        r_p_values[output_order],
        rtol=1e-10,
        atol=1e-12,
    )
    nt.assert_array_equal(
        np.asarray(gwas["effect"], dtype=np.float64),
        np.round(r_effects[output_order], 6),
    )
    nt.assert_array_equal(
        np.asarray(gwas["maf"], dtype=np.float64),
        np.round(retained_maf[output_order], 4),
    )
    nt.assert_array_equal(gwas["nobs"], len(analysis_phenotype))
    nt.assert_array_equal(
        np.asarray(gwas["FDR.Adjusted.P.values"], dtype=np.float64),
        np.round(r_adjusted[output_order], 6),
    )
