"""Regression tests on GAPIT's bundled maize diversity-panel dataset."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd

from pygapit._typing import FloatMatrix, FloatVector, StrVector
from pygapit.gapit import GAPIT, GAPITResult
from pygapit.gs.blup import gblup
from tests.cross_language.r_bridge import RBridge, RList, RMatrix
from tests.cross_language.workflow import r_scalar

IndexVector: t.TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]


@dataclass(frozen=True, slots=True)
class OfficialDataset:
    phenotype: pd.DataFrame
    genotype: pd.DataFrame
    marker_map: pd.DataFrame
    taxa: StrVector
    phenotype_values: FloatVector
    genotype_values: FloatMatrix
    marker_names: StrVector
    output_order: IndexVector


def _load_official_dataset(r_root: Path) -> OfficialDataset:
    data_root = r_root.parent / "inst" / "extdata"
    phenotype = pd.read_csv(data_root / "mdp_traits.txt.gz", sep="\t")
    genotype = pd.read_csv(data_root / "mdp_numeric.txt.gz", sep="\t")
    marker_map = pd.read_csv(
        data_root / "mdp_SNP_information.txt.gz",
        sep="\t",
    )

    genotype_by_taxon = genotype.set_index(genotype.columns[0])
    phenotype_taxa = phenotype["Taxa"].astype(str)
    trait_values = pd.to_numeric(phenotype["EarHT"], errors="coerce")
    valid = (
        phenotype_taxa.isin(genotype_by_taxon.index.astype(str)) & trait_values.notna()
    )
    taxa_index = pd.Index(phenotype_taxa[valid], dtype="str")
    taxa = taxa_index.to_numpy(dtype=np.str_)
    values = genotype_by_taxon.loc[taxa_index].to_numpy(dtype=np.float64)
    phenotype_values = trait_values[valid].to_numpy(dtype=np.float64)

    allele_frequency = np.mean(values, axis=0) / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    keep = maf >= 0.05
    filtered_map = marker_map.loc[keep].reset_index(drop=True).copy()
    filtered_map["_chromosome"] = filtered_map["Chromosome"].astype(str)
    output_order = filtered_map.sort_values(["_chromosome", "Position"]).index.to_numpy(
        dtype=np.intp
    )
    return OfficialDataset(
        phenotype=phenotype,
        genotype=genotype,
        marker_map=marker_map,
        taxa=taxa,
        phenotype_values=phenotype_values,
        genotype_values=values[:, keep],
        marker_names=np.asarray(genotype.columns[1:], dtype=np.str_)[keep],
        output_order=output_order,
    )


def _r_pca_scores(
    r_bridge: RBridge,
    r_root: Path,
    inputs: OfficialDataset,
    component_count: int,
) -> FloatMatrix:
    r_pca = r_bridge.source_function(
        r_root,
        "GAPIT.PCA.R",
        "GAPIT.PCA",
        returns=RList,
    )
    result = r_pca(
        r_bridge.matrix(inputs.genotype_values),
        r_bridge.float_vector(np.arange(len(inputs.taxa), dtype=np.float64)),
        PC_number=component_count,
        file_output=False,
        PCA_total=component_count,
    )
    scores_with_taxa = r_bridge.float_array(r_bridge.component(result, "PCs"))
    if scores_with_taxa.shape[0] != len(inputs.taxa):
        scores_with_taxa = scores_with_taxa.T
    return scores_with_taxa[:, 1 : component_count + 1]


def _r_vanraden_kinship(
    r_bridge: RBridge,
    r_root: Path,
    genotype_values: FloatMatrix,
) -> FloatMatrix:
    r_kinship_function = r_bridge.source_function(
        r_root,
        "GAPIT.kinship.VanRaden.R",
        "GAPIT.kinship.VanRaden",
        returns=RMatrix,
    )
    return r_bridge.float_array(r_kinship_function(r_bridge.matrix(genotype_values)))


def _iterative_workflow_matrices(
    r_bridge: RBridge,
    r_root: Path,
    inputs: OfficialDataset,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix, FloatMatrix]:
    """Build the shared R inputs for official FarmCPU and BLINK workflows."""
    marker_count = inputs.genotype_values.shape[1]
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    marker_index = pd.Index(inputs.marker_names, dtype="str")
    filtered_map = inputs.marker_map.set_index("SNP").loc[marker_index]
    phenotype = np.column_stack([
        np.arange(len(inputs.phenotype_values), dtype=np.float64),
        inputs.phenotype_values,
    ])
    genotype = np.column_stack([
        np.arange(len(inputs.phenotype_values), dtype=np.float64),
        inputs.genotype_values,
    ])
    marker_map = np.column_stack([
        np.arange(1, marker_count + 1, dtype=np.float64),
        filtered_map["Chromosome"].to_numpy(dtype=np.float64),
        filtered_map["Position"].to_numpy(dtype=np.float64),
    ])
    return r_scores, phenotype, genotype, marker_map


def _run_official_r_blink(
    r_bridge: RBridge,
    r_root: Path,
    r_scores: FloatMatrix,
    phenotype: FloatMatrix,
    genotype: FloatMatrix,
    marker_map: FloatMatrix,
    candidate_threshold: float,
    *,
    include_covariates_in_bic: bool,
) -> RList:
    """Run upstream or corrected-CV GAPIT BLINK on regular matrices."""
    replacements = None
    if include_covariates_in_bic:
        replacements = {
            "Y = Y1,\n                                         orientation = orientation,": (
                "Y = Y1,\n                                         CV = CV1,\n"
                "                                         orientation = orientation,"
            ),
            "Y=Y1,orientation=orientation,BIC.method=BIC.method": (
                "Y=Y1,CV=CV1,orientation=orientation,BIC.method=BIC.method"
            ),
        }
    r_bridge.source(r_root, "GAPIT.Specify.R")
    r_bridge.source_for_regular_matrices(r_root, "GAPIT.FarmCPU.R")
    r_bridge.source_for_regular_matrices(
        r_root,
        "GAPIT.Blink.R",
        replacements=replacements,
    )
    r_blink = r_bridge.function("Blink")
    r_marker_map = r_bridge.function("as.data.frame")(
        r_bridge.matrix(marker_map, column_names=["SNP", "Chr", "Pos"])
    )
    return r_blink(
        Y=r_bridge.matrix(phenotype, column_names=["Taxa", "EarHT"]),
        GD=r_bridge.matrix(genotype),
        GM=r_marker_map,
        CV=r_bridge.matrix(r_scores),
        file_output=False,
        maxLoop=5,
        LD=0.7,
        p_threshold=candidate_threshold,
        maf_threshold=0.0,
        converge=1.0,
    )


def test_official_maize_glm_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
):
    """Compare the public GLM workflow on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    r_glm = r_bridge.source_function(
        r_root,
        "GAPIT.FarmCPU.R",
        "FarmCPU.LM",
        returns=RList,
    )
    r_result = r_glm(
        r_bridge.float_vector(inputs.phenotype_values),
        w=r_bridge.matrix(r_scores),
        GDP=r_bridge.matrix(inputs.genotype_values),
        orientation="col",
        model="A",
        ncpus=1,
    )
    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        model="GLM",
        trait="EarHT",
        PCA_total=3,
        maf_threshold=0.05,
        file_output=False,
    )

    assert isinstance(py_result, GAPITResult)
    assert py_result.GWAS is not None
    assert py_result.pca is not None
    nt.assert_array_equal(py_result.taxa, inputs.taxa)
    nt.assert_array_equal(
        py_result.GWAS["SNP"],
        inputs.marker_names[inputs.output_order],
    )
    for component in range(r_scores.shape[1]):
        py_scores = py_result.pca.scores[:, component]
        sign = np.sign(np.dot(py_scores, r_scores[:, component])) or 1.0
        nt.assert_allclose(
            py_scores,
            sign * r_scores[:, component],
            rtol=1e-10,
            atol=1e-10,
        )

    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "PF"))
    r_effects = r_bridge.float_array(r_bridge.component(r_result, "B")).reshape(-1)
    nt.assert_allclose(
        np.asarray(py_result.GWAS["P.value"], dtype=np.float64),
        r_p_values[inputs.output_order],
        rtol=1e-9,
        atol=1e-12,
    )
    nt.assert_array_equal(
        np.asarray(py_result.GWAS["effect"], dtype=np.float64),
        np.round(r_effects[inputs.output_order], 6),
    )


def test_official_maize_mlm_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
):
    """Compare the public MLM workflow on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    design = np.column_stack([np.ones(len(inputs.taxa)), r_scores])
    covariates_with_taxa = np.column_stack([
        np.arange(len(inputs.taxa), dtype=np.float64),
        r_scores,
    ])
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, filename)
    r_kinship = _r_vanraden_kinship(r_bridge, r_root, inputs.genotype_values)
    r_mlm = r_bridge.source_function(
        r_root,
        "GAPIT.EMMAxP3D.R",
        "GAPIT.EMMAxP3D",
        returns=RList,
    )
    r_null = r_bridge.evaluate("NULL")
    r_result = r_mlm(
        ys=r_bridge.matrix(inputs.phenotype_values[np.newaxis, :]),
        xs=r_bridge.matrix(inputs.genotype_values),
        K=r_bridge.matrix(r_kinship),
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
    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        model="MLM",
        trait="EarHT",
        PCA_total=3,
        maf_threshold=0.05,
        file_output=False,
    )

    assert isinstance(py_result, GAPITResult)
    assert py_result.GWAS is not None
    assert py_result.kinship is not None
    nt.assert_allclose(py_result.kinship, r_kinship, rtol=1e-12, atol=1e-12)
    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "ps")).reshape(-1)
    r_effects = r_bridge.float_array(
        r_bridge.component(r_result, "effect.est")
    ).reshape(-1)
    r_standard_errors = r_bridge.float_array(
        r_bridge.component(r_result, "stderr")
    ).reshape(-1)
    nt.assert_allclose(
        np.asarray(py_result.GWAS["P.value"], dtype=np.float64),
        r_p_values[inputs.output_order],
        rtol=5e-5,
        atol=2e-8,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["effect"], dtype=np.float64),
        r_effects[inputs.output_order],
        rtol=5e-5,
        atol=4e-5,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["se"], dtype=np.float64),
        r_standard_errors[inputs.output_order],
        rtol=5e-5,
        atol=4e-5,
    )
    r_vg = r_scalar(r_bridge, r_result, "vgs")
    r_ve = r_scalar(r_bridge, r_result, "ves")
    nt.assert_allclose(py_result.vg, r_vg, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_result.ve, r_ve, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_result.h2, r_vg / (r_vg + r_ve), rtol=2e-5, atol=1e-8)


def test_official_maize_gblup_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
):
    """Compare official-data BLUE, BLUP, PEV, and prediction with GAPIT."""
    inputs = _load_official_dataset(r_root)
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    design = np.column_stack([np.ones(len(inputs.taxa)), r_scores])
    covariates_with_taxa = np.column_stack([
        np.arange(len(inputs.taxa), dtype=np.float64),
        r_scores,
    ])
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, filename)
    r_kinship = _r_vanraden_kinship(r_bridge, r_root, inputs.genotype_values)
    r_gblup = r_bridge.source_function(
        r_root,
        "GAPIT.EMMAxP3D.R",
        "GAPIT.EMMAxP3D",
        returns=RList,
    )
    r_null = r_bridge.evaluate("NULL")
    r_result = r_gblup(
        ys=r_bridge.matrix(inputs.phenotype_values[np.newaxis, :]),
        xs=r_bridge.matrix(inputs.genotype_values[:, :1]),
        K=r_bridge.matrix(r_kinship),
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
    py_direct = gblup(
        inputs.phenotype_values,
        design,
        r_kinship,
        taxa=inputs.taxa,
    )

    r_blue = np.sum(r_bridge.float_array(r_bridge.component(r_result, "BLUE")), axis=1)
    r_blup = r_bridge.float_array(r_bridge.component(r_result, "BLUP")).reshape(-1)
    r_pev = r_bridge.float_array(r_bridge.component(r_result, "PEV")).reshape(-1)
    r_vg = r_scalar(r_bridge, r_result, "vgs")
    r_ve = r_scalar(r_bridge, r_result, "ves")

    nt.assert_array_equal(py_direct.taxa, inputs.taxa)
    nt.assert_allclose(py_direct.blue, r_blue, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_direct.blup, r_blup, rtol=2e-5, atol=2e-4)
    nt.assert_allclose(py_direct.pev, r_pev, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_direct.prediction, r_blue + r_blup, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_direct.vg, r_vg, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_direct.ve, r_ve, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_direct.h2, r_vg / (r_vg + r_ve), rtol=2e-5, atol=1e-8)

    workflow_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        model="gBLUP",
        trait="EarHT",
        PCA_total=3,
        maf_threshold=0.05,
        file_output=False,
    )

    assert isinstance(workflow_result, GAPITResult)
    assert workflow_result.Pred is not None
    assert workflow_result.kinship is not None
    nt.assert_array_equal(workflow_result.Pred["Taxa"], inputs.taxa)
    nt.assert_allclose(workflow_result.kinship, r_kinship, rtol=1e-12, atol=1e-12)
    nt.assert_array_equal(workflow_result.Pred["BLUE"], np.round(py_direct.blue, 4))
    nt.assert_array_equal(workflow_result.Pred["BLUP"], np.round(py_direct.blup, 4))
    nt.assert_array_equal(workflow_result.Pred["PEV"], np.round(py_direct.pev, 6))
    nt.assert_array_equal(
        workflow_result.Pred["Prediction"], np.round(py_direct.prediction, 4)
    )


def test_official_maize_cmlm_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
):
    """Compare fixed-compression CMLM on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    group_count = 40
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    design = np.column_stack([np.ones(len(inputs.taxa)), r_scores])
    covariates_with_taxa = np.column_stack([
        np.arange(len(inputs.taxa), dtype=np.float64),
        r_scores,
    ])
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
        "GAPIT.Compress.R",
    ):
        r_bridge.source(r_root, filename)
    r_kinship = _r_vanraden_kinship(r_bridge, r_root, inputs.genotype_values)
    r_compress = r_bridge.function(
        "function(KI, groups) {"
        " out <- GAPIT.Compress(KI, GN=groups, Timmer=NULL, Memory=NULL);"
        " list(labels=as.numeric(out$GA[,2]), kinship=out$KG)"
        "}",
        returns=RList,
    )
    kinship_with_taxa = np.column_stack([
        np.arange(len(inputs.taxa), dtype=np.float64),
        r_kinship,
    ])
    r_compression = r_compress(
        r_bridge.matrix(kinship_with_taxa),
        groups=group_count,
    )
    r_labels = r_bridge.float_array(r_bridge.component(r_compression, "labels")).astype(
        int
    )
    r_group_kinship = r_bridge.float_array(r_bridge.component(r_compression, "kinship"))
    r_incidence = np.zeros((len(inputs.taxa), group_count), dtype=np.float64)
    r_incidence[np.arange(len(inputs.taxa)), r_labels - 1] = 1.0

    r_mlm = r_bridge.source_function(
        r_root,
        "GAPIT.EMMAxP3D.R",
        "GAPIT.EMMAxP3D",
        returns=RList,
    )
    r_null = r_bridge.evaluate("NULL")
    r_result = r_mlm(
        ys=r_bridge.matrix(inputs.phenotype_values[np.newaxis, :]),
        xs=r_bridge.matrix(inputs.genotype_values),
        K=r_bridge.matrix(r_group_kinship),
        Z=r_bridge.matrix(r_incidence),
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
    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        model="CMLM",
        trait="EarHT",
        PCA_total=3,
        maf_threshold=0.05,
        group_from=group_count,
        group_to=group_count,
        file_output=False,
    )

    assert isinstance(py_result, GAPITResult)
    assert py_result.GWAS is not None
    assert py_result.kinship is not None
    assert py_result.model == "CMLM"
    nt.assert_array_equal(
        py_result.GWAS["SNP"],
        inputs.marker_names[inputs.output_order],
    )
    nt.assert_allclose(py_result.kinship, r_kinship, rtol=1e-12, atol=1e-12)
    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "ps")).reshape(-1)
    r_effects = r_bridge.float_array(
        r_bridge.component(r_result, "effect.est")
    ).reshape(-1)
    r_standard_errors = r_bridge.float_array(
        r_bridge.component(r_result, "stderr")
    ).reshape(-1)
    nt.assert_allclose(
        np.asarray(py_result.GWAS["P.value"], dtype=np.float64),
        r_p_values[inputs.output_order],
        rtol=5e-5,
        atol=2e-8,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["effect"], dtype=np.float64),
        r_effects[inputs.output_order],
        rtol=5e-5,
        atol=4e-5,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["se"], dtype=np.float64),
        r_standard_errors[inputs.output_order],
        rtol=5e-5,
        atol=4e-5,
    )
    r_vg = r_scalar(r_bridge, r_result, "vgs")
    r_ve = r_scalar(r_bridge, r_result, "ves")
    nt.assert_allclose(py_result.vg, r_vg, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_result.ve, r_ve, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(py_result.h2, r_vg / (r_vg + r_ve), rtol=2e-5, atol=1e-8)


def test_official_maize_mlmm_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
):
    """Compare covariate-free MLMM on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    r_bridge.source(r_root, "GAPIT.emma.R")
    r_bridge.source(r_root, "GAPIT.mlmm.R")
    r_mlmm = r_bridge.function(
        "function(y, X, K, nbchunks, maxsteps) {"
        " fit <- mlmm(y, X, K, nbchunks=nbchunks, maxsteps=maxsteps);"
        " list(opt_extBIC=fit$opt_extBIC,"
        "      h2=fit$step_table$h2[which.min(fit$step_table$extBIC)],"
        "      seq_all_na=as.numeric(all(is.na(fit$seqQTN))),"
        "      cof_all_na=as.numeric(all(is.na(fit$opt_extBIC$cof))))"
        "}",
        returns=RList,
    )
    r_kinship = _r_vanraden_kinship(r_bridge, r_root, inputs.genotype_values)
    marker_names = inputs.marker_names.tolist()
    r_result = r_mlmm(
        r_bridge.float_vector(inputs.phenotype_values),
        r_bridge.matrix(inputs.genotype_values, column_names=marker_names),
        r_bridge.matrix(r_kinship),
        nbchunks=10,
        maxsteps=10,
    )
    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        model="MLMM",
        trait="EarHT",
        PCA_total=0,
        maf_threshold=0.05,
        file_output=False,
    )

    assert isinstance(py_result, GAPITResult)
    assert py_result.GWAS is not None
    assert py_result.QTNs is not None
    assert py_result.kinship is not None
    assert py_result.model == "MLMM"
    nt.assert_array_equal(
        py_result.GWAS["SNP"],
        inputs.marker_names[inputs.output_order],
    )
    nt.assert_allclose(py_result.kinship, r_kinship, rtol=1e-12, atol=1e-12)

    r_optimum = r_bridge.component(r_result, "opt_extBIC")
    r_markers = np.asarray(
        r_bridge.component(r_bridge.component(r_optimum, "out"), "SNP"),
        dtype=np.str_,
    )
    missing_markers = iter(name for name in marker_names if name not in r_markers)
    r_markers = np.asarray(
        [name or next(missing_markers) for name in r_markers],
        dtype=np.str_,
    )
    canonical_order = np.asarray(
        [int(np.flatnonzero(r_markers == name)[0]) for name in marker_names],
        dtype=np.intp,
    )
    r_p_values = r_bridge.float_array(
        r_bridge.component(r_bridge.component(r_optimum, "out"), "pval")
    )[canonical_order]
    r_effects = r_bridge.float_array(
        r_bridge.component(r_bridge.component(r_optimum, "out"), "effect")
    )[canonical_order]
    r_h2 = r_bridge.float_array(r_bridge.component(r_result, "h2"))[0]
    r_sequence_all_na = r_bridge.float_array(
        r_bridge.component(r_result, "seq_all_na")
    )[0]
    r_cofactor_all_na = r_bridge.float_array(
        r_bridge.component(r_result, "cof_all_na")
    )[0]

    assert r_sequence_all_na == 1.0
    assert r_cofactor_all_na == 1.0
    assert len(py_result.QTNs) == 0
    nt.assert_allclose(py_result.h2, r_h2, rtol=2e-5, atol=1e-8)
    nt.assert_allclose(
        np.asarray(py_result.GWAS["P.value"], dtype=np.float64),
        r_p_values[inputs.output_order],
        rtol=5e-5,
        atol=1e-12,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["effect"], dtype=np.float64),
        np.round(r_effects[inputs.output_order], 6),
        rtol=5e-5,
        atol=4e-5,
    )


def test_official_maize_farmcpu_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
):
    """Compare the public FarmCPU workflow on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    marker_count = inputs.genotype_values.shape[1]
    r_scores, phenotype, genotype, marker_map = _iterative_workflow_matrices(
        r_bridge,
        r_root,
        inputs,
    )

    r_bridge.source(r_root, "GAPIT.Specify.R")
    r_bridge.source(r_root, "GAPIT.Power.R")
    r_bridge.source_for_regular_matrices(r_root, "GAPIT.FarmCPU.R")
    r_farmcpu = r_bridge.function("FarmCPU")
    r_marker_map = r_bridge.function("as.data.frame")(
        r_bridge.matrix(marker_map, column_names=["SNP", "Chr", "Pos"])
    )
    candidate_threshold = 1.0 / marker_count
    r_result = r_farmcpu(
        Y=r_bridge.matrix(phenotype, column_names=["Taxa", "EarHT"]),
        GD=r_bridge.matrix(genotype),
        GM=r_marker_map,
        CV=r_bridge.matrix(r_scores),
        file_output=False,
        method_bin="static",
        bin_size=r_bridge.float_vector(np.full(3, 5_000_000.0, dtype=np.float64)),
        bin_selection=r_bridge.float_vector(np.array([1.0], dtype=np.float64)),
        maxLoop=5,
        p_threshold=candidate_threshold,
        QTN_threshold=candidate_threshold,
        maf_threshold=0.0,
        converge=1.0,
        ncpus=1,
    )
    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        model="FarmCPU",
        trait="EarHT",
        PCA_total=3,
        maf_threshold=0.05,
        maxLoop=5,
        bin_size=5_000_000,
        p_threshold=candidate_threshold,
        file_output=False,
    )

    r_gwas = r_bridge.float_array(r_bridge.component(r_result, "GWAS")).T
    r_gwas = r_gwas[np.argsort(r_gwas[:, 0])][inputs.output_order]
    r_qtns = (
        r_bridge.float_array(r_bridge.component(r_result, "seqQTN")).astype(int) - 1
    )
    assert isinstance(py_result, GAPITResult)
    assert py_result.GWAS is not None
    assert py_result.QTNs is not None
    nt.assert_array_equal(np.sort(py_result.QTNs), np.sort(r_qtns))
    nt.assert_array_equal(
        np.asarray(py_result.GWAS["Chr"], dtype=np.float64), r_gwas[:, 1]
    )
    nt.assert_array_equal(
        np.asarray(py_result.GWAS["Pos"], dtype=np.float64), r_gwas[:, 2]
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["P.value"], dtype=np.float64),
        r_gwas[:, 3],
        rtol=1e-8,
        atol=1e-12,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["effect"], dtype=np.float64),
        r_gwas[:, 5],
        rtol=1e-8,
        atol=5e-7,
    )


def test_official_maize_blink_workflow_matches_covariate_corrected_gapit(
    r_bridge: RBridge,
    r_root: Path,
):
    """Compare BLINK after forwarding its supplied PCA covariates into BIC."""
    inputs = _load_official_dataset(r_root)
    marker_count = inputs.genotype_values.shape[1]
    r_scores, phenotype, genotype, marker_map = _iterative_workflow_matrices(
        r_bridge,
        r_root,
        inputs,
    )

    candidate_threshold = 1.0 / marker_count
    upstream_r_result = _run_official_r_blink(
        r_bridge,
        r_root,
        r_scores,
        phenotype,
        genotype,
        marker_map,
        candidate_threshold,
        include_covariates_in_bic=False,
    )
    corrected_r_result = _run_official_r_blink(
        r_bridge,
        r_root,
        r_scores,
        phenotype,
        genotype,
        marker_map,
        candidate_threshold,
        include_covariates_in_bic=True,
    )
    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        model="BLINK",
        trait="EarHT",
        PCA_total=3,
        maf_threshold=0.05,
        maxLoop=5,
        LD=0.7,
        p_threshold=candidate_threshold,
        file_output=False,
    )

    r_gwas = r_bridge.float_array(r_bridge.component(corrected_r_result, "GWAS")).T
    r_gwas = r_gwas[np.argsort(r_gwas[:, 0])][inputs.output_order]
    upstream_qtn_component = r_bridge.component(upstream_r_result, "seqQTN")
    upstream_qtns = (
        np.array([], dtype=int)
        if r_bridge.is_null(upstream_qtn_component)
        else r_bridge.float_array(upstream_qtn_component).astype(int) - 1
    )
    r_qtn_component = r_bridge.component(corrected_r_result, "seqQTN")
    r_qtns = (
        np.array([], dtype=int)
        if r_bridge.is_null(r_qtn_component)
        else r_bridge.float_array(r_qtn_component).astype(int) - 1
    )
    r_effects = r_bridge.float_array(
        r_bridge.component(corrected_r_result, "Beta")
    ).reshape(-1)
    assert isinstance(py_result, GAPITResult)
    assert py_result.GWAS is not None
    assert py_result.QTNs is not None
    nt.assert_array_equal(
        np.sort(upstream_qtns),
        np.array(
            [
                300,
                402,
                410,
                412,
                531,
                661,
                871,
                872,
                883,
                1301,
                1946,
                2216,
                2267,
                2283,
                2527,
                2529,
                2606,
            ],
            dtype=int,
        ),
    )
    nt.assert_array_equal(
        np.sort(r_qtns),
        np.array(
            [300, 402, 531, 661, 871, 1301, 1946, 2216, 2283, 2529, 2606],
            dtype=int,
        ),
    )
    nt.assert_array_equal(np.sort(py_result.QTNs), np.sort(r_qtns))
    nt.assert_array_equal(
        np.asarray(py_result.GWAS["Chr"], dtype=np.float64), r_gwas[:, 1]
    )
    nt.assert_array_equal(
        np.asarray(py_result.GWAS["Pos"], dtype=np.float64), r_gwas[:, 2]
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["P.value"], dtype=np.float64),
        r_gwas[:, 3],
        rtol=1e-8,
        atol=1e-12,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["effect"], dtype=np.float64),
        r_effects[inputs.output_order],
        rtol=1e-8,
        atol=5e-7,
    )
