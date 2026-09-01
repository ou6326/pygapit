"""Regression tests on GAPIT's bundled maize diversity-panel dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
from numpy.typing import NDArray

from pygapit.gapit import GAPIT, GAPITResult
from tests.cross_language.r_bridge import RBridge
from tests.cross_language.workflow import r_scalar

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class OfficialDataset:
    phenotype: pd.DataFrame
    genotype: pd.DataFrame
    marker_map: pd.DataFrame
    taxa: StringArray
    phenotype_values: FloatArray
    genotype_values: FloatArray
    marker_names: StringArray
    output_order: IndexArray


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
) -> FloatArray:
    r_pca = r_bridge.source_function(r_root, "GAPIT.PCA.R", "GAPIT.PCA")
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


def test_official_maize_glm_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
) -> None:
    """Compare the public GLM workflow on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    r_glm = r_bridge.source_function(r_root, "GAPIT.FarmCPU.R", "FarmCPU.LM")
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
) -> None:
    """Compare the public MLM workflow on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    design = np.column_stack([np.ones(len(inputs.taxa)), r_scores])
    covariates_with_taxa = np.column_stack(
        [np.arange(len(inputs.taxa), dtype=np.float64), r_scores]
    )
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, filename)
    r_kinship_function = r_bridge.source_function(
        r_root,
        "GAPIT.kinship.VanRaden.R",
        "GAPIT.kinship.VanRaden",
    )
    r_kinship = r_bridge.float_array(
        r_kinship_function(r_bridge.matrix(inputs.genotype_values))
    )
    r_mlm = r_bridge.source_function(r_root, "GAPIT.EMMAxP3D.R", "GAPIT.EMMAxP3D")
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


def test_official_maize_farmcpu_workflow_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
) -> None:
    """Compare the public FarmCPU workflow on GAPIT's bundled maize data."""
    inputs = _load_official_dataset(r_root)
    marker_count = inputs.genotype_values.shape[1]
    r_scores = _r_pca_scores(r_bridge, r_root, inputs, component_count=3)
    marker_index = pd.Index(inputs.marker_names, dtype="str")
    filtered_map = inputs.marker_map.set_index("SNP").loc[marker_index]
    chromosomes = filtered_map["Chromosome"].to_numpy(dtype=np.float64)
    positions = filtered_map["Position"].to_numpy(dtype=np.float64)
    phenotype = np.column_stack(
        [
            np.arange(len(inputs.phenotype_values), dtype=np.float64),
            inputs.phenotype_values,
        ]
    )
    genotype = np.column_stack(
        [
            np.arange(len(inputs.phenotype_values), dtype=np.float64),
            inputs.genotype_values,
        ]
    )
    marker_map = np.column_stack(
        [
            np.arange(1, marker_count + 1, dtype=np.float64),
            chromosomes,
            positions,
        ]
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
