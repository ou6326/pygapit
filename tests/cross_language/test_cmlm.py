"""Top-level CMLM workflow alignment with GAPIT 3.5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
from numpy.typing import NDArray

from pygapit.gapit import GAPIT
from pygapit.gwas.mlm import cmlm_gwas, compress_kinship
from tests.cross_language.r_bridge import RBridge
from tests.cross_language.workflow import (
    assert_top_level_preparation,
    make_workflow_inputs,
    r_design_with_pca,
    r_scalar,
)


def test_top_level_cmlm_with_fixed_compression_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare a fixed five-group CMLM through the public Python pipeline."""
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
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
    ):
        r_bridge.source(r_root, filename)
    r_bridge.source(r_root, "GAPIT.Compress.R")
    r_compress_fixed = r_bridge.function(
        "function(KI, groups) {"
        " out <- GAPIT.Compress(KI, GN=groups, Timmer=NULL, Memory=NULL);"
        " list(labels=as.numeric(out$GA[,2]), kinship=out$KG)"
        "}"
    )
    kinship_with_taxa = np.column_stack([
        np.arange(len(inputs.taxa), dtype=np.float64),
        inputs.kinship_values,
    ])
    r_compression = r_compress_fixed(
        r_bridge.matrix(kinship_with_taxa),
        groups=5,
    )
    r_labels = r_bridge.float_array(r_bridge.component(r_compression, "labels")).astype(
        int
    )
    r_group_kinship = r_bridge.float_array(r_bridge.component(r_compression, "kinship"))
    r_incidence = np.zeros((len(inputs.taxa), 5), dtype=np.float64)
    r_incidence[np.arange(len(inputs.taxa)), r_labels - 1] = 1.0
    py_group_kinship, py_incidence = compress_kinship(inputs.kinship_values, 5)
    py_labels = np.argmax(py_incidence, axis=1) + 1
    nt.assert_array_equal(
        py_labels[:, None] == py_labels[None, :],
        r_labels[:, None] == r_labels[None, :],
    )
    r_group_order = [
        r_labels[np.argmax(py_labels == label)] - 1 for label in range(1, 6)
    ]
    nt.assert_allclose(
        py_group_kinship,
        r_group_kinship[np.ix_(r_group_order, r_group_order)],
        rtol=1e-12,
        atol=1e-12,
    )

    r_mlm = r_bridge.source_function(r_root, "GAPIT.EMMAxP3D.R", "GAPIT.EMMAxP3D")
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
        CV=inputs.covariate,
        KI=inputs.kinship,
        model="CMLM",
        trait="Trait",
        PCA_total=2,
        maf_threshold=0.0,
        group_from=5,
        group_to=5,
        file_output=False,
    )

    assert not isinstance(py_result, dict)
    assert py_result.GWAS is not None
    assert py_result.model == "CMLM"
    assert_top_level_preparation(py_result, inputs, r_scores)
    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "ps")).reshape(-1)
    r_effects = r_bridge.float_array(
        r_bridge.component(r_result, "effect.est")
    ).reshape(-1)
    r_standard_errors = r_bridge.float_array(
        r_bridge.component(r_result, "stderr")
    ).reshape(-1)
    r_vg = r_scalar(r_bridge, r_result, "vgs")
    r_ve = r_scalar(r_bridge, r_result, "ves")

    nt.assert_allclose(py_result.GWAS["P.value"], r_p_values, rtol=2e-6, atol=1e-12)
    nt.assert_array_equal(py_result.GWAS["effect"], np.round(r_effects, 6))
    nt.assert_array_equal(py_result.GWAS["se"], np.round(r_standard_errors, 6))
    nt.assert_allclose(py_result.vg, r_vg, rtol=2e-6, atol=1e-12)
    nt.assert_allclose(py_result.ve, r_ve, rtol=2e-6, atol=1e-12)
    nt.assert_allclose(py_result.h2, r_vg / (r_vg + r_ve), rtol=2e-6, atol=1e-12)


def test_cmlm_selects_same_compression_as_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare REML selection across several valid compression levels."""
    inputs = make_workflow_inputs(
        fixed_genotypes,
        fixed_phenotype,
        fixed_covariate,
        fixed_gapit_inputs,
    )
    design, _ = r_design_with_pca(r_bridge, r_root, inputs, pca_total=2)
    for filename in (
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Compress.R",
    ):
        r_bridge.source(r_root, filename)

    r_select = r_bridge.function(
        "function(KI, y, X, groups) {"
        " best <- NULL;"
        " for (g in groups) {"
        "   compressed <- GAPIT.Compress(KI, GN=g, Timmer=NULL, Memory=NULL);"
        "   labels <- as.numeric(compressed$GA[,2]);"
        "   Z <- matrix(0, nrow=length(labels), ncol=g);"
        "   Z[cbind(seq_along(labels), labels)] <- 1;"
        "   fit <- GAPIT.emma.REMLE(y, X, compressed$KG, Z=Z);"
        "   if (is.null(best) || fit$REML > best$reml)"
        "     best <- list(group=g, reml=fit$REML);"
        " };"
        " best"
        "}"
    )
    kinship_with_taxa = np.column_stack([
        np.arange(len(inputs.taxa), dtype=np.float64),
        inputs.kinship_values,
    ])
    r_best = r_select(
        r_bridge.matrix(kinship_with_taxa),
        r_bridge.float_vector(inputs.phenotype_values),
        r_bridge.matrix(design),
        r_bridge.float_vector(np.arange(5, 8, dtype=np.float64)),
    )
    expected_group = int(r_scalar(r_bridge, r_best, "group"))

    selected = cmlm_gwas(
        inputs.phenotype_values,
        design,
        inputs.genotype_values,
        inputs.kinship_values,
        group_from=5,
        group_to=7,
    )
    assert selected.method == f"CMLM(g={expected_group})"

    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        CV=inputs.covariate,
        KI=inputs.kinship,
        model="CMLM",
        trait="Trait",
        PCA_total=2,
        maf_threshold=0.0,
        group_from=5,
        group_to=7,
        file_output=False,
    )

    assert not isinstance(py_result, dict)
    assert py_result.model == "CMLM"
    assert py_result.GWAS is not None
    nt.assert_allclose(py_result.GWAS["P.value"], selected.p_values)
    nt.assert_allclose(py_result.GWAS["effect"], np.round(selected.effects, 6))
