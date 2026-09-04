"""Top-level MLMM workflow alignment with GAPIT 3.5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest
from numpy.typing import NDArray

from pygapit.gapit import GAPIT
from pygapit.gwas.mlmm import _ext_bic, _normalize_kinship
from tests.cross_language.r_bridge import RBridge
from tests.cross_language.workflow import (
    assert_top_level_preparation,
    make_workflow_inputs,
    r_design_with_pca,
)


def test_corrected_ext_bic_excludes_covariates_from_marker_penalty(
    r_bridge: RBridge,
) -> None:
    """Lock the intentional model-choice divergence from GAPIT's penalty bug."""
    n = 100
    marker_count = 1_000
    baseline_fixed_effect_count = 5
    log_likelihoods = np.array([-100.0, -91.6])
    selected_marker_counts = np.array([0, 1], dtype=np.int64)
    fixed_effect_counts = baseline_fixed_effect_count + selected_marker_counts

    r_ext_bic = r_bridge.function(
        "function(log_lik, nfixed, n, m) {"
        " -2 * log_lik + (nfixed + 1) * log(n) +"
        " 2 * lchoose(m, nfixed - 1)"
        "}"
    )
    r_scores = r_bridge.float_array(
        r_ext_bic(
            r_bridge.float_vector(log_likelihoods),
            r_bridge.float_vector(fixed_effect_counts.astype(np.float64)),
            n,
            marker_count,
        )
    )
    py_scores = np.asarray([
        _ext_bic(
            log_likelihood,
            n,
            int(n_fixed),
            marker_count,
            int(n_selected),
        )
        for log_likelihood, n_fixed, n_selected in zip(
            log_likelihoods,
            fixed_effect_counts,
            selected_marker_counts,
            strict=True,
        )
    ])

    assert int(np.argmin(r_scores)) == 1
    assert int(np.argmin(py_scores)) == 0


def test_indefinite_kinship_characterization(
    r_bridge: RBridge,
    r_root: Path,
) -> None:
    """Reject an indefinite K that makes GAPIT return an invalid likelihood."""
    r_bridge.source(r_root, "GAPIT.emma.R")
    r_fit = r_bridge.function(
        "function(y, X, K) {"
        " fit <- tryCatch(emma.MLE(y, X, K), error=function(e) e);"
        " if (inherits(fit, 'error'))"
        "   return(list(ok=FALSE, value=conditionMessage(fit)));"
        " list(ok=TRUE, value=fit$ML)"
        "}"
    )
    sample_count = 6
    kinship = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -0.25])
    result = r_fit(
        r_bridge.float_vector(np.array([0.2, 1.1, -0.4, 0.8, -1.2, 0.5])),
        r_bridge.matrix(np.ones((sample_count, 1))),
        r_bridge.matrix(kinship),
    )
    r_likelihood = r_bridge.float_array(r_bridge.component(result, "value"))[0]

    assert np.isneginf(r_likelihood)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _normalize_kinship(kinship)

    rounding_noise = kinship.copy()
    rounding_noise[-1, -1] = -1e-10
    assert np.isfinite(_normalize_kinship(rounding_noise)).all()

    asymmetric = np.eye(sample_count)
    asymmetric[0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetric"):
        _normalize_kinship(asymmetric)


def test_top_level_mlmm_with_pca_cv_ki_and_missing_phenotype_matches_gapit(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    fixed_gapit_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Compare public MLMM selection and statistics with GAPIT's workflow."""
    inputs = make_workflow_inputs(
        fixed_genotypes,
        fixed_phenotype,
        fixed_covariate,
        fixed_gapit_inputs,
    )
    design, r_scores = r_design_with_pca(r_bridge, r_root, inputs, pca_total=2)
    for filename in (
        "GAPIT.emma.R",
        "GAPIT.mlmm_cof.R",
    ):
        r_bridge.source(r_root, filename)
    r_run = r_bridge.function(
        "function(y, X, cofs, K) {"
        " fit <- mlmm_cof(y, X, cofs, K, nbchunks=2, maxsteps=3);"
        " list(SNP=as.character(fit$opt_extBIC$out$SNP),"
        "      p=as.numeric(fit$opt_extBIC$out$pval),"
        "      effect=as.numeric(fit$opt_extBIC$out$effect),"
        "      cof=as.character(fit$opt_extBIC$cof),"
        "      seq=as.numeric(fit$seqQTN),"
        "      table=fit$step_table)"
        "}"
    )
    marker_names = inputs.marker_map["SNP"].astype(str).tolist()
    r_result = r_run(
        r_bridge.float_vector(inputs.phenotype_values),
        r_bridge.matrix(inputs.genotype_values, column_names=marker_names),
        r_bridge.matrix(design[:, 1:]),
        r_bridge.matrix(inputs.kinship_values),
    )
    py_result = GAPIT(
        Y=inputs.phenotype,
        GD=inputs.genotype,
        GM=inputs.marker_map,
        CV=inputs.covariate,
        KI=inputs.kinship,
        model="MLMM",
        trait="Trait",
        PCA_total=2,
        maf_threshold=0.0,
        file_output=False,
    )
    assert not isinstance(py_result, dict)
    assert py_result.GWAS is not None
    assert py_result.QTNs is not None
    assert_top_level_preparation(py_result, inputs, r_scores)

    r_markers = np.asarray(r_bridge.component(r_result, "SNP"), dtype=np.str_)
    missing_markers = iter(name for name in marker_names if name not in r_markers)
    r_markers = np.asarray(
        [name or next(missing_markers) for name in r_markers],
        dtype=np.str_,
    )
    canonical_order = np.asarray(
        [int(np.flatnonzero(r_markers == name)[0]) for name in marker_names],
        dtype=np.intp,
    )
    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "p"))[
        canonical_order
    ]
    r_effects = r_bridge.float_array(r_bridge.component(r_result, "effect"))[
        canonical_order
    ]
    r_cofactors = np.asarray(r_bridge.component(r_result, "cof"), dtype=np.str_)
    r_sequence = r_bridge.float_array(r_bridge.component(r_result, "seq"))

    nt.assert_array_equal(py_result.QTNs + 1, r_sequence.astype(np.int64))
    nt.assert_array_equal(
        np.asarray(marker_names)[py_result.QTNs],
        r_cofactors,
    )
    nt.assert_allclose(
        np.asarray(py_result.GWAS["P.value"], dtype=np.float64),
        r_p_values,
        rtol=1e-6,
        atol=1e-12,
    )
    nt.assert_array_equal(
        np.asarray(py_result.GWAS["effect"], dtype=np.float64),
        np.round(r_effects, 6),
    )
