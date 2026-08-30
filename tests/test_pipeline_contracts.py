"""Top-level GAPIT orchestration and parameter contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygapit.gapit import GAPIT, ModelRunResult, _normalize_models, _select_traits
from pygapit.gwas.blink import _candidate_mask
from pygapit.stats.kinship import zhang_kinship
from pygapit.stats.testing import benjamini_hochberg


def _inputs(n: int = 12, invariant: bool = False) -> tuple[pd.DataFrame, ...]:
    taxa = [f"T{i:02d}" for i in range(n)]
    phenotype = pd.DataFrame(
        {
            "Taxa": taxa,
            "height": np.linspace(1.0, float(n), n),
            "yield": np.linspace(float(n), 1.0, n),
        }
    )
    if invariant:
        marker_values = np.zeros((n, 4))
    else:
        marker_values = np.column_stack(
            [
                np.arange(n) % 3,
                (np.arange(n) + 1) % 3,
                np.arange(n) % 2,
                (np.arange(n) // 2) % 3,
            ]
        )
    genotype = pd.DataFrame(marker_values, columns=["s1", "s2", "s3", "s4"])
    genotype.insert(0, "Taxa", taxa)
    marker_map = pd.DataFrame(
        {
            "SNP": ["s1", "s2", "s3", "s4"],
            "Chromosome": [1, 1, 2, 2],
            "Position": [10, 20, 10, 20],
        }
    )
    return phenotype, genotype, marker_map


def test_model_run_result_rejects_inconsistent_marker_arrays() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ModelRunResult(np.ones(2), np.ones(1), np.ones(2))


def test_model_normalization_rejects_empty_duplicate_and_unknown_models() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _normalize_models([])
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_models(["glm", "GLM"])
    with pytest.raises(ValueError, match="Unknown model"):
        _normalize_models(["unknown"])
    with pytest.raises(ValueError, match=r"pygapit\.sblup"):
        _normalize_models(["sBLUP"])


def test_invalid_numeric_options_fail_before_input_loading() -> None:
    with pytest.raises(ValueError, match="PCA_total"):
        GAPIT(PCA_total=-1)
    with pytest.raises(ValueError, match="maf_threshold"):
        GAPIT(maf_threshold=0.6)
    with pytest.raises(ValueError, match="p_threshold"):
        GAPIT(p_threshold=0.0)
    with pytest.raises(ValueError, match="maxLoop"):
        GAPIT(maxLoop=0)
    with pytest.raises(ValueError, match="provided together"):
        GAPIT(h2=0.5)


def test_trait_selection_has_explicit_membership_and_bounds() -> None:
    with pytest.raises(ValueError, match="Unknown trait"):
        _select_traits(["height"], "yield")
    with pytest.raises(ValueError, match="out of range"):
        _select_traits(["height"], 1)
    with pytest.raises(TypeError, match="not bool"):
        _select_traits(["height"], True)


def test_maf_filter_rejects_analysis_with_no_remaining_markers() -> None:
    phenotype, genotype, marker_map = _inputs(invariant=True)

    with pytest.raises(ValueError, match="No SNPs remain"):
        GAPIT(
            Y=phenotype,
            GD=genotype,
            GM=marker_map,
            model="GLM",
            trait="height",
            file_output=False,
        )


def test_no_completed_traits_is_an_explicit_error() -> None:
    phenotype, genotype, marker_map = _inputs(n=9)

    with (
        pytest.warns(UserWarning, match="Only 9 individuals"),
        pytest.raises(ValueError, match="No analyses completed"),
    ):
        GAPIT(
            Y=phenotype,
            GD=genotype,
            GM=marker_map,
            model="GLM",
            trait="height",
            file_output=False,
        )


def test_multiple_traits_and_models_return_named_results() -> None:
    phenotype, genotype, marker_map = _inputs()

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model=["GLM", "MLM"],
        PCA_total=2,
        file_output=False,
    )

    assert isinstance(result, dict)
    assert set(result) == {"height_GLM", "height_MLM", "yield_GLM", "yield_MLM"}


def test_blink_fdr_cut_uses_bh_adjusted_p_values() -> None:
    p_values = np.array([0.001, 0.01, 0.04, 0.5], dtype=np.float64)

    mask = _candidate_mask(p_values, p_threshold=0.25, fdr_alpha=0.05)

    np.testing.assert_array_equal(mask, benjamini_hochberg(p_values) <= 0.05)


def test_zhang_kinship_is_exposed_through_gapit() -> None:
    phenotype, genotype, marker_map = _inputs()

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        trait="height",
        PCA_total=1,
        maf_threshold=0.0,
        kinship_algorithm="Zhang",
        file_output=False,
    )

    assert not isinstance(result, dict)
    expected = zhang_kinship(genotype.iloc[:, 1:].to_numpy(dtype=np.float64))
    assert result.kinship is not None
    np.testing.assert_allclose(result.kinship, expected)


def test_incidence_matrix_expands_random_effect_kinship() -> None:
    phenotype, genotype, marker_map = _inputs()
    incidence = np.zeros((len(phenotype), 3))
    incidence[np.arange(len(phenotype)), np.arange(len(phenotype)) % 3] = 1.0
    random_kinship = np.array([[1.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 2.0]])

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        KI=random_kinship,
        Z=incidence,
        model="GLM",
        trait="height",
        PCA_total=1,
        maf_threshold=0.0,
        file_output=False,
    )

    assert not isinstance(result, dict)
    assert result.kinship is not None
    np.testing.assert_allclose(result.kinship, incidence @ random_kinship @ incidence.T)


def test_prediction_model_overrides_default_gs_path() -> None:
    phenotype, genotype, marker_map = _inputs()

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        prediction_model="cBLUP",
        trait="height",
        PCA_total=1,
        maf_threshold=0.0,
        file_output=False,
    )

    assert not isinstance(result, dict)
    assert result.Pred is not None
    assert list(result.Pred["Taxa"]) == list(phenotype["Taxa"])


def test_sblup_prediction_override_requires_selected_qtns() -> None:
    phenotype, genotype, marker_map = _inputs()

    with pytest.raises(ValueError, match="requires selected QTNs"):
        GAPIT(
            Y=phenotype,
            GD=genotype,
            GM=marker_map,
            model="GLM",
            prediction_model="sBLUP",
            trait="height",
            PCA_total=1,
            maf_threshold=0.0,
            file_output=False,
        )


def test_multiple_analysis_writes_combined_plots(tmp_path: Path) -> None:
    phenotype, genotype, marker_map = _inputs()

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model=["GLM", "MLM"],
        trait="height",
        PCA_total=1,
        maf_threshold=0.0,
        Multiple_analysis=True,
        file_output=True,
        output_dir=tmp_path,
    )

    assert isinstance(result, dict)
    for model_result in result.values():
        assert len(model_result.multiple_output_files) == 2
        assert all(path.exists() for path in model_result.multiple_output_files)
