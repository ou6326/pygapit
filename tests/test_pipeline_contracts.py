"""Top-level GAPIT orchestration and parameter contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygapit._typing import FloatMatrix
from pygapit.gapit import (
    GAPIT,
    GAPITResult,
    ModelRunResult,
    _align_multiple_gwas,
    _normalize_models,
    _select_traits,
)
from pygapit.gwas.blink import _candidate_mask
from pygapit.stats.kinship import vanraden_kinship, zhang_kinship
from pygapit.stats.pca import PCAResult, compute_pca


def _inputs(n: int = 12, invariant: bool = False) -> tuple[pd.DataFrame, ...]:
    taxa = [f"T{i:02d}" for i in range(n)]
    phenotype = pd.DataFrame(
        {
            "Taxa": taxa,
            "height": np.linspace(1.0, n, n),
            "yield": np.linspace(n, 1.0, n),
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
    assert _normalize_models(["sBLUP"]) == ("SBLUP",)


def test_invalid_numeric_options_fail_before_input_loading() -> None:
    with pytest.raises(ValueError, match="PCA_total"):
        GAPIT(PCA_total=-1)
    with pytest.raises(ValueError, match="maf_threshold"):
        GAPIT(maf_threshold=0.6)
    with pytest.raises(ValueError, match="p_threshold"):
        GAPIT(p_threshold=0.0)
    with pytest.raises(ValueError, match="maxLoop"):
        GAPIT(maxLoop=0)
    with pytest.raises(ValueError, match="super_bin_size"):
        GAPIT(super_bin_size=0)
    with pytest.raises(ValueError, match="super_qtn_counts"):
        GAPIT(super_qtn_counts=[])
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


@pytest.mark.parametrize(
    ("missing_cells", "expected_calls"),
    [
        ((), 1),
        (((0, "height"), (1, "yield")), 2),
    ],
)
def test_trait_preparation_cache_respects_observed_taxa(
    monkeypatch: pytest.MonkeyPatch,
    missing_cells: tuple[tuple[int, str], ...],
    expected_calls: int,
) -> None:
    phenotype, genotype, marker_map = _inputs()
    for row, column in missing_cells:
        phenotype.loc[row, column] = np.nan

    pca_calls = 0
    kinship_calls = 0

    def counting_pca(
        GD: FloatMatrix,
        n_components: int = 3,
        maf_filter: float = 0.05,
    ) -> PCAResult:
        nonlocal pca_calls
        pca_calls += 1
        return compute_pca(GD, n_components=n_components, maf_filter=maf_filter)

    def counting_kinship(GD: FloatMatrix) -> FloatMatrix:
        nonlocal kinship_calls
        kinship_calls += 1
        return vanraden_kinship(GD)

    monkeypatch.setattr("pygapit.gapit.compute_pca", counting_pca)
    monkeypatch.setattr("pygapit.gapit.vanraden_kinship", counting_kinship)

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        PCA_total=1,
        file_output=False,
    )

    assert isinstance(result, dict)
    assert set(result) == {"height_GLM", "yield_GLM"}
    assert pca_calls == expected_calls
    assert kinship_calls == expected_calls


def test_cached_trait_results_match_independent_runs() -> None:
    phenotype, genotype, marker_map = _inputs()

    combined = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        PCA_total=1,
        file_output=False,
    )
    height = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        trait="height",
        PCA_total=1,
        file_output=False,
    )
    yield_result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        trait="yield",
        PCA_total=1,
        file_output=False,
    )

    assert isinstance(combined, dict)
    assert isinstance(height, GAPITResult)
    assert isinstance(yield_result, GAPITResult)
    combined_height = combined["height_GLM"]
    combined_yield = combined["yield_GLM"]
    assert combined_height.GWAS is not None
    assert combined_yield.GWAS is not None
    assert height.GWAS is not None
    assert yield_result.GWAS is not None
    pd.testing.assert_frame_equal(combined_height.GWAS, height.GWAS)
    pd.testing.assert_frame_equal(combined_yield.GWAS, yield_result.GWAS)
    assert combined_height.kinship is combined_yield.kinship
    assert combined_height.pca is combined_yield.pca


def test_blink_fdr_cut_uses_gapit_threshold() -> None:
    p_values = np.array([0.001, 0.01, 0.04, 0.5], dtype=np.float64)

    mask = _candidate_mask(p_values, p_threshold=0.25, fdr_alpha=0.05)

    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))


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


def test_labeled_incidence_and_kinship_are_aligned_by_name() -> None:
    phenotype, genotype, marker_map = _inputs()
    taxa = phenotype["Taxa"].astype(str).tolist()
    raw_incidence = np.zeros((len(taxa), 3))
    raw_incidence[np.arange(len(taxa)), np.arange(len(taxa)) % 3] = 1.0
    incidence = pd.DataFrame(raw_incidence, columns=["A", "B", "C"])
    incidence.insert(0, "Taxa", taxa)
    incidence = incidence.iloc[::-1][["Taxa", "C", "A", "B"]].reset_index(drop=True)

    canonical_kinship = pd.DataFrame(
        [[1.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 2.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    kinship = canonical_kinship.loc[["B", "C", "A"], ["C", "A", "B"]]

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        KI=kinship,
        Z=incidence,
        model="GLM",
        trait="height",
        PCA_total=1,
        maf_threshold=0.0,
        file_output=False,
    )

    assert not isinstance(result, dict)
    assert result.kinship is not None
    expected = raw_incidence @ canonical_kinship.to_numpy() @ raw_incidence.T
    np.testing.assert_allclose(result.kinship, expected)


def test_labeled_incidence_rejects_mismatched_random_effects() -> None:
    phenotype, genotype, marker_map = _inputs()
    taxa = phenotype["Taxa"].astype(str).tolist()
    incidence = pd.DataFrame({"Taxa": taxa, "A": 1.0, "B": 0.0})
    kinship = pd.DataFrame(np.eye(2), index=["A", "C"], columns=["A", "C"])

    with pytest.raises(ValueError, match="Z columns must exactly match KI"):
        GAPIT(
            Y=phenotype,
            GD=genotype,
            GM=marker_map,
            KI=kinship,
            Z=incidence,
            model="GLM",
            trait="height",
            file_output=False,
        )


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


def test_multiple_analysis_aligns_models_by_marker_coordinates() -> None:
    first = GAPITResult(
        GWAS=pd.DataFrame(
            {
                "SNP": ["s1", "s2"],
                "Chr": ["1", "1"],
                "Pos": [10.0, 20.0],
                "P.value": [0.01, 0.02],
            }
        ),
        model="GLM",
    )
    second = GAPITResult(
        GWAS=pd.DataFrame(
            {
                "SNP": ["s3", "s2"],
                "Chr": ["2", "1"],
                "Pos": [5.0, 20.0],
                "P.value": [0.03, 0.2],
            }
        ),
        model="MLM",
    )

    markers, aligned = _align_multiple_gwas([first, second])

    assert markers["SNP"].tolist() == ["s1", "s2", "s3"]
    np.testing.assert_allclose(aligned[0][1][:2], [0.01, 0.02])
    assert np.isnan(aligned[0][1][2])
    assert np.isnan(aligned[1][1][0])
    np.testing.assert_allclose(aligned[1][1][1:], [0.2, 0.03])
