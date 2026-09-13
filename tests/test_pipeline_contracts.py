"""Top-level GAPIT orchestration and parameter contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pygapit._typing import FloatMatrix, FloatVector, IntVector, StrVector
from pygapit.gapit import (
    GAPIT,
    GAPITResult,
    ModelRunResult,
    _align_multiple_gwas,
    _normalize_models,
    _select_traits,
)
from pygapit.gs.blup import cblup, gblup, sblup, select_super_qtns
from pygapit.gwas.blink import _candidate_mask
from pygapit.gwas.mlm import mlm_gwas
from pygapit.io.formats import GenotypeData, PhenotypeData, align_inputs
from pygapit.io.storage import (
    ArrayGenotypeStore,
    open_genotype_store,
    write_numpy_genotype,
)
from pygapit.stats.emma import EMMASpectrum, prepare_emma_spectrum
from pygapit.stats.kinship import vanraden_kinship, zhang_kinship
from pygapit.stats.pca import PCAResult, compute_pca


class _RecordingLabeledStore:
    def __init__(self, genotype: GenotypeData) -> None:
        self._values: FloatMatrix = genotype.GD
        self.taxa: StrVector = genotype.taxa
        self.marker_ids: StrVector = np.asarray(genotype.GM["SNP"], dtype=str)
        self.chromosomes: StrVector = np.asarray(genotype.GM["Chromosome"], dtype=str)
        self.positions: FloatVector = np.asarray(
            genotype.GM["Position"], dtype=np.float64
        )
        self.marker_slices: list[slice] = []
        self.sample_requests: list[IntVector | slice | None] = []

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> FloatMatrix:
        raise AssertionError("GAPIT must not materialize a complete genotype store")

    def read_markers(
        self,
        marker_slice: slice,
        sample_indices: IntVector | slice | None = None,
    ) -> FloatMatrix:
        self.marker_slices.append(marker_slice)
        self.sample_requests.append(sample_indices)
        block = self._values[:, marker_slice]
        if sample_indices is not None:
            block = block[sample_indices]
        return block.copy()


def _inputs(n: int = 12, invariant: bool = False) -> tuple[pd.DataFrame, ...]:
    taxa = [f"T{i:02d}" for i in range(n)]
    phenotype = pd.DataFrame({
        "Taxa": taxa,
        "height": np.linspace(1.0, n, n),
        "yield": np.linspace(n, 1.0, n),
    })
    if invariant:
        marker_values = np.zeros((n, 4))
    else:
        marker_values = np.column_stack([
            np.arange(n) % 3,
            (np.arange(n) + 1) % 3,
            np.arange(n) % 2,
            (np.arange(n) // 2) % 3,
        ])
    genotype = pd.DataFrame(marker_values, columns=["s1", "s2", "s3", "s4"])
    genotype.insert(0, "Taxa", taxa)
    marker_map = pd.DataFrame({
        "SNP": ["s1", "s2", "s3", "s4"],
        "Chromosome": [1, 1, 2, 2],
        "Position": [10, 20, 10, 20],
    })
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
    with pytest.raises(ValueError, match="marker_workspace_mib"):
        GAPIT(marker_workspace_mib=0.0)
    with pytest.raises(TypeError, match="marker_workspace_mib"):
        GAPIT(marker_workspace_mib=True)
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


@pytest.mark.parametrize("model", ["GLM", "MLM", "gBLUP", "sBLUP"])
def test_gapit_disk_store_matches_aligned_in_memory_pipeline(
    tmp_path: Path,
    model: str,
) -> None:
    phenotype, genotype, marker_map = _inputs()
    phenotype.loc[0, "height"] = np.nan
    reordered = genotype.iloc[::-1].reset_index(drop=True)
    genotype_data = GenotypeData.from_numeric_frame(reordered, marker_map)
    store_path = tmp_path / "genotype-store"
    write_numpy_genotype(store_path, genotype_data, marker_chunk_size=2)

    expected = GAPIT(
        Y=phenotype,
        GD=reordered,
        GM=marker_map,
        model=model,
        trait="height",
        PCA_total=1,
        maf_threshold=0.0,
        marker_workspace_mib=0.0002,
        file_output=False,
    )
    with open_genotype_store(store_path, backend="numpy") as store:
        actual = GAPIT(
            Y=phenotype,
            GD=store,
            model=model,
            trait="height",
            PCA_total=1,
            maf_threshold=0.0,
            marker_workspace_mib=0.0002,
            file_output=False,
        )

    assert isinstance(expected, GAPITResult)
    assert isinstance(actual, GAPITResult)
    assert expected.GWAS is not None
    assert actual.GWAS is not None
    assert expected.kinship is not None
    assert actual.kinship is not None
    pd.testing.assert_frame_equal(actual.GWAS, expected.GWAS)
    np.testing.assert_allclose(actual.kinship, expected.kinship, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual.taxa, expected.taxa)
    if expected.Pred is None:
        assert actual.Pred is None
    else:
        assert actual.Pred is not None
        pd.testing.assert_frame_equal(actual.Pred, expected.Pred)


def test_gapit_combined_store_pipeline_never_materializes_complete_genotype() -> None:
    phenotype, genotype, marker_map = _inputs()
    phenotype.loc[0, "height"] = np.nan
    reordered = genotype.iloc[::-1].reset_index(drop=True)
    store = _RecordingLabeledStore(
        GenotypeData.from_numeric_frame(reordered, marker_map)
    )

    result = GAPIT(
        Y=phenotype,
        GD=store,
        model=["GLM", "MLM"],
        trait="height",
        PCA_total=1,
        maf_threshold=0.2,
        marker_workspace_mib=0.0002,
        file_output=False,
    )

    assert isinstance(result, dict)
    assert set(result) == {"height_GLM", "height_MLM"}
    assert len(store.marker_slices) > 2
    assert all(
        marker_slice.start is not None
        and marker_slice.stop is not None
        and marker_slice.stop - marker_slice.start <= 2
        for marker_slice in store.marker_slices
    )
    assert all(request is not None for request in store.sample_requests)


def test_super_store_reuses_one_bounded_candidate_pool() -> None:
    phenotype, genotype, marker_map = _inputs()
    genotype_data = GenotypeData.from_numeric_frame(genotype, marker_map)
    store = _RecordingLabeledStore(genotype_data)
    y = phenotype["height"].to_numpy(dtype=np.float64)
    design = np.ones((len(y), 1), dtype=np.float64)

    selection = select_super_qtns(
        y,
        design,
        store,
        store.chromosomes,
        store.positions,
        np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64),
        bin_size=10,
        candidate_counts=[1, 2, 4],
    )

    assert selection.qtn_indices.size > 0
    assert store.marker_slices == [slice(0, 4)]


def test_align_inputs_validates_custom_store_metadata_lengths() -> None:
    phenotype, genotype, marker_map = _inputs()
    store = _RecordingLabeledStore(
        GenotypeData.from_numeric_frame(genotype, marker_map)
    )
    store.marker_ids = store.marker_ids[:-1]

    with pytest.raises(ValueError, match="marker ids must contain 4 values"):
        align_inputs(PhenotypeData.from_frame(phenotype), store)


def test_gapit_requires_labeled_store_metadata() -> None:
    phenotype, genotype, _ = _inputs()
    unlabeled = ArrayGenotypeStore(
        genotype.drop(columns="Taxa").to_numpy(dtype=np.float64)
    )

    with pytest.raises(TypeError, match="LabeledGenotypeStore"):
        GAPIT(Y=phenotype, GD=unlabeled, model="GLM", file_output=False)


def test_gapit_disk_store_rejects_unsupported_paths(tmp_path: Path) -> None:
    phenotype, genotype, marker_map = _inputs()
    genotype_data = GenotypeData.from_numeric_frame(genotype, marker_map)
    store_path = tmp_path / "genotype-store"
    write_numpy_genotype(store_path, genotype_data)

    with open_genotype_store(store_path, backend="numpy") as store:
        with pytest.raises(ValueError, match="supports GLM, MLM, gBLUP, and sBLUP"):
            GAPIT(Y=phenotype, GD=store, model="BLINK", file_output=False)
        with pytest.raises(ValueError, match="requires kinship_algorithm"):
            GAPIT(
                Y=phenotype,
                GD=store,
                model="GLM",
                kinship_algorithm="Zhang",
                file_output=False,
            )
        prediction = GAPIT(
            Y=phenotype,
            GD=store,
            model="GLM",
            buspred=True,
            trait="height",
            file_output=False,
        )
        assert isinstance(prediction, GAPITResult)
        assert prediction.Pred is not None
        with pytest.raises(ValueError, match="cBLUP prediction"):
            GAPIT(
                Y=phenotype,
                GD=store,
                model="GLM",
                prediction_model="cBLUP",
                file_output=False,
            )
        with pytest.raises(ValueError, match="GM must not be provided"):
            GAPIT(
                Y=phenotype,
                GD=store,
                GM=marker_map,
                model="GLM",
                file_output=False,
            )


def test_gapit_disk_store_requires_preimputed_values(tmp_path: Path) -> None:
    phenotype, genotype, marker_map = _inputs()
    genotype.loc[0, "s1"] = np.nan
    genotype_data = GenotypeData.from_numeric_frame(
        genotype,
        marker_map,
        impute_method="none",
    )
    store_path = tmp_path / "genotype-store"
    write_numpy_genotype(store_path, genotype_data)

    with (
        open_genotype_store(store_path, backend="numpy") as store,
        pytest.raises(ValueError, match="finite, pre-imputed"),
    ):
        GAPIT(Y=phenotype, GD=store, model="GLM", file_output=False)


@pytest.mark.parametrize(
    ("model", "target", "implementation"),
    [
        ("GBLUP", "pygapit.gapit.gblup", gblup),
        ("CBLUP", "pygapit.gapit.cblup", cblup),
        ("SBLUP", "pygapit.gapit.sblup", sblup),
    ],
)
def test_prediction_models_reuse_their_fitted_result(
    model: str,
    target: str,
    implementation: object,
) -> None:
    phenotype, genotype, marker_map = _inputs()

    with patch(target, wraps=implementation) as prediction_fit:
        result = GAPIT(
            Y=phenotype,
            GD=genotype,
            GM=marker_map,
            model=model,
            PCA_total=1,
            maf_threshold=0.0,
            group_to=6,
            super_qtn_counts=(1, 2),
            file_output=False,
        )

    assert isinstance(result, dict)
    assert prediction_fit.call_count == 2
    assert all(item.Pred is not None for item in result.values())


def test_mlm_and_sblup_share_one_marker_scan_per_trait() -> None:
    phenotype, genotype, marker_map = _inputs()

    with patch("pygapit.gapit.mlm_gwas", wraps=mlm_gwas) as mlm_scan:
        result = GAPIT(
            Y=phenotype,
            GD=genotype,
            GM=marker_map,
            model=["SBLUP", "MLM"],
            PCA_total=1,
            maf_threshold=0.0,
            super_qtn_counts=(1, 2),
            file_output=False,
        )

    assert isinstance(result, dict)
    assert mlm_scan.call_count == 2
    for trait in ("height", "yield"):
        sblup_gwas = result[f"{trait}_SBLUP"].GWAS
        mlm_gwas_result = result[f"{trait}_MLM"].GWAS
        assert sblup_gwas is not None
        assert mlm_gwas_result is not None
        np.testing.assert_allclose(
            sblup_gwas["P.value"].to_numpy(dtype=np.float64),
            mlm_gwas_result["P.value"].to_numpy(dtype=np.float64),
        )


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
        *,
        marker_workspace_mib: float = 32.0,
    ) -> PCAResult:
        nonlocal pca_calls
        pca_calls += 1
        return compute_pca(
            GD,
            n_components=n_components,
            maf_filter=maf_filter,
            marker_workspace_mib=marker_workspace_mib,
        )

    def counting_kinship(
        GD: FloatMatrix,
        *,
        marker_workspace_mib: float = 32.0,
    ) -> FloatMatrix:
        nonlocal kinship_calls
        kinship_calls += 1
        return vanraden_kinship(
            GD,
            marker_workspace_mib=marker_workspace_mib,
        )

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


def test_gapits_forwards_workspace_budget_to_vanraden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phenotype, genotype, marker_map = _inputs()
    kinship_budgets: list[float] = []
    pca_budgets: list[float] = []

    def recording_kinship(
        GD: FloatMatrix,
        *,
        marker_workspace_mib: float = 32.0,
    ) -> FloatMatrix:
        kinship_budgets.append(marker_workspace_mib)
        return np.eye(len(GD), dtype=np.float64)

    def recording_pca(
        GD: FloatMatrix,
        n_components: int = 3,
        maf_filter: float = 0.05,
        *,
        marker_workspace_mib: float = 32.0,
    ) -> PCAResult:
        pca_budgets.append(marker_workspace_mib)
        return compute_pca(
            GD,
            n_components=n_components,
            maf_filter=maf_filter,
            marker_workspace_mib=marker_workspace_mib,
        )

    monkeypatch.setattr("pygapit.gapit.vanraden_kinship", recording_kinship)
    monkeypatch.setattr("pygapit.gapit.compute_pca", recording_pca)
    GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        trait="height",
        PCA_total=1,
        maf_threshold=0.0,
        marker_workspace_mib=0.001,
        file_output=False,
    )

    assert kinship_budgets == [0.001]
    assert pca_budgets == [0.001]


@pytest.mark.parametrize(
    ("missing_cells", "expected_calls"),
    [
        ((), 1),
        (((0, "height"), (1, "yield")), 2),
    ],
)
def test_mlm_spectrum_cache_respects_observed_taxa(
    monkeypatch: pytest.MonkeyPatch,
    missing_cells: tuple[tuple[int, str], ...],
    expected_calls: int,
) -> None:
    phenotype, genotype, marker_map = _inputs()
    for row, column in missing_cells:
        phenotype.loc[row, column] = np.nan

    spectrum_calls = 0

    def counting_spectrum(K: FloatMatrix, X: FloatMatrix) -> EMMASpectrum:
        nonlocal spectrum_calls
        spectrum_calls += 1
        return prepare_emma_spectrum(K, X)

    monkeypatch.setattr("pygapit.gapit.prepare_emma_spectrum", counting_spectrum)
    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="MLM",
        PCA_total=1,
        maf_threshold=0.0,
        file_output=False,
    )

    assert isinstance(result, dict)
    assert spectrum_calls == expected_calls


@pytest.mark.parametrize("model", ["GLM", "MLM"])
def test_cached_trait_results_match_independent_runs(model: str) -> None:
    phenotype, genotype, marker_map = _inputs()

    combined = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model=model,
        PCA_total=1,
        file_output=False,
    )
    height = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model=model,
        trait="height",
        PCA_total=1,
        file_output=False,
    )
    yield_result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model=model,
        trait="yield",
        PCA_total=1,
        file_output=False,
    )

    assert isinstance(combined, dict)
    assert isinstance(height, GAPITResult)
    assert isinstance(yield_result, GAPITResult)
    combined_height = combined[f"height_{model}"]
    combined_yield = combined[f"yield_{model}"]
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
        GWAS=pd.DataFrame({
            "SNP": ["s1", "s2"],
            "Chr": ["1", "1"],
            "Pos": [10.0, 20.0],
            "P.value": [0.01, 0.02],
        }),
        model="GLM",
    )
    second = GAPITResult(
        GWAS=pd.DataFrame({
            "SNP": ["s3", "s2"],
            "Chr": ["2", "1"],
            "Pos": [5.0, 20.0],
            "P.value": [0.03, 0.2],
        }),
        model="MLM",
    )

    markers, aligned = _align_multiple_gwas([first, second])

    assert markers["SNP"].tolist() == ["s1", "s2", "s3"]
    np.testing.assert_allclose(aligned[0][1][:2], [0.01, 0.02])
    assert np.isnan(aligned[0][1][2])
    assert np.isnan(aligned[1][1][0])
    np.testing.assert_allclose(aligned[1][1][1:], [0.2, 0.03])
