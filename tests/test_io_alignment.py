"""Input alignment and matrix-validation tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pygapit.gapit import _cv_to_df, _ki_to_df
from pygapit.io.formats import (
    AlignedData,
    GenotypeData,
    PhenotypeData,
    align_inputs,
    align_taxa,
)


def _inputs() -> tuple[PhenotypeData, GenotypeData]:
    phenotype_taxa = np.array(["C", "A", "B", "E"])
    phenotype = PhenotypeData(
        Y=pd.DataFrame({"Taxa": phenotype_taxa, "trait": [30.0, 10.0, 20.0, 50.0]}),
        taxa=phenotype_taxa,
        trait_names=["trait"],
    )
    genotype = GenotypeData(
        GD=np.array([[10.0], [20.0], [30.0], [60.0]]),
        GM=pd.DataFrame({"SNP": ["s1"], "Chromosome": [1], "Position": [10]}),
        taxa=np.array(["A", "B", "C", "F"]),
    )
    return phenotype, genotype


def test_align_inputs_preserves_phenotype_order_and_reorders_all_inputs() -> None:
    phenotype, genotype = _inputs()
    covariates = pd.DataFrame({"Taxa": ["B", "C", "A"], "group": [2.0, 3.0, 1.0]})
    kinship = pd.DataFrame(
        {
            "Taxa": ["B", "A", "C"],
            "B": [22.0, 12.0, 23.0],
            "A": [12.0, 11.0, 13.0],
            "C": [23.0, 13.0, 33.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = align_inputs(phenotype, genotype, cv_df=covariates, ki_df=kinship)

    assert isinstance(aligned, AlignedData)
    np.testing.assert_array_equal(aligned.taxa, ["C", "A", "B"])
    np.testing.assert_array_equal(aligned.phenotypes["Taxa"], ["C", "A", "B"])
    np.testing.assert_array_equal(aligned.genotypes[:, 0], [30.0, 10.0, 20.0])
    assert aligned.covariates is not None
    assert aligned.kinship is not None
    np.testing.assert_array_equal(aligned.covariates[:, 0], [3.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        aligned.kinship,
        [[33.0, 13.0, 23.0], [13.0, 11.0, 12.0], [23.0, 12.0, 22.0]],
    )


def test_align_taxa_retains_legacy_mapping_interface() -> None:
    phenotype, genotype = _inputs()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = align_taxa(phenotype, genotype)

    assert set(aligned) == {"taxa", "Y", "GD", "GM"}
    np.testing.assert_array_equal(aligned["taxa"], ["C", "A", "B"])


@pytest.mark.parametrize("source", ["phenotype", "genotype"])
def test_align_inputs_rejects_duplicate_taxa(source: str) -> None:
    phenotype, genotype = _inputs()
    if source == "phenotype":
        phenotype.taxa[1] = "C"
    else:
        genotype.taxa[1] = "A"

    with pytest.raises(ValueError, match=rf"Duplicate taxa in {source}"):
        align_inputs(phenotype, genotype)


@pytest.mark.parametrize(
    ("matrix", "markers", "message"),
    [
        (np.ones(4), 1, "two-dimensional"),
        (np.ones((3, 1)), 1, "rows and genotype taxa"),
        (np.ones((4, 2)), 1, "columns and marker-map rows"),
    ],
)
def test_align_inputs_rejects_invalid_genotype_dimensions(
    matrix: np.ndarray, markers: int, message: str
) -> None:
    phenotype, genotype = _inputs()
    genotype.GD = matrix
    genotype.GM = genotype.GM.iloc[:markers]

    with pytest.raises(ValueError, match=message):
        align_inputs(phenotype, genotype)


def test_align_inputs_rejects_non_square_kinship_data() -> None:
    phenotype, genotype = _inputs()
    kinship = pd.DataFrame(
        {"Taxa": ["A", "B", "C"], "A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0]}
    )

    with pytest.raises(ValueError, match="square matrix"):
        align_inputs(phenotype, genotype, ki_df=kinship)


def test_array_kinship_must_match_phenotype_taxa() -> None:
    with pytest.raises(ValueError, match="matching phenotype taxa"):
        _ki_to_df(np.eye(2), np.array(["A", "B", "C"]))


def test_array_covariates_must_match_phenotype_taxa() -> None:
    with pytest.raises(ValueError, match="one row per phenotype taxon"):
        _cv_to_df(np.ones((2, 1)), np.array(["A", "B", "C"]))


def test_one_dimensional_covariates_become_one_column() -> None:
    result = _cv_to_df(np.array([1.0, 2.0]), np.array(["A", "B"]))

    assert result.columns.tolist() == ["Taxa", "CV1"]
    np.testing.assert_array_equal(result["CV1"], [1.0, 2.0])
