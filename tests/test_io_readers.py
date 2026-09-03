"""Reader and in-memory input normalization tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygapit.gapit import _load_data
from pygapit.io.formats import read_hapmap, read_numeric, read_phenotype


def _phenotype() -> pd.DataFrame:
    return pd.DataFrame({"Taxa": ["A", "B"], "trait": [1.0, 2.0]})


def _marker_map(*snps: str) -> pd.DataFrame:
    return pd.DataFrame({
        "SNP": list(snps),
        "Chromosome": [1] * len(snps),
        "Position": np.arange(1, len(snps) + 1),
    })


def _hapmap(position: object = 10) -> pd.DataFrame:
    metadata = [
        "rs",
        "alleles",
        "chrom",
        "pos",
        "strand",
        "assembly",
        "center",
        "protLSID",
        "assayLSID",
        "panelLSID",
        "QCcode",
    ]
    return pd.DataFrame(
        [
            [
                "s1",
                "A/T",
                1,
                position,
                "+",
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "AA",
                "TT",
            ]
        ],
        columns=[*metadata, "A", "B"],
    )


@pytest.mark.parametrize(
    ("phenotype", "message"),
    [
        (pd.DataFrame({"Taxa": ["A"]}), "at least one trait"),
        (pd.DataFrame(columns=["Taxa", "trait"]), "at least one phenotype row"),
        (pd.DataFrame({"Taxa": ["A"], "trait": ["high"]}), "must contain numeric"),
        (pd.DataFrame({"Taxa": [None], "trait": [1.0]}), "missing values"),
        (pd.DataFrame({"Taxa": ["A"], "trait": [np.inf]}), "contain infinity"),
    ],
)
def test_phenotype_frame_rejects_malformed_data(
    phenotype: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_data(phenotype, None, np.ones((1, 1)), _marker_map("s1"), "none")


def test_read_phenotype_accepts_path_objects(tmp_path: Path) -> None:
    path = tmp_path / "phenotype.tsv"
    _phenotype().to_csv(path, sep="\t", index=False)

    result = read_phenotype(path)

    np.testing.assert_array_equal(result.taxa, ["A", "B"])
    assert result.trait_names == ["trait"]


def test_numeric_reader_rejects_marker_order_mismatch(tmp_path: Path) -> None:
    genotype_path = tmp_path / "genotype.tsv"
    pd.DataFrame({"Taxa": ["A", "B"], "s2": [0, 1], "s1": [2, 1]}).to_csv(
        genotype_path, sep="\t", index=False
    )

    with pytest.raises(ValueError, match="match marker-map rows in order"):
        read_numeric(genotype_path, _marker_map("s1", "s2"))


def test_numeric_reader_rejects_non_numeric_values(tmp_path: Path) -> None:
    genotype_path = tmp_path / "genotype.tsv"
    pd.DataFrame({"Taxa": ["A"], "s1": ["heterozygous"]}).to_csv(
        genotype_path, sep="\t", index=False
    )

    with pytest.raises(ValueError, match="SNP values must be numeric"):
        read_numeric(genotype_path, _marker_map("s1"))


def test_hapmap_reader_rejects_invalid_position() -> None:
    with pytest.raises(ValueError, match="positions must be numeric"):
        read_hapmap(_hapmap(position="unknown"))


def test_hapmap_reader_rejects_missing_taxa_columns() -> None:
    hapmap = _hapmap().iloc[:, :11]

    with pytest.raises(ValueError, match="11 metadata columns and taxa"):
        read_hapmap(hapmap)


def test_ndarray_genotype_uses_phenotype_taxa_order() -> None:
    phenotype, genotype = _load_data(
        _phenotype(), None, np.array([[0.0], [2.0]]), _marker_map("s1"), "none"
    )

    np.testing.assert_array_equal(genotype.taxa, phenotype.taxa)


def test_ndarray_genotype_rejects_row_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one row per supplied taxon"):
        _load_data(_phenotype(), None, np.array([[0.0]]), _marker_map("s1"), "none")


def test_load_data_rejects_ambiguous_genotype_formats() -> None:
    with pytest.raises(ValueError, match="not both input formats"):
        _load_data(
            _phenotype(),
            _hapmap(),
            np.ones((2, 1)),
            _marker_map("s1"),
            "none",
        )


def test_ndarray_genotype_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="array values must be numeric"):
        _load_data(
            _phenotype(),
            None,
            np.array([["AA"], ["TT"]]),
            _marker_map("s1"),
            "none",
        )


def test_marker_map_rejects_duplicate_snp_identifiers() -> None:
    with pytest.raises(ValueError, match="duplicate SNP identifiers"):
        _load_data(
            _phenotype(),
            None,
            np.ones((2, 2)),
            _marker_map("s1", "s1"),
            "none",
        )


@pytest.mark.parametrize(
    ("genotype", "marker_map"),
    [(np.ones((2, 1)), None), (None, _marker_map("s1"))],
)
def test_load_data_requires_numeric_genotype_and_map_together(
    genotype: np.ndarray | None, marker_map: pd.DataFrame | None
) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        _load_data(_phenotype(), None, genotype, marker_map, "none")
