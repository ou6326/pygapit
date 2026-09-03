"""Result-object and filesystem output contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pygapit.gapit import GAPIT, GAPITResult, _output_prefix


def _small_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    taxa = [f"T{i:02d}" for i in range(12)]
    phenotype = pd.DataFrame({"Taxa": taxa, "height": np.linspace(1.0, 12.0, 12)})
    genotype = pd.DataFrame({
        "Taxa": taxa,
        "s1": np.tile([0.0, 1.0, 2.0], 4),
        "s2": np.tile([2.0, 1.0, 0.0], 4),
        "s3": np.tile([0.0, 2.0], 6),
        "s4": np.tile([1.0, 2.0, 0.0, 1.0], 3),
    })
    marker_map = pd.DataFrame({
        "SNP": ["s1", "s2", "s3", "s4"],
        "Chromosome": [1, 1, 2, 2],
        "Position": [10, 20, 10, 20],
    })
    return phenotype, genotype, marker_map


def test_file_output_false_does_not_create_output_directory(tmp_path: Path) -> None:
    phenotype, genotype, marker_map = _small_inputs()
    output_dir = tmp_path / "not-created"

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        PCA_total=2,
        file_output=False,
        output_dir=output_dir,
    )

    assert isinstance(result, GAPITResult)
    assert result.output_files is None
    assert not output_dir.exists()


def test_result_lists_successfully_written_output_files(tmp_path: Path) -> None:
    phenotype, genotype, marker_map = _small_inputs()
    output_dir = tmp_path / "nested" / "results"

    result = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        PCA_total=2,
        file_output=True,
        output_dir=output_dir,
    )

    assert isinstance(result, GAPITResult)
    assert result.output_files is not None
    written = result.output_files.paths()
    assert written
    assert all(path.is_file() for path in written)
    assert all(path.parent == output_dir for path in written)
    assert result.output_files.gwas.name == "GAPIT.GLM.height.GWAS.Results.csv"


def test_output_prefix_sanitizes_user_controlled_path_characters() -> None:
    assert _output_prefix("GLM", "height/../unsafe:trait") == (
        "GAPIT.GLM.height_.._unsafe_trait"
    )


def test_multiple_traits_have_distinct_shared_output_paths(tmp_path: Path) -> None:
    phenotype, genotype, marker_map = _small_inputs()
    phenotype["yield"] = phenotype["height"].to_numpy()[::-1]

    results = GAPIT(
        Y=phenotype,
        GD=genotype,
        GM=marker_map,
        model="GLM",
        PCA_total=2,
        file_output=True,
        output_dir=tmp_path,
    )

    assert isinstance(results, dict)
    height_files = results["height_GLM"].output_files
    yield_files = results["yield_GLM"].output_files
    assert height_files is not None
    assert yield_files is not None
    assert height_files.kinship != yield_files.kinship
    assert height_files.pca != yield_files.pca
    assert height_files.kinship_plot != yield_files.kinship_plot
    assert height_files.pca_plot != yield_files.pca_plot
    assert all(path.exists() for path in height_files.paths())
    assert all(path.exists() for path in yield_files.paths())
