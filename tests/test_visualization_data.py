"""Backend-independent visualization data contracts."""

from __future__ import annotations

import numpy as np
import pytest

from pygapit.visualization import prepare_genomic_axis, prepare_manhattan_data


def test_prepare_manhattan_data_reuses_compatible_inputs() -> None:
    snps = np.asarray(["s1", "s2", "s3", "s4"])
    chromosomes = np.asarray(["1", "1", "2", "2"])
    positions = np.asarray([10.0, 20.0, 5.0, 15.0])
    p_values = np.asarray([0.05, np.nan, 0.0, 1e-4])

    prepared = prepare_manhattan_data(
        snps,
        chromosomes,
        positions,
        p_values,
    )

    assert np.shares_memory(prepared.snp_names, snps)
    assert np.shares_memory(prepared.chromosomes, chromosomes)
    assert np.shares_memory(prepared.positions, positions)
    assert np.shares_memory(prepared.p_values, p_values)
    np.testing.assert_allclose(prepared.x_values, [0.0, 10.0, 5_000_010.0, 5_000_020.0])
    np.testing.assert_allclose(
        prepared.log_p_values,
        [-np.log10(0.05), 0.0, 0.0, 4.0],
    )
    assert prepared.chromosome_labels == ("1", "2")
    np.testing.assert_allclose(prepared.chromosome_centers, [5.0, 5_000_015.0])
    assert prepared.significance_threshold == pytest.approx(0.0125)
    assert prepared.suggestive_threshold == pytest.approx(0.25)


def test_prepare_genomic_axis_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="positions must have length 2"):
        prepare_genomic_axis(
            np.asarray(["1", "2"]),
            np.asarray([1.0]),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        prepare_genomic_axis(
            np.asarray(["1"]),
            np.asarray([1.0]),
            chromosome_gap=np.inf,
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.inf, np.nan, 1.1])
def test_prepare_manhattan_data_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="significance_threshold"):
        prepare_manhattan_data(
            np.asarray(["s1"]),
            np.asarray(["1"]),
            np.asarray([1.0]),
            np.asarray([0.5]),
            significance_threshold=threshold,
        )
