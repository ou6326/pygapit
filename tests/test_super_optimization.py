"""Regression tests for low-rank SUPER selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.testing as nt
import pytest

from pygapit._typing import FloatMatrix, FloatVector, IntVector
from pygapit.gs.blup import SUPERSelectionResult, select_super_qtns
from pygapit.stats.emma import emma_remle, prepare_emma_factor_spectrum
from pygapit.stats.kinship import vanraden_factor, vanraden_kinship


def _reference_super_selection(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    chromosomes: IntVector,
    positions: FloatVector,
    p_values: FloatVector,
    candidate_counts: Sequence[int],
) -> SUPERSelectionResult:
    representatives: dict[tuple[int, int], int] = {}
    for marker_value in np.flatnonzero(np.isfinite(p_values)):
        marker = marker_value.item()
        key = (chromosomes[marker], int(np.floor(positions[marker] / 10_000)))
        current = representatives.get(key)
        if current is None or p_values[marker] < p_values[current]:
            representatives[key] = marker
    ranked = np.asarray(tuple(representatives.values()), dtype=np.int_)
    ranked = ranked[np.lexsort((ranked, p_values[ranked]))]
    counts = np.unique(np.minimum(np.asarray(candidate_counts), len(ranked)))

    fitted_counts: list[int] = []
    fitted_reml: list[float] = []
    fitted_qtns: list[IntVector] = []
    for count in counts:
        qtns = ranked[:count]
        qtns = qtns[np.var(GD[:, qtns], axis=0) > 0.0]
        if len(qtns) == 0:
            continue
        fit = emma_remle(y, X0, vanraden_kinship(GD[:, qtns]))
        fitted_counts.append(int(count))
        fitted_reml.append(fit.reml)
        fitted_qtns.append(np.sort(qtns))

    best = int(np.argmax(fitted_reml))
    return SUPERSelectionResult(
        qtn_indices=fitted_qtns[best],
        candidate_counts=np.asarray(fitted_counts, dtype=np.int_),
        reml=np.asarray(fitted_reml, dtype=np.float64),
    )


def _super_inputs() -> tuple[
    FloatVector,
    FloatMatrix,
    FloatMatrix,
    IntVector,
    FloatVector,
    FloatVector,
]:
    rng = np.random.default_rng(20260904)
    n_individuals = 80
    n_markers = 60
    genotype = rng.binomial(2, 0.35, size=(n_individuals, n_markers)).astype(np.float64)
    phenotype = genotype[:, 4] * 0.7 - genotype[:, 23] * 0.5
    phenotype += rng.normal(0.0, 0.8, size=n_individuals)
    design = np.column_stack([
        np.ones(n_individuals),
        np.linspace(-1.0, 1.0, n_individuals),
    ])
    chromosomes = np.arange(n_markers, dtype=np.int64) % 5 + 1
    positions = np.arange(n_markers, dtype=np.float64) * 10_000.0
    p_values = rng.uniform(0.0, 1.0, size=n_markers)
    return phenotype, design, genotype, chromosomes, positions, p_values


def test_factor_spectrum_matches_dense_vanraden_reml() -> None:
    phenotype, design, genotype, *_ = _super_inputs()
    factor = vanraden_factor(genotype[:, :15])
    kinship = vanraden_kinship(genotype[:, :15])
    nt.assert_allclose(factor @ factor.T, kinship, rtol=1e-14, atol=1e-14)

    expected = emma_remle(phenotype, design, kinship)
    spectrum = prepare_emma_factor_spectrum(factor, design)
    actual = emma_remle(phenotype, design, kinship, spectrum=spectrum)

    assert actual.reml == pytest.approx(expected.reml, rel=1e-10, abs=1e-11)
    assert actual.delta == pytest.approx(expected.delta, rel=1e-9)
    assert actual.vg == pytest.approx(expected.vg, rel=1e-9)
    assert actual.ve == pytest.approx(expected.ve, rel=1e-9)


def test_super_selection_matches_dense_reference_and_avoids_sample_eigh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phenotype, design, genotype, chromosomes, positions, p_values = _super_inputs()
    candidate_counts = (5, 10, 20)
    expected = _reference_super_selection(
        phenotype,
        design,
        genotype,
        chromosomes,
        positions,
        p_values,
        candidate_counts,
    )

    original_eigh = np.linalg.eigh
    dimensions: list[int] = []

    def counted_eigh(matrix: FloatMatrix) -> tuple[FloatVector, FloatMatrix]:
        dimensions.append(matrix.shape[0])
        return original_eigh(matrix)

    monkeypatch.setattr(np.linalg, "eigh", counted_eigh)
    actual = select_super_qtns(
        phenotype,
        design,
        genotype,
        chromosomes,
        positions,
        p_values,
        candidate_counts=candidate_counts,
    )

    nt.assert_array_equal(actual.qtn_indices, expected.qtn_indices)
    nt.assert_array_equal(actual.candidate_counts, expected.candidate_counts)
    nt.assert_allclose(actual.reml, expected.reml, rtol=1e-10, atol=1e-11)
    assert dimensions
    assert max(dimensions) <= max(candidate_counts)
