"""Regression tests for optimized MLMM marker scans."""

from __future__ import annotations

import numpy as np
import numpy.testing as nt
import pytest
from scipy.stats import f as f_dist

from pygapit._typing import FloatMatrix, FloatVector
from pygapit.gwas.mlmm import _conditioned_marker_scan, _normalize_kinship
from pygapit.stats.emma import GWASResult, emma_remle
from pygapit.stats.kinship import vanraden_kinship


def _reference_conditioned_marker_scan(
    y: FloatVector,
    X0: FloatMatrix,
    GD: FloatMatrix,
    K: FloatMatrix,
    cofactors: list[int],
    ngrids: int,
) -> GWASResult:
    """Previous explicit-projection, marker-at-a-time implementation."""
    design: FloatMatrix = (
        np.column_stack([X0] + [GD[:, index] for index in cofactors])
        if cofactors
        else X0
    )
    remle = emma_remle(y, design, K, ngrids=ngrids)
    covariance = remle.vg * K + remle.ve * np.eye(len(y))
    cholesky = np.linalg.cholesky(covariance)
    transformed_y = np.linalg.solve(cholesky, y)
    transformed_design = np.linalg.solve(cholesky, design)
    design_beta, *_ = np.linalg.lstsq(transformed_design, transformed_y, rcond=None)
    residual = transformed_y - transformed_design @ design_beta
    projection = np.eye(len(y)) - transformed_design @ np.linalg.pinv(
        transformed_design
    )
    transformed_genotypes = projection @ np.linalg.solve(cholesky, GD)
    null_rss = np.sum(residual**2)
    degrees_of_freedom = len(y) - design.shape[1] - 1

    p_values = np.ones(GD.shape[1])
    effects = np.full(GD.shape[1], np.nan)
    standard_errors = np.full(GD.shape[1], np.nan)
    statistics = np.full(GD.shape[1], np.nan)
    for marker in range(GD.shape[1]):
        if marker in cofactors:
            continue
        genotype = transformed_genotypes[:, marker]
        genotype_sum_squares = np.sum(genotype**2)
        if genotype_sum_squares < 1e-12:
            continue
        effect = np.sum(genotype * residual) / genotype_sum_squares
        marker_rss = np.sum((residual - genotype * effect) ** 2)
        f_statistic = np.maximum(
            (null_rss / marker_rss - 1.0) * degrees_of_freedom,
            0.0,
        )
        statistic = np.sign(effect) * np.sqrt(f_statistic)
        p_values[marker] = f_dist.sf(f_statistic, 1, degrees_of_freedom)
        effects[marker] = effect
        statistics[marker] = statistic
        if f_statistic > 0.0:
            standard_errors[marker] = abs(effect) / np.sqrt(f_statistic)

    return GWASResult(
        p_values=p_values,
        effects=effects,
        se=standard_errors,
        stats=statistics,
        vg=remle.vg,
        ve=remle.ve,
        h2=remle.h2,
    )


@pytest.mark.parametrize("cofactors", [[], [1, 4]])
def test_conditioned_marker_scan_matches_explicit_projection(
    cofactors: list[int],
) -> None:
    rng = np.random.default_rng(20260904)
    n_individuals = 40
    genotype = rng.binomial(2, 0.35, size=(n_individuals, 16)).astype(np.float64)
    genotype[:, 0] = 1.0
    phenotype = genotype[:, 3] * 0.8 - genotype[:, 9] * 0.4
    phenotype += rng.normal(0.0, 0.75, size=n_individuals)
    design = np.column_stack([
        np.ones(n_individuals),
        np.linspace(-1.0, 1.0, n_individuals),
    ])
    kinship = _normalize_kinship(vanraden_kinship(genotype))

    expected = _reference_conditioned_marker_scan(
        phenotype,
        design,
        genotype,
        kinship,
        cofactors,
        ngrids=30,
    )
    actual = _conditioned_marker_scan(
        phenotype,
        design,
        genotype,
        kinship,
        cofactors,
        ngrids=30,
    )

    nt.assert_allclose(actual.p_values, expected.p_values, rtol=1e-9, atol=1e-12)
    nt.assert_allclose(actual.effects, expected.effects, rtol=1e-9, atol=1e-12)
    nt.assert_allclose(actual.se, expected.se, rtol=1e-9, atol=1e-12)
    nt.assert_allclose(actual.stats, expected.stats, rtol=1e-9, atol=1e-12)
    assert actual.vg == pytest.approx(expected.vg)
    assert actual.ve == pytest.approx(expected.ve)
    assert actual.h2 == pytest.approx(expected.h2)
