"""Random-number-generator contracts for BayesB."""

from __future__ import annotations

import numpy as np

from pygapit.models.genomic_prediction import BayesB


def test_bayesb_is_reproducible_with_explicit_generator() -> None:
    data_rng = np.random.default_rng(20260904)
    genotype = data_rng.binomial(2, 0.35, size=(24, 12)).astype(np.float64)
    phenotype = genotype[:, :3] @ np.array([0.8, -0.5, 0.3])
    phenotype += data_rng.normal(0.0, 0.2, size=len(phenotype))
    original_phenotype = phenotype.copy()

    first = BayesB(
        phenotype,
        genotype,
        n_iter=30,
        burn_in=10,
        rng=np.random.default_rng(42),
    )
    second = BayesB(
        phenotype,
        genotype,
        n_iter=30,
        burn_in=10,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(phenotype, original_phenotype)
