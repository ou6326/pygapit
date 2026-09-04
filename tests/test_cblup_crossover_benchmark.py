"""Correctness checks for the manual cBLUP eigensolver benchmark."""

from __future__ import annotations

import numpy as np

from benchmarks.benchmark_cblup_eigensolvers import (
    _benchmark_case,
    _group_space_decomposition,
    _make_case,
    _observation_space_decomposition,
    _subspace_residual,
)


def test_cblup_benchmark_decompositions_match_for_full_rank_kinship() -> None:
    incidence, kinship, fixed_design = _make_case(
        n_individuals=40,
        n_groups=16,
        kinship_rank=16,
        n_fixed_effects=3,
        seed=20260904,
    )

    group = _group_space_decomposition(incidence, kinship, fixed_design)
    observation = _observation_space_decomposition(
        incidence,
        kinship,
        fixed_design,
    )

    assert group is not None
    np.testing.assert_allclose(group.values, observation.values)
    assert _subspace_residual(group.basis, observation.basis) < 1e-10


def test_cblup_benchmark_marks_insufficient_kinship_rank() -> None:
    measurement = _benchmark_case(
        n_individuals=30,
        group_ratio=0.5,
        kinship_rank_fraction=0.5,
        n_fixed_effects=3,
        seed=20260904,
        warmups=0,
        repeats=1,
    )

    assert measurement.group_space_available is False
    assert measurement.faster_path == "observation-required"
