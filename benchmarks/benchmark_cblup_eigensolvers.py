"""Benchmark cBLUP incidence eigensolvers across compression and rank regimes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh as scipy_eigh

from pygapit._typing import FloatMatrix, FloatVector


@dataclass(frozen=True, slots=True)
class Decomposition:
    values: FloatVector
    basis: FloatMatrix


@dataclass(frozen=True, slots=True)
class CrossoverResult:
    individuals: int
    groups: int
    group_ratio: float
    kinship_rank: int
    kinship_rank_fraction: float
    random_rank: int
    group_space_available: bool
    group_median_seconds: float | None
    observation_median_seconds: float
    observation_over_group: float | None
    faster_path: str
    eigenvalue_max_abs_error: float | None
    subspace_residual: float | None


def _make_case(
    *,
    n_individuals: int,
    n_groups: int,
    kinship_rank: int,
    n_fixed_effects: int,
    seed: int,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    if not 0 < n_fixed_effects < n_groups <= n_individuals:
        raise ValueError("require 0 < fixed effects < groups <= individuals")
    if not 0 < kinship_rank <= n_groups:
        raise ValueError("kinship rank must be between one and the group count")

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, n_individuals, n_groups, kinship_rank])
    )
    labels = np.arange(n_individuals) * n_groups // n_individuals
    incidence = np.zeros((n_individuals, n_groups), dtype=np.float64)
    incidence[np.arange(n_individuals), labels] = 1.0

    fixed_columns = np.column_stack([
        np.ones(n_individuals, dtype=np.float64),
        rng.normal(size=(n_individuals, n_fixed_effects - 1)),
    ])
    fixed_design, _ = np.linalg.qr(fixed_columns, mode="reduced")

    kinship_basis, _ = np.linalg.qr(
        rng.normal(size=(n_groups, kinship_rank)),
        mode="reduced",
    )
    spectrum = np.geomspace(1.0, 0.1, kinship_rank)
    kinship = (kinship_basis * spectrum) @ kinship_basis.T
    kinship = (kinship + kinship.T) / 2.0
    return incidence, kinship, fixed_design


def _residualize_incidence(
    incidence: FloatMatrix,
    fixed_design: FloatMatrix,
) -> FloatMatrix:
    projection_coefficients, *_ = np.linalg.lstsq(
        fixed_design,
        incidence,
        rcond=None,
    )
    return incidence - fixed_design @ projection_coefficients


def _group_space_decomposition(
    incidence: FloatMatrix,
    kinship: FloatMatrix,
    fixed_design: FloatMatrix,
) -> Decomposition | None:
    """Run the group-space branch used by ``_eigen_R_w_Z`` when viable."""
    n_individuals, n_groups = incidence.shape
    random_rank = n_groups - fixed_design.shape[1]
    residualized = _residualize_incidence(incidence, fixed_design)

    kinship_values, kinship_vectors = np.linalg.eigh(kinship)
    kinship_scale = np.max([np.max(np.abs(kinship_values)), 1.0])
    tolerance = np.finfo(np.float64).eps * n_groups * kinship_scale
    positive = kinship_values > tolerance
    if np.count_nonzero(positive) < random_rank:
        return None

    kinship_factor = kinship_vectors[:, positive] * np.sqrt(kinship_values[positive])
    observation_factor = residualized @ kinship_factor
    group_covariance = observation_factor.T @ observation_factor
    group_covariance = (group_covariance + group_covariance.T) / 2.0
    group_values, group_vectors = np.linalg.eigh(group_covariance)
    values = group_values[::-1][:random_rank]
    value_scale = np.max([np.max(np.abs(values)), 1.0])
    value_tolerance = np.finfo(np.float64).eps * n_individuals * value_scale
    if np.min(values) <= value_tolerance:
        return None

    vectors = group_vectors[:, ::-1][:, :random_rank]
    basis = observation_factor @ vectors
    basis /= np.sqrt(values)[np.newaxis, :]
    return Decomposition(values=values, basis=basis)


def _observation_space_decomposition(
    incidence: FloatMatrix,
    kinship: FloatMatrix,
    fixed_design: FloatMatrix,
) -> Decomposition:
    """Run the observation-space fallback used by ``_eigen_R_w_Z``."""
    n_individuals, n_groups = incidence.shape
    random_rank = n_groups - fixed_design.shape[1]
    residualized = _residualize_incidence(incidence, fixed_design)
    covariance = residualized @ kinship @ residualized.T
    covariance = (covariance + covariance.T) / 2.0
    if random_rank * 2 < n_individuals:
        values, basis = scipy_eigh(
            covariance,
            subset_by_index=(n_individuals - random_rank, n_individuals - 1),
        )
    else:
        values, basis = np.linalg.eigh(covariance)
    return Decomposition(
        values=values[::-1][:random_rank],
        basis=basis[:, ::-1][:, :random_rank],
    )


def _median_time(
    operation: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> float:
    for _ in range(warmups):
        operation()
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - started)
    return np.median(np.asarray(timings, dtype=np.float64)).item()


def _subspace_residual(
    candidate_basis: FloatMatrix,
    reference_basis: FloatMatrix,
) -> float:
    residual = candidate_basis - reference_basis @ (reference_basis.T @ candidate_basis)
    return np.linalg.norm(residual, ord="fro").item() / np.sqrt(
        candidate_basis.shape[1]
    )


def _benchmark_case(
    *,
    n_individuals: int,
    group_ratio: float,
    kinship_rank_fraction: float,
    n_fixed_effects: int,
    seed: int,
    warmups: int,
    repeats: int,
) -> CrossoverResult:
    n_groups = min(
        n_individuals,
        max(n_fixed_effects + 1, round(n_individuals * group_ratio)),
    )
    kinship_rank = min(
        n_groups,
        max(1, round(n_groups * kinship_rank_fraction)),
    )
    incidence, kinship, fixed_design = _make_case(
        n_individuals=n_individuals,
        n_groups=n_groups,
        kinship_rank=kinship_rank,
        n_fixed_effects=n_fixed_effects,
        seed=seed,
    )
    observation = _observation_space_decomposition(
        incidence,
        kinship,
        fixed_design,
    )
    group = _group_space_decomposition(incidence, kinship, fixed_design)
    observation_seconds = _median_time(
        lambda: _observation_space_decomposition(
            incidence,
            kinship,
            fixed_design,
        ),
        warmups=warmups,
        repeats=repeats,
    )
    if group is None:
        return CrossoverResult(
            individuals=n_individuals,
            groups=n_groups,
            group_ratio=n_groups / n_individuals,
            kinship_rank=kinship_rank,
            kinship_rank_fraction=kinship_rank / n_groups,
            random_rank=n_groups - n_fixed_effects,
            group_space_available=False,
            group_median_seconds=None,
            observation_median_seconds=observation_seconds,
            observation_over_group=None,
            faster_path="observation-required",
            eigenvalue_max_abs_error=None,
            subspace_residual=None,
        )

    np.testing.assert_allclose(group.values, observation.values, rtol=1e-8, atol=1e-9)
    eigenvalue_error = np.max(np.abs(group.values - observation.values)).item()
    subspace_error = _subspace_residual(group.basis, observation.basis)
    if subspace_error > 1e-8:
        raise AssertionError(
            f"eigensolver subspaces differ by {subspace_error:.3e} for "
            f"n={n_individuals}, groups={n_groups}, rank={kinship_rank}"
        )

    def run_group_space() -> Decomposition:
        result = _group_space_decomposition(incidence, kinship, fixed_design)
        if result is None:
            raise RuntimeError("validated group-space case became unavailable")
        return result

    group_seconds = _median_time(
        run_group_space,
        warmups=warmups,
        repeats=repeats,
    )
    return CrossoverResult(
        individuals=n_individuals,
        groups=n_groups,
        group_ratio=n_groups / n_individuals,
        kinship_rank=kinship_rank,
        kinship_rank_fraction=kinship_rank / n_groups,
        random_rank=n_groups - n_fixed_effects,
        group_space_available=True,
        group_median_seconds=group_seconds,
        observation_median_seconds=observation_seconds,
        observation_over_group=observation_seconds / group_seconds,
        faster_path="group" if group_seconds < observation_seconds else "observation",
        eigenvalue_max_abs_error=eigenvalue_error,
        subspace_residual=subspace_error,
    )


def run_crossover_benchmark(
    *,
    individual_counts: Sequence[int],
    group_ratios: Sequence[float],
    kinship_rank_fractions: Sequence[float],
    n_fixed_effects: int,
    seed: int,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    """Run all configured eigensolver cases and return a JSON report."""
    if any(count <= n_fixed_effects for count in individual_counts):
        raise ValueError("individual counts must exceed the fixed-effect count")
    if any(not 0.0 < ratio <= 1.0 for ratio in group_ratios):
        raise ValueError("group ratios must be in (0, 1]")
    if any(not 0.0 < fraction <= 1.0 for fraction in kinship_rank_fractions):
        raise ValueError("kinship rank fractions must be in (0, 1]")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    results = [
        _benchmark_case(
            n_individuals=n_individuals,
            group_ratio=group_ratio,
            kinship_rank_fraction=rank_fraction,
            n_fixed_effects=n_fixed_effects,
            seed=seed,
            warmups=warmups,
            repeats=repeats,
        )
        for n_individuals in individual_counts
        for rank_fraction in kinship_rank_fractions
        for group_ratio in group_ratios
    ]
    return {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_variables": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "workload": {
            "individual_counts": list(individual_counts),
            "group_ratios": list(group_ratios),
            "kinship_rank_fractions": list(kinship_rank_fractions),
            "fixed_effects": n_fixed_effects,
            "seed": seed,
            "warmups": warmups,
            "repeats": repeats,
        },
        "measurements": [asdict(result) for result in results],
        "interpretation": (
            "observation_over_group values above one favor group space; "
            "observation-required means numerical rank was insufficient for "
            "the group-space formulation"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument(
        "--group-ratios",
        type=float,
        nargs="+",
        default=[0.2, 0.35, 0.5, 0.6, 0.65, 0.8, 1.0],
    )
    parser.add_argument(
        "--kinship-rank-fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.5],
    )
    parser.add_argument("--fixed-effects", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise print the report to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_crossover_benchmark(
        individual_counts=args.individuals,
        group_ratios=args.group_ratios,
        kinship_rank_fractions=args.kinship_rank_fractions,
        n_fixed_effects=args.fixed_effects,
        seed=args.seed,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote crossover report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
