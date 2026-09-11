"""Benchmark exact PCA around the sample/marker-space crossover."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from benchmarks.run_baseline import BenchmarkResult, _benchmark, _make_data
from pygapit._resources import (
    iter_marker_slices,
    validate_marker_workspace_mib,
)
from pygapit._typing import FloatMatrix, FloatVector, IntVector
from pygapit.io.storage import GenotypeStore, GenotypeView, as_genotype_store
from pygapit.stats.pca import PCAResult, compute_pca


@dataclass(frozen=True, slots=True)
class PCAComparison:
    individuals: int
    markers: int
    marker_to_sample_ratio: float
    centered_matrix_mib: float
    marker_gram_mib: float
    production: BenchmarkResult
    sample_batched: BenchmarkResult | None


def _sample_batch_size(marker_count: int, workspace_mib: float) -> int:
    target_bytes = int(validate_marker_workspace_mib(workspace_mib) * 1024**2)
    return max(1, target_bytes // (marker_count * np.dtype(np.float64).itemsize))


def _sample_batched_tall_pca(
    GD: FloatMatrix | GenotypeStore,
    n_components: int,
    maf_filter: float,
    *,
    marker_workspace_mib: float,
) -> PCAResult:
    """Benchmark candidate that never retains the complete centered matrix."""
    genotype = as_genotype_store(GD)
    n, total_marker_count = genotype.shape
    frequencies = np.empty(total_marker_count, dtype=np.float64)
    for marker_slice in iter_marker_slices(
        n,
        total_marker_count,
        marker_workspace_mib,
    ):
        block = genotype.read_markers(marker_slice)
        frequencies[marker_slice] = np.sum(block, axis=0) / (2.0 * n)

    maf = np.minimum(frequencies, 1.0 - frequencies)
    valid = maf >= maf_filter
    if np.sum(valid) < n_components:
        valid = maf > 0.0
    marker_indices: IntVector = np.flatnonzero(valid)
    marker_count = len(marker_indices)
    if not 0 < marker_count < n:
        raise ValueError("sample-batched candidate requires 0 < markers < samples")

    k = min(n_components, marker_count)
    filtered = GenotypeView(genotype, marker_indices=marker_indices)
    means = 2.0 * frequencies[marker_indices]
    batch_size = min(n, _sample_batch_size(marker_count, marker_workspace_mib))
    gram: FloatMatrix = np.zeros((marker_count, marker_count), dtype=np.float64)
    total_sum = np.float64(0.0)
    for start in range(0, n, batch_size):
        sample_slice = slice(start, min(start + batch_size, n))
        centered = np.array(
            filtered.read_markers(slice(None), sample_slice),
            dtype=np.float64,
            copy=True,
        )
        centered -= means
        gram += centered.T @ centered
        batch_sum: np.float64 = np.einsum("ij,ij->", centered, centered)
        total_sum += batch_sum

    gram_values, loadings = scipy_eigh(
        gram,
        subset_by_index=(marker_count - k, marker_count - 1),
        check_finite=False,
    )
    gram_values = np.maximum(gram_values[::-1], 0.0)
    loadings = loadings[:, ::-1]
    scores = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, batch_size):
        sample_slice = slice(start, min(start + batch_size, n))
        centered = np.array(
            filtered.read_markers(slice(None), sample_slice),
            dtype=np.float64,
            copy=True,
        )
        centered -= means
        scores[sample_slice] = centered @ loadings

    eigenvalues: FloatVector = gram_values / (n - 1)
    total_variance = total_sum / (n - 1)
    var_explained: FloatVector = eigenvalues / total_variance
    return PCAResult(
        scores=scores,
        loadings=loadings,
        var_explained=var_explained,
        eigenvalues=eigenvalues,
    )


def _verify_equivalence(reference: PCAResult, candidate: PCAResult) -> None:
    np.testing.assert_allclose(
        candidate.eigenvalues,
        reference.eigenvalues,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        candidate.var_explained,
        reference.var_explained,
        rtol=1e-10,
        atol=1e-12,
    )
    for component in range(reference.scores.shape[1]):
        alignment = candidate.scores[:, component] @ reference.scores[:, component]
        sign = np.sign(alignment) or 1.0
        np.testing.assert_allclose(
            candidate.scores[:, component],
            sign * reference.scores[:, component],
            rtol=1e-9,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            candidate.loadings[:, component],
            sign * reference.loadings[:, component],
            rtol=1e-9,
            atol=1e-10,
        )


def run_pca_crossover_benchmark(
    *,
    n_individuals: int,
    marker_ratios: list[float],
    n_components: int,
    seed: int,
    warmups: int,
    repeats: int,
    marker_workspace_mib: float,
) -> dict[str, object]:
    """Compare production PCA with an exact sample-batched tall candidate."""
    if n_individuals < 10:
        raise ValueError("individuals must be at least 10")
    if not marker_ratios or any(ratio <= 0.0 for ratio in marker_ratios):
        raise ValueError("marker ratios must be positive")
    if n_components < 1:
        raise ValueError("components must be positive")
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats positive")
    marker_workspace_mib = validate_marker_workspace_mib(marker_workspace_mib)

    comparisons: list[PCAComparison] = []
    for index, ratio in enumerate(marker_ratios):
        marker_count = max(10, round(n_individuals * ratio))
        genotype, _phenotype, _chromosomes, _positions = _make_data(
            n_individuals,
            marker_count,
            seed + index,
        )
        store = as_genotype_store(genotype)

        def production(current_store: GenotypeStore = store) -> PCAResult:
            return compute_pca(
                current_store,
                n_components=n_components,
                marker_workspace_mib=marker_workspace_mib,
            )

        reference = production()
        candidate_result: BenchmarkResult | None = None
        if marker_count < n_individuals:

            def candidate(current_store: GenotypeStore = store) -> PCAResult:
                return _sample_batched_tall_pca(
                    current_store,
                    n_components,
                    0.05,
                    marker_workspace_mib=marker_workspace_mib,
                )

            _verify_equivalence(reference, candidate())
            candidate_result = _benchmark(
                "sample_batched",
                candidate,
                warmups=warmups,
                repeats=repeats,
            )

        comparisons.append(
            PCAComparison(
                individuals=n_individuals,
                markers=marker_count,
                marker_to_sample_ratio=marker_count / n_individuals,
                centered_matrix_mib=(
                    n_individuals
                    * marker_count
                    * np.dtype(np.float64).itemsize
                    / 1024**2
                ),
                marker_gram_mib=(
                    marker_count**2 * np.dtype(np.float64).itemsize / 1024**2
                ),
                production=_benchmark(
                    "production",
                    production,
                    warmups=warmups,
                    repeats=repeats,
                ),
                sample_batched=candidate_result,
            )
        )

    return {
        "workload": {
            "individuals": n_individuals,
            "marker_ratios": marker_ratios,
            "components": n_components,
            "seed": seed,
            "warmups": warmups,
            "repeats": repeats,
            "marker_workspace_mib": marker_workspace_mib,
        },
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "memory_note": (
            "traced_peak_mib is measured with Python tracemalloc and may exclude "
            "native BLAS allocations; the genotype input is allocated before tracing"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", type=int, default=1_000)
    parser.add_argument(
        "--marker-ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
    )
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--marker-workspace-mib",
        type=float,
        default=4.0,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_pca_crossover_benchmark(
        n_individuals=args.individuals,
        marker_ratios=args.marker_ratios,
        n_components=args.components,
        seed=args.seed,
        warmups=args.warmups,
        repeats=args.repeats,
        marker_workspace_mib=args.marker_workspace_mib,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
