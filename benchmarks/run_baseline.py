"""Reproducible performance baseline for pyGAPIT's main numerical workflows."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from pygapit._typing import FloatMatrix, FloatVector, IntVector
from pygapit.gapit import GAPIT
from pygapit.gs.blup import select_super_qtns
from pygapit.gwas.blink import blink_gwas
from pygapit.gwas.farmcpu import farmcpu_gwas
from pygapit.gwas.glm import glm_gwas
from pygapit.gwas.mlm import mlm_gwas
from pygapit.gwas.mlmm import mlmm_gwas
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import build_covariate_matrix, compute_pca


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    traced_peak_mib: float


def _make_data(
    n_individuals: int,
    n_markers: int,
    seed: int,
) -> tuple[FloatMatrix, FloatVector, IntVector, FloatVector]:
    """Create deterministic polymorphic markers and a quantitative trait."""
    if n_individuals < 10:
        raise ValueError("individuals must be at least 10")
    if n_markers < 10:
        raise ValueError("markers must be at least 10")

    rng = np.random.default_rng(seed)
    allele_frequency = rng.uniform(0.1, 0.5, size=n_markers)
    genotype = rng.binomial(2, allele_frequency, size=(n_individuals, n_markers))
    genotype = genotype.astype(np.float64)

    causal_count = min(8, n_markers)
    causal_markers = np.linspace(0, n_markers - 1, causal_count, dtype=int)
    effects = rng.normal(0.0, 1.0, size=causal_count)
    phenotype = genotype[:, causal_markers] @ effects
    phenotype += rng.normal(0.0, np.std(phenotype) * 0.75, size=n_individuals)

    chromosomes = np.arange(n_markers, dtype=np.int64) % 10 + 1
    positions = (np.arange(n_markers, dtype=np.float64) // 10 + 1) * 10_000.0
    return genotype, phenotype, chromosomes, positions


def _make_pipeline_frames(
    genotype: FloatMatrix,
    phenotype: FloatVector,
    chromosomes: IntVector,
    positions: FloatVector,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build single- and multi-trait pipeline inputs from one workload."""
    n_individuals, n_markers = genotype.shape
    taxa = np.asarray([f"T{i:05d}" for i in range(n_individuals)], dtype=str)
    marker_names = np.asarray([f"SNP{i:06d}" for i in range(n_markers)], dtype=str)
    phenotype_frame = pd.DataFrame({"Taxa": taxa, "trait": phenotype})
    multitrait_frame = pd.DataFrame({
        "Taxa": taxa,
        "trait_1": phenotype,
        "trait_2": 0.75 * phenotype + 0.25 * genotype[:, 0],
        "trait_3": -0.5 * phenotype + 0.5 * genotype[:, 1],
        "trait_4": phenotype + 0.1 * genotype[:, 2],
    })
    genotype_frame = pd.DataFrame(genotype, columns=marker_names)
    genotype_frame.insert(0, "Taxa", taxa)
    marker_map = pd.DataFrame({
        "SNP": marker_names,
        "Chromosome": chromosomes,
        "Position": positions,
    })
    return phenotype_frame, multitrait_frame, genotype_frame, marker_map


def _measure_time(
    operation: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmups):
        operation()

    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - started)
    return timings


def _measure_peak_memory(operation: Callable[[], object]) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        operation()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024.0**2)


def _benchmark(
    name: str,
    operation: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> BenchmarkResult:
    timings = np.asarray(
        _measure_time(operation, warmups=warmups, repeats=repeats),
        dtype=np.float64,
    )
    return BenchmarkResult(
        name=name,
        median_seconds=np.median(timings).item(),
        minimum_seconds=np.min(timings).item(),
        maximum_seconds=np.max(timings).item(),
        traced_peak_mib=_measure_peak_memory(operation),
    )


def run_baseline(
    *,
    n_individuals: int,
    n_markers: int,
    seed: int,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    """Run the fixed workload and return a JSON-serializable report."""
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    genotype, phenotype, chromosomes, positions = _make_data(
        n_individuals,
        n_markers,
        seed,
    )
    phenotype_frame, multitrait_frame, genotype_frame, marker_map = (
        _make_pipeline_frames(
            genotype,
            phenotype,
            chromosomes,
            positions,
        )
    )
    pca = compute_pca(genotype, n_components=3)
    design = build_covariate_matrix(pca, n_pcs=3)
    kinship = vanraden_kinship(genotype)
    super_p_values = mlm_gwas(phenotype, design, genotype, kinship).p_values
    candidate_threshold = 1.0 / n_markers

    def run_pipeline(
        model: str | list[str],
        traits: pd.DataFrame = phenotype_frame,
    ) -> object:
        with contextlib.redirect_stdout(io.StringIO()):
            return GAPIT(
                Y=traits,
                GD=genotype_frame,
                GM=marker_map,
                model=model,
                file_output=False,
            )

    operations: Sequence[tuple[str, Callable[[], object]]] = (
        ("pca", lambda: compute_pca(genotype, n_components=3)),
        ("vanraden_kinship", lambda: vanraden_kinship(genotype)),
        ("glm", lambda: glm_gwas(phenotype, design, genotype)),
        ("mlm", lambda: mlm_gwas(phenotype, design, genotype, kinship)),
        (
            "mlmm",
            lambda: mlmm_gwas(
                phenotype,
                design,
                genotype,
                kinship,
                max_steps=5,
            ),
        ),
        (
            "super_selection",
            lambda: select_super_qtns(
                phenotype,
                design,
                genotype,
                chromosomes,
                positions,
                super_p_values,
            ),
        ),
        (
            "farmcpu",
            lambda: farmcpu_gwas(
                phenotype,
                design,
                genotype,
                chromosomes,
                positions,
                max_iterations=5,
                p_threshold=candidate_threshold,
            ),
        ),
        (
            "blink",
            lambda: blink_gwas(
                phenotype,
                design,
                genotype,
                max_iterations=5,
                p_threshold=candidate_threshold,
            ),
        ),
        (
            "pipeline_glm",
            lambda: run_pipeline("GLM"),
        ),
        (
            "pipeline_mlm",
            lambda: run_pipeline("MLM"),
        ),
        (
            "pipeline_multitrait_glm",
            lambda: run_pipeline("GLM", multitrait_frame),
        ),
        (
            "pipeline_multitrait_mlm",
            lambda: run_pipeline("MLM", multitrait_frame),
        ),
        (
            "pipeline_multitrait_gblup",
            lambda: run_pipeline("GBLUP", multitrait_frame),
        ),
        (
            "pipeline_multitrait_cblup",
            lambda: run_pipeline("CBLUP", multitrait_frame),
        ),
        (
            "pipeline_multitrait_sblup",
            lambda: run_pipeline("SBLUP", multitrait_frame),
        ),
        (
            "pipeline_multitrait_mlm_sblup",
            lambda: run_pipeline(["MLM", "SBLUP"], multitrait_frame),
        ),
    )
    results = [
        _benchmark(name, operation, warmups=warmups, repeats=repeats)
        for name, operation in operations
    ]
    return {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
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
            "individuals": n_individuals,
            "markers": n_markers,
            "seed": seed,
            "warmups": warmups,
            "repeats": repeats,
            "pca_components": 3,
            "iterative_max_iterations": 5,
            "multitrait_count": 4,
        },
        "measurements": [asdict(result) for result in results],
        "memory_note": (
            "traced_peak_mib is measured with Python tracemalloc and may exclude "
            "native BLAS workspace allocations"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", type=int, default=200)
    parser.add_argument("--markers", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260901)
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
    report = run_baseline(
        n_individuals=args.individuals,
        n_markers=args.markers,
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
        print(f"Wrote benchmark report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
