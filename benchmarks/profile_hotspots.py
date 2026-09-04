"""Reproducible cProfile scenarios for pyGAPIT's core model paths."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from benchmarks.run_baseline import _make_data
from pygapit.gs.blup import cblup
from pygapit.gwas.blink import blink_gwas
from pygapit.gwas.farmcpu import farmcpu_gwas
from pygapit.gwas.mlm import mlm_gwas
from pygapit.gwas.mlmm import mlmm_gwas
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import build_covariate_matrix, compute_pca

_SCENARIOS = {
    "marker-heavy": (200, 5_000),
    "sample-heavy": (500, 1_000),
}
_MODEL_NAMES = ("mlm", "mlmm", "cblup", "farmcpu", "blink")


def profile_models(
    *,
    n_individuals: int,
    n_markers: int,
    seed: int,
    models: Sequence[str],
    max_iterations: int,
    limit: int,
    output_dir: Path | None,
) -> None:
    """Profile selected models on one deterministic synthetic workload."""
    genotype, phenotype, chromosomes, positions = _make_data(
        n_individuals,
        n_markers,
        seed,
    )
    pca = compute_pca(genotype, n_components=3)
    design = build_covariate_matrix(pca, n_pcs=3)
    kinship = vanraden_kinship(genotype)
    threshold = 1.0 / n_markers
    operations: dict[str, Callable[[], object]] = {
        "mlm": lambda: mlm_gwas(phenotype, design, genotype, kinship),
        "mlmm": lambda: mlmm_gwas(
            phenotype,
            design,
            genotype,
            kinship,
            max_steps=max_iterations,
        ),
        "cblup": lambda: cblup(phenotype, design, genotype),
        "farmcpu": lambda: farmcpu_gwas(
            phenotype,
            design,
            genotype,
            chromosomes,
            positions,
            max_iterations=max_iterations,
            p_threshold=threshold,
        ),
        "blink": lambda: blink_gwas(
            phenotype,
            design,
            genotype,
            max_iterations=max_iterations,
            p_threshold=threshold,
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        print(
            f"PROFILE {model} | individuals={n_individuals} markers={n_markers}",
            flush=True,
        )
        profiler = cProfile.Profile()
        profiler.runcall(operations[model])
        if output_dir is not None:
            profiler.dump_stats(
                output_dir / f"{model}-{n_individuals}x{n_markers}.prof"
            )
        pstats.Stats(profiler, stream=sys.stdout).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(limit)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=_SCENARIOS, default="marker-heavy")
    parser.add_argument("--individuals", type=int)
    parser.add_argument("--markers", type=int)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--models", nargs="+", choices=_MODEL_NAMES, default=_MODEL_NAMES
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    default_individuals, default_markers = _SCENARIOS[args.scenario]
    profile_models(
        n_individuals=args.individuals or default_individuals,
        n_markers=args.markers or default_markers,
        seed=args.seed,
        models=args.models,
        max_iterations=args.max_iterations,
        limit=args.limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
