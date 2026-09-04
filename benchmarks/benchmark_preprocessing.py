"""Benchmark genotype I/O, missing-value handling, PCA, and kinship."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.run_baseline import _benchmark, _make_data
from pygapit.io.formats import impute_missing, read_hapmap, read_numeric
from pygapit.stats.kinship import vanraden_kinship
from pygapit.stats.pca import compute_pca


def _numeric_inputs(
    genotype: np.ndarray,
    chromosomes: np.ndarray,
    positions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_individuals, n_markers = genotype.shape
    marker_names = [f"SNP{i:06d}" for i in range(n_markers)]
    frame = pd.DataFrame(genotype, columns=marker_names)
    frame.insert(0, "Taxa", [f"T{i:05d}" for i in range(n_individuals)])
    marker_map = pd.DataFrame({
        "SNP": marker_names,
        "Chromosome": chromosomes,
        "Position": positions,
    })
    return frame, marker_map


def _hapmap_frame(
    genotype: np.ndarray,
    chromosomes: np.ndarray,
    positions: np.ndarray,
) -> pd.DataFrame:
    n_individuals, n_markers = genotype.shape
    calls = np.asarray(["AA", "AT", "TT", "NN"])[
        np.where(np.isnan(genotype), 3, genotype).astype(np.intp)
    ].T
    metadata = pd.DataFrame({
        "rs": [f"SNP{i:06d}" for i in range(n_markers)],
        "alleles": "A/T",
        "chrom": chromosomes,
        "pos": positions,
        "strand": "+",
        "assembly": "NA",
        "center": "NA",
        "protLSID": "NA",
        "assayLSID": "NA",
        "panelLSID": "NA",
        "QCcode": "NA",
    })
    taxa = [f"T{i:05d}" for i in range(n_individuals)]
    return pd.concat([metadata, pd.DataFrame(calls, columns=taxa)], axis=1)


def run_preprocessing_benchmark(
    *,
    n_individuals: int,
    n_markers: int,
    missing_rate: float,
    seed: int,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    """Run a deterministic preprocessing workload."""
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be between zero and one")

    genotype, _phenotype, chromosomes, positions = _make_data(
        n_individuals,
        n_markers,
        seed,
    )
    rng = np.random.default_rng(seed + 1)
    genotype_missing = genotype.copy()
    genotype_missing[rng.random(genotype.shape) < missing_rate] = np.nan
    numeric_frame, marker_map = _numeric_inputs(
        genotype_missing,
        chromosomes,
        positions,
    )
    hapmap_frame = _hapmap_frame(genotype_missing, chromosomes, positions)

    with tempfile.TemporaryDirectory(prefix="pygapit-preprocessing-") as temp_dir:
        data_dir = Path(temp_dir)
        numeric_path = data_dir / "genotype.tsv"
        hapmap_path = data_dir / "genotype.hmp.txt"
        numeric_frame.to_csv(numeric_path, sep="\t", index=False)
        hapmap_frame.to_csv(hapmap_path, sep="\t", index=False)

        operations = (
            ("read_numeric", lambda: read_numeric(numeric_path, marker_map, "mean")),
            ("read_hapmap", lambda: read_hapmap(hapmap_path, impute_method="mean")),
            ("impute_middle", lambda: impute_missing(genotype_missing, "middle")),
            ("impute_mean", lambda: impute_missing(genotype_missing, "mean")),
            ("pca", lambda: compute_pca(genotype, n_components=3)),
            ("vanraden_kinship", lambda: vanraden_kinship(genotype)),
        )
        results = [
            _benchmark(name, operation, warmups=warmups, repeats=repeats)
            for name, operation in operations
        ]

    return {
        "workload": {
            "individuals": n_individuals,
            "markers": n_markers,
            "missing_rate": missing_rate,
            "seed": seed,
            "warmups": warmups,
            "repeats": repeats,
        },
        "measurements": [asdict(result) for result in results],
        "memory_note": (
            "traced_peak_mib is measured with Python tracemalloc and may exclude "
            "native parser and BLAS allocations"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", type=int, default=500)
    parser.add_argument("--markers", type=int, default=5_000)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_preprocessing_benchmark(
        n_individuals=args.individuals,
        n_markers=args.markers,
        missing_rate=args.missing_rate,
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


if __name__ == "__main__":
    main()
