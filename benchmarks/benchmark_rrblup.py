"""Compare equivalent primal ridge and dimension-adaptive RR-BLUP solves."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.run_baseline import _benchmark, _make_data
from pygapit._typing import FloatVector
from pygapit.gs.validation import _ridge_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", type=int, default=200)
    parser.add_argument("--markers", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    genotype, y, _, _ = _make_data(args.individuals, args.markers, 20260905)
    penalty = 10.0

    def primal() -> FloatVector:
        z = genotype - genotype.mean(axis=0)
        effects = np.linalg.solve(
            z.T @ z + penalty * np.eye(z.shape[1]), z.T @ (y - y.mean())
        )
        return z @ effects

    def adaptive() -> FloatVector:
        return _ridge_fit(y, genotype, genotype, penalty)[0]

    np.testing.assert_allclose(primal(), adaptive(), rtol=1e-10, atol=1e-10)
    results = [
        asdict(_benchmark(name, operation, warmups=1, repeats=args.repeats))
        for name, operation in (("primal_reference", primal), ("adaptive", adaptive))
    ]
    report = {
        "individuals": args.individuals,
        "markers": args.markers,
        "seed": 20260905,
        "lambda": penalty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "threads": {
            name: os.environ.get(name)
            for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "measurements": results,
        "note": "Equivalent centered fixed-lambda solves; excludes CV and REML. traced_peak_mib is not process RSS.",
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
