# Performance baseline

`run_baseline.py` records a reproducible baseline for the main numerical
workflows without making performance claims or enforcing timing thresholds in
CI. It covers PCA, VanRaden kinship, GLM, MLM, FarmCPU, and BLINK on one
deterministic synthetic dataset.

Run the standard workload in the full pixi environment:

```powershell
pixi run -e full python benchmarks/run_baseline.py --output benchmarks/results/baseline.json
```

The default workload uses 200 individuals, 5,000 markers, one warm-up, and
three timed repetitions. Record the generated JSON together with the commit,
CPU, and thread configuration when comparing changes. For quick validation:

```powershell
pixi run -e full python benchmarks/run_baseline.py --individuals 60 --markers 200 --warmups 0 --repeats 1
```

Elapsed times are measured without memory tracing. Peak memory is measured in
a separate invocation using Python's `tracemalloc`; it can exclude memory used
internally by native BLAS libraries and should not be presented as whole-process
peak RSS. The script deliberately remains outside the regular pytest and CI
suites so noisy machine-dependent timings cannot fail correctness checks.
