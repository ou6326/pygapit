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

## Hotspot profiles

After recording a baseline, use deterministic profiler scenarios to separate
marker-heavy iterative work from sample-heavy mixed-model work:

```powershell
pixi run -e full python benchmarks/profile_hotspots.py --scenario marker-heavy --models farmcpu blink
pixi run -e full python benchmarks/profile_hotspots.py --scenario sample-heavy --models mlm mlmm cblup
```

Pass `--output-dir benchmarks/results/profiles` to retain standard `.prof`
files for `python -m pstats` or another compatible viewer. Compare profiles
only on the same machine and thread configuration used for the baseline.

## cBLUP eigensolver crossover

`benchmark_cblup_eigensolvers.py` compares the group-space and
observation-space incidence decompositions across sample counts, compression
ratios from 0.2 through the uncompressed case, and kinship ranks. Every
full-rank timing is preceded by an eigenvalue and eigenspace-equivalence check.
Rank-deficient cases record when group space is mathematically unavailable and
the observation-space fallback is required.

Run the default multi-scale workload and retain its JSON report:

```powershell
pixi run -e full python benchmarks/benchmark_cblup_eigensolvers.py --output benchmarks/results/cblup-eigensolver-crossover.json
```

For a quick validation run:

```powershell
pixi run -e full python benchmarks/benchmark_cblup_eigensolvers.py --individuals 120 --group-ratios 0.25 0.5 --kinship-rank-fractions 1.0 0.5 --warmups 0 --repeats 1
```

An `observation_over_group` value above one favors group space. Use results
only to revise the production crossover heuristic when the trend is stable
across relevant sample sizes and the same BLAS/thread configuration.
The checked-in heuristic uses group space through `groups / individuals = 0.8`.
With the centered-kinship contrast factorization this remains consistently
faster across the default sample sizes; 0.9 is deliberately excluded because
the two paths are effectively tied there.
