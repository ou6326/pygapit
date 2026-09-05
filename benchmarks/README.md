# Performance baseline

`run_baseline.py` records a reproducible baseline for the main numerical
workflows without making performance claims or enforcing timing thresholds in
CI. It covers PCA, VanRaden kinship, GLM, MLM, MLMM, SUPER selection, FarmCPU,
and BLINK on one deterministic synthetic dataset.

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

The preprocessing benchmark also accepts `--marker-workspace-mib` for wide
PCA and VanRaden. On the development Windows machine (Python 3.12.14, NumPy
2.5.2; 500 individuals, 50,000 markers; one warm-up, three repetitions), PCA
before batching measured 0.198 s / 197.36 MiB traced peak. The equivalent
32 MiB batched path measured 0.227 s / 21.02 MiB, while a 1 MiB budget measured
0.304 s / 6.39 MiB. These workload-specific values document the memory/runtime
trade-off and are not CI thresholds.

## Hotspot profiles

After recording a baseline, use deterministic profiler scenarios to separate
marker-heavy iterative work from sample-heavy mixed-model work:

```powershell
pixi run -e full python benchmarks/profile_hotspots.py --scenario marker-heavy --models farmcpu blink
pixi run -e full python benchmarks/profile_hotspots.py --scenario sample-heavy --models mlm mlmm cblup super
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

## RR-BLUP dimension-adaptive solve

```powershell
pixi run -e full python -m benchmarks.benchmark_rrblup --output benchmarks/results/rrblup.json
```

The benchmark checks numerical equivalence before comparing a centered,
fixed-penalty marker-space reference with the adaptive production solve.
It includes centering and effects, but excludes REML and cross-validation;
the reference corrects the legacy intercept semantics to isolate solver cost.
Memory is measured separately with tracemalloc, not whole-process RSS.

On the development Windows machine (Python 3.12.14, NumPy 2.5.2; 200 samples,
2,000 markers; one warm-up, three repetitions), the 2026-09-05 run measured
median 0.869 s / 94.61 MiB traced peak for the reference and 0.145 s / 9.25 MiB
for the adaptive path. These are workload-specific observations, not an
end-to-end speed claim or a CI threshold. The generated report records
environment thread overrides; no project BLAS setting is changed.
