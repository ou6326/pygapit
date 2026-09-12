# Performance baseline

`run_baseline.py` records a reproducible baseline for the main numerical
workflows without making performance claims or enforcing timing thresholds in
CI. It covers PCA, VanRaden kinship, GLM, MLM, MLMM, SUPER selection, FarmCPU,
and BLINK on one deterministic synthetic dataset.

Run the standard workload in the default Pixi environment:

```powershell
pixi run python benchmarks/run_baseline.py --output benchmarks/results/baseline.json
```

The default workload uses 200 individuals, 5,000 markers, one warm-up, and
three timed repetitions. Record the generated JSON together with the commit,
CPU, and thread configuration when comparing changes. For quick validation:

```powershell
pixi run python benchmarks/run_baseline.py --individuals 60 --markers 200 --warmups 0 --repeats 1
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

On the same machine, mean imputation of a 500 by 20,000 matrix with 5% missing
values improved from 0.0755 s / 172.12 MiB to 0.0405 s / 86.37 MiB by reusing
the owned output buffer instead of creating `np.nansum`'s full-size temporary.

## PCA sample/marker-space crossover

`benchmark_pca_crossover.py` measures exact PCA across tall, square, and wide
genotype shapes. For tall inputs it compares a retained-centered reference with
sample-batched marker-space Gram accumulation. Each candidate timing is
preceded by eigenvalue, explained-variance, score, and loading equivalence
checks. The report records the enlarged workspace used to force the retained
reference after production PCA gained its adaptive path.

Run the default workload and retain its JSON report:

```powershell
pixi run python benchmarks/benchmark_pca_crossover.py --output benchmarks/results/pca-crossover.json
```

For a quick validation run:

```powershell
pixi run python benchmarks/benchmark_pca_crossover.py --individuals 120 --marker-ratios 0.25 0.75 1 2 --warmups 0 --repeats 1
```

The sample-batched candidate makes a second pass to compute scores and can
increase disk I/O. Prefer it only when the traced-memory reduction remains
material at representative scales without an unacceptable runtime penalty.

`benchmark_pca_store_io.py` isolates another disk-specific effect: a dense
retained marker range can be served by one parent read, while an interleaved MAF
selection may require many small reads for every sample batch. It compares both
layouts on NumPy mmap and, when installed, HDF5 while recording parent read
calls and cells transferred:

```powershell
pixi run python benchmarks/benchmark_pca_store_io.py --output benchmarks/results/pca-store-io.json
```

Use this report to verify the storage-chunk-aware read planner against new
backends and chunk layouts; timing differences alone are not CI thresholds.

On the development Windows machine (2,000 samples, 2,000 parent markers, 500
retained markers, 1 MiB workspace), the original interleaved HDF5 path used
8,500 parent reads and 5.209 s. Chunk-aware coalescing reduced that case to 41
reads and 0.207 s while transferring 11,922,000 cells, below the planner's 4x
over-read limit relative to the 3,000,000 requested cells. Dense HDF5 remained
at 24 reads. NumPy mmap deliberately retains exact sparse reads because it does
not expose a physical marker-chunk width. This workload demonstrates I/O call
amplification rather than establishing a universal timing threshold.

## Hotspot profiles

After recording a baseline, use deterministic profiler scenarios to separate
marker-heavy iterative work from sample-heavy mixed-model work:

```powershell
pixi run python benchmarks/profile_hotspots.py --scenario marker-heavy --models farmcpu blink
pixi run python benchmarks/profile_hotspots.py --scenario sample-heavy --models mlm mlmm cblup super
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
pixi run python benchmarks/benchmark_cblup_eigensolvers.py --output benchmarks/results/cblup-eigensolver-crossover.json
```

For a quick validation run:

```powershell
pixi run python benchmarks/benchmark_cblup_eigensolvers.py --individuals 120 --group-ratios 0.25 0.5 --kinship-rank-fractions 1.0 0.5 --warmups 0 --repeats 1
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
pixi run python -m benchmarks.benchmark_rrblup --output benchmarks/results/rrblup.json
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
