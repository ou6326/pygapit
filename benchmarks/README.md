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

## Large-marker manual workload

`benchmark_large_scale.py` combines the disk-backed paths that matter at
marker scale: bounded numeric-file import into a NumPy, HDF5, or Zarr store,
VanRaden, PCA, a GLM marker scan, Manhattan preparation, exact and Datashader
HoloViews objects, and a renderer pass. It uses 64 individuals and 100,000
markers by default, with no warm-up and one repeat so it remains practical on a
developer machine. Its synthetic GD/GM source is written in 4,096-marker
blocks, so selecting 1M or 10M does not first allocate a whole genotype matrix
or marker table in RAM. It is manual-only and does not run those scales unless
selected. The 100k workload includes exact-point HoloViews construction and
rendering; the 1M and 10M selections deliberately skip exact points and retain
genomic axis, complete Manhattan data, Datashader aggregation, and aggregate
rendering.

`--store-backend` selects the store under test: `numpy` (default), `hdf5`, or
`zarr`. The optional backends need `pygapit-ng[bigdata]`; spot-check them at the
same marker count to compare their bounded import, scan, and rendering cost.

```powershell
pixi run python benchmarks/benchmark_large_scale.py --markers 100000 --output benchmarks/results/large-100k.json
pixi run python benchmarks/benchmark_large_scale.py --markers 100000 --store-backend hdf5 --output benchmarks/results/large-100k-hdf5.json
pixi run python benchmarks/benchmark_large_scale.py --markers 100000 --store-backend zarr --output benchmarks/results/large-100k-zarr.json
pixi run python benchmarks/benchmark_large_scale.py --markers 1000000 --output benchmarks/results/large-1m.json
pixi run python benchmarks/benchmark_large_scale.py --markers 10000000 --output benchmarks/results/large-10m.json
```

All stages report wall time plus `traced_peak_mib`; one separate
`scenario_process_peak_rss_mib` covers the entire import, numerical, and
rendering run. On Windows it samples the process working set through the native
process API; on POSIX it uses `resource.ru_maxrss`. The POSIX value is a process
high-water mark, so use a fresh `pixi run` process for comparisons and do not
attribute it to an individual stage. It includes parser, mmap, renderer, and
native numerical memory; it is the appropriate field for machine-capacity
planning, while tracemalloc remains useful for Python-allocation comparisons.

On the development Windows machine (64 individuals, 100,000 markers, 32 MiB
marker workspace, one repeat, Bokeh rendering), the streaming numeric import
took 8.53 s into a NumPy store, 10.06 s into HDF5, and 8.70 s into Zarr, each
with a 304 MiB tracemalloc peak. Whole-scenario process peak RSS was 970 MiB,
1014 MiB, and 1005 MiB respectively. These are machine-specific spot-check
observations, not CI thresholds.

At 1,000,000 markers the same machine is import-bound: the streaming import
took 566.6 s (NumPy), 566.3 s (HDF5), and 501.2 s (Zarr), each with a 2338 MiB
tracemalloc peak, while VanRaden kinship, PCA, and the GLM marker scan each
finished in seconds and the whole-scenario process peak RSS stayed near
7.9-8.0 GiB. The three backends stay within the same order of magnitude with
no backend-specific blow-up, so the cost is dominated by text parsing and
imputation rather than the store write; 10M stays a capacity-planning exercise
rather than a routine spot check.

To attribute numeric-import time within that scenario, run the stage profiler:

```powershell
pixi run python benchmarks/benchmark_numeric_import_profile.py --markers 100000 --output benchmarks/results/numeric-import-100k.json
```

It separates source and metadata setup, TSV parsing plus imputation, and NumPy
matrix/metadata persistence. The profiler deliberately performs additional
imports: one set supplies end-to-end timing and tracemalloc measurements, and a
separate set supplies stage attribution without changing the production import
path. Results are machine-dependent diagnostics, not release gates. On the
development Windows machine (64 individuals, 100,000 markers, 32 MiB marker
workspace, no warm-up, one repeat), the total import took 7.82 s with a 304 MiB
tracemalloc peak: setup took 3.62 s, parsing and `Middle` imputation 4.35 s, and
NumPy matrix plus metadata persistence 0.12 s. The source uses direct sample-row
writes, so row-to-marker reordering took 0 s and allocated no reorder buffer.

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

## Manhattan plot pipeline

`benchmark_plot_preparation.py` separates genomic-axis construction, complete
Manhattan data preparation, exact-point HoloViews construction, and Datashader
HoloViews construction. Inputs are allocated before memory tracing so the
report isolates each stage's additional workspace.

Run marker scales separately to avoid retaining multiple large input sets:

```powershell
pixi run python benchmarks/benchmark_plot_preparation.py --markers 100000
pixi run python benchmarks/benchmark_plot_preparation.py --markers 1000000
pixi run python benchmarks/benchmark_plot_preparation.py --markers 10000000
```

Renderer execution is deliberately opt-in because rendering 1M or 10M exact
points can be expensive. Repeat `--render-backend` to compare the same
HoloViews specification across renderers:

```powershell
pixi run python benchmarks/benchmark_plot_preparation.py --markers 100000 --render-backend matplotlib --render-backend bokeh --render-backend plotly
```

Renderer measurements clone a pre-built template before every render so a
populated DynamicMap/Datashader cache is not reused across repetitions.

The benchmark uses chromosome-contiguous data, matching GWAS tables emitted by
the top-level pipeline. Timings and traced peaks are diagnostic observations,
not CI thresholds; renderer results should be compared only on the same
machine.

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
