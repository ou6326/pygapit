# Changelog

All notable changes to pyGAPIT will be documented in this file. The project
uses [Semantic Versioning](https://semver.org/) and the structure follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [2.0.0rc1] - 2026-09-23

### Added

- Typed `Visualization` objects with independently assignable notebook
  backends, direct native `render()`, and format-aware saving. Static formats
  select Matplotlib automatically; 3D PCA excludes Bokeh.
- Backend-neutral HoloViews objects for Manhattan, QQ, kinship, PCA, genomic
  selection, and phenotype plots, with rendering and saving delegated to
  HoloViews' Matplotlib, Bokeh, and Plotly backends, built on shared,
  backend-independent Manhattan data and genomic-axis preparation.
- A reproducible benchmark for genomic-axis and complete Manhattan data
  preparation, HoloViews object construction, and opt-in Matplotlib, Bokeh, or
  Plotly rendering at configurable marker counts.
- An optional backend using the Zarr Python 3.x API and Zarr format 2 storage,
  with the same labeled, chunk-readable contract as NumPy and HDF5.
- Chunked conversion between labeled NumPy, HDF5, and Zarr genotype stores
  without materializing the complete genotype matrix.
- Direct bounded sample-row import from numeric GD/GM files into any
  genotype-store backend without rescanning the source for every marker block.
- Single-pass marker-block import from HapMap files into any genotype-store
  backend.
- A reproducible large-marker benchmark over bounded numeric import, VanRaden
  kinship, PCA, a GLM marker scan, Manhattan preparation, and rendering at
  100k, 1M, and 10M markers with selectable NumPy, HDF5, and Zarr store
  backends, reporting wall time and process peak RSS.
- A staged numeric-import profiler that separates source and metadata setup,
  TSV parsing plus imputation, and NumPy persistence.

### Changed

- Document the Python 3.12 and HoloViews visualization migration from 1.2.4.
- Reuse precomputed genomic ordering when building GWAS tables and bypass
  DataFrame sorting and merging when multi-model results already share marker
  order.
- Store Manhattan hover fields as HoloViews value dimensions instead of
  allocating backend-specific formatted labels.
- Replace backend-specific plotting arguments and native figure return values
  with a typed HoloViews visualization view; construction selects notebook
  display preference while `render()` and `save_plot()` handle output.
- Use a linear, low-allocation genomic-axis path when chromosome blocks are
  contiguous while preserving arbitrary-order input compatibility.
- Reuse zero-copy chromosome slices while constructing exact Manhattan points
  and combine alternating chromosome colors in one categorical Datashader
  raster.
- Require Python 3.12 or newer and test the complete optional-storage stack on
  Python 3.12 and 3.14.
- Extend disk-backed top-level workflows through SUPER selection and sBLUP
  prediction while materializing and reusing only the bounded pseudo-QTN
  candidate pool.
- Support bounded disk-backed marker scans in direct and top-level CMLM.
- Extend direct and top-level MLMM to disk-backed marker batches while reading
  only its selected cofactors into each fixed-effect design.
- Support disk-backed FarmCPU iterations through the shared bounded GLM scan,
  reward substitution, and pseudo-QTN kinship paths.
- Add disk-backed BLINK with bounded GLM and LD-pruning reads, plus reuse of its
  truncated BIC candidate pool.
- Support disk-backed cBLUP prediction by streaming VanRaden marker blocks;
  compression still operates on the complete in-memory kinship matrix.

### Removed

- Backend-native plotting functions are removed rather than deprecated, with
  no compatibility wrappers:

  - `manhattan_plot()` → `manhattan()` plus `save_plot()`
  - `manhattan_interactive()` → `manhattan(..., backend="bokeh" | "plotly")`
  - `pca_plot_3d_interactive()` → `pca_plot_3d()`

- The plotting `save_path=` parameters, such as `qq_plot(..., save_path=)` and
  `kinship_heatmap(..., save_path=)`. Construct a `Visualization` and call
  `save_plot()` instead. `MIGRATING.md` records the full mapping.

### Fixed

- Failed NumPy, HDF5, and Zarr writes remove the incomplete genotype store so
  the same destination can be retried safely.
- Failed plot generation no longer reports stale files from an earlier run as
  outputs produced by the current analysis.
- Static-bin FarmCPU no longer derives a pseudo-kinship REML variance fit.
  GAPIT's static-bin path returns no variance components, so the result fields
  `vg`, `ve`, and `h2` are now fixed at `0.0` instead of an unsupported
  estimate.
- sBLUP PEV for intentionally low-rank pseudo-kinship now inverts the kinship
  at GAPIT's `MASS::ginv` tolerance instead of NumPy's smaller default cutoff,
  so PEV and prediction-error values match the R reference.
- MLMM reproduces GAPIT's complete forward and backward cofactor paths, so the
  retained model and its marker statistics match the R reference at every
  iteration.

## [1.2.4] - 2026-09-12

### Added

- A chunk-readable `GenotypeStore` contract with in-memory, NumPy memory-mapped,
  and optional HDF5 backends. Unified open/write functions select an available
  disk backend automatically; MAF filtering, VanRaden kinship, PCA, and direct
  GLM/MLM scans read each in bounded marker blocks. Wide-matrix PCA, VanRaden,
  and marker scans avoid materializing a complete centered genotype matrix;
  exact tall-matrix PCA switches to two-pass sample batching when its centered
  matrix would substantially exceed the workspace budget.
- `GenotypeView` composes sample and marker selections over any genotype store
  while preserving bounded, contiguous reads from the parent store. Sparse
  selections over chunked backends are coalesced within physical marker chunks
  under a bounded over-read policy.
- Top-level `GAPIT()` GLM and MLM workflows accept labeled genotype stores and
  perform taxa alignment, trait filtering, MAF filtering, PCA, kinship, and
  marker scans through bounded source reads.

### Changed

- Extend tested Python support through Python 3.14 while retaining Python 3.10
  as the minimum supported version. The default Pixi environment now uses
  Python 3.14 with development and HDF5 dependencies; `full310` verifies the
  lower bound.
- Stop declaring Biopython, Dask, Jinja2, Joblib, Seaborn, and Zarr as direct
  project dependencies ahead of integrations that use them. Most remain
  planned for later features; in 1.2.4 the optional `bigdata` feature adds only
  the implemented HDF5 support.
- Phenotype simulation and its example use local NumPy generators without
  mutating process-wide random state.

## [1.2.3] - 2026-09-06

### Added

- A canonical `rrblup()` full-data fit and immutable `RRBLUPResult` containing
  marker means, effects, intercept, regularization, GEBV, and phenotype
  prediction.
- A validated `marker_workspace_mib` option for wide-matrix PCA, VanRaden
  kinship construction, and direct and top-level GLM/MLM scans, also available
  from the command line.
- Fold-local RR-BLUP and gBLUP cross-validation APIs with seeded or grouped
  splits, explicit fold assignments, immutable out-of-fold predictions,
  per-fold regularization, Pearson correlation, and RMSE.
- An equivalent-solve RR-BLUP benchmark comparing marker-space reference
  equations against dimension-adaptive solving.
- Full-fit RR-BLUP R references for centering, intercept, marker effects and
  GAPIT's BLUP component, plus automatic-REML constant-phenotype regressions.

### Fixed

- Full-data RR-BLUP accepts two finite samples independently of the stricter
  three-sample cross-validation contract; shared ridge fitting internals no
  longer depend on the validation module.
- Validate marker workspace budgets at every direct GLM/MLM public boundary,
  even when the genotype contains no variable markers to scan.
- gBLUP CV now enforces the documented PSD kinship requirement with a
  scale-aware roundoff tolerance and a consistent symmetric copy.
- CV grouping accepts Pandas object/string columns containing homogeneous,
  non-missing string or integer IDs.
- Standalone RR-BLUP and GBLUP validation now predicts every sample, including
  remainder samples when the sample count is not divisible by the fold count.
- Standalone RR-BLUP estimates imputation means and REML penalties separately
  within each training fold, fits an unpenalized intercept, and uses a kernel
  scale consistent with its marker penalty. Its returned GEBV excludes the
  intercept; old uncentered predictions and CV scores intentionally change.
- Standalone GBLUP reuses canonical gBLUP for its full fit and estimates a GLS
  intercept within each validation fold, without a full-data fallback.

### Changed

- Whiten MLM genotypes in bounded marker batches instead of allocating a
  second full transformed-genotype matrix.
- Expose canonical genomic prediction fits and validation APIs consistently
  from both `pygapit` and `pygapit.gs`, while retaining the uppercase
  standalone functions as compatibility wrappers.
- Use Cholesky solves for positive-penalty RR-BLUP and gBLUP validation
  systems, with a general-solve fallback for numerical failures.
- Clarify the distinction between training-only RR-BLUP preprocessing and
  gBLUP fitting over an externally supplied, possibly transductive kinship.
- Standalone RR-BLUP solves in sample space when markers outnumber training
  samples, avoiding the marker-by-marker dense system.

## [1.2.2] - 2026-09-05

### Added

- Deterministic profiling and benchmark workloads for GWAS pipelines, cBLUP
  eigensolver selection, multi-trait execution, and genotype preprocessing.

### Changed

- Reduce repeated matrix decompositions, inversions, projections, and marker
  work across GLM, MLM, cBLUP, MLMM, FarmCPU, BLINK, SUPER, and genomic
  prediction paths.
- Reuse compatible fits and genotype-derived preparation across traits while
  preserving trait-specific missing-data masks and prediction settings.
- Accelerate numeric and HapMap input conversion, missing-value imputation,
  PCA, and VanRaden kinship construction with lower peak allocations.
- Use an explicit NumPy generator in BayesB so sampling no longer depends on
  process-global random state.

### Fixed

- Stabilize covariate projection and define degenerate marker statistics in
  vectorized EMMAX scans.
- Preserve caller-owned numeric genotype frames during imputation and reject
  duplicate HapMap taxa before Pandas can normalize their column names.

## [1.2.1] - 2026-09-02

### Changed

- Publish the distribution as `pygapit-ng` while retaining `pygapit` as the
  import package and command-line entry point.

## [1.2.0] - 2026-09-02

### Added

- GAPIT-style GLM, MLM, CMLM, MLMM, FarmCPU, and BLINK association workflows.
- Direct and top-level gBLUP, cBLUP, and SUPER-based sBLUP prediction workflows.
- R-backed GAPIT 3.5 regression tests, including official maize data coverage.
- Typed result objects, label-aware input alignment, plotting, CLI, and optional
  big-data support.

### Changed

- Build PyPI wheel and source distributions with Hatchling instead of
  setuptools.
- Statistically invalid or ambiguous GAPIT 3.5 behavior is intentionally
  normalized or rejected when a characterization test documents the upstream
  result.
- Public replacement parameters use descriptive snake-case names while legacy
  GAPIT spellings remain available for migration compatibility.

### Fixed

- Native-incidence CMLM and cBLUP estimation, iterative FarmCPU and BLINK model
  selection, MLMM extended BIC, and SUPER pseudo-QTN selection now follow their
  documented statistical objectives.

[Unreleased]: https://github.com/ou6326/pygapit/compare/v2.0.0rc1...main
[2.0.0rc1]: https://github.com/ou6326/pygapit/compare/v1.2.4...v2.0.0rc1
[1.2.4]: https://github.com/ou6326/pygapit/compare/v1.2.3...v1.2.4
[1.2.3]: https://github.com/ou6326/pygapit/compare/v1.2.2...v1.2.3
[1.2.2]: https://github.com/ou6326/pygapit/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/ou6326/pygapit/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/ou6326/pygapit/releases/tag/v1.2.0
