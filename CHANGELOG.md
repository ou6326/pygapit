# Changelog

All notable changes to pyGAPIT will be documented in this file. The project
uses [Semantic Versioning](https://semver.org/) and the structure follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A canonical `rrblup()` full-data fit and immutable `RRBLUPResult` containing
  marker means, effects, intercept, regularization, GEBV, and phenotype
  prediction.
- A validated `marker_workspace_mib` option for direct and top-level GLM/MLM
  scans and VanRaden kinship construction, also available from the command
  line.
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

[Unreleased]: https://github.com/ou6326/pygapit/compare/v1.2.2...main
[1.2.2]: https://github.com/ou6326/pygapit/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/ou6326/pygapit/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/ou6326/pygapit/releases/tag/v1.2.0
