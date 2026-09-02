# Changelog

All notable changes to pyGAPIT will be documented in this file. The project
uses [Semantic Versioning](https://semver.org/) and the structure follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/ou6326/pygapit/compare/v1.2.0...main
[1.2.0]: https://github.com/ou6326/pygapit/releases/tag/v1.2.0
