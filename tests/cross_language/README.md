# Cross-language alignment tests

This directory is deliberately separate from the ordinary Python unit tests. It compares deterministic Python implementations with the GAPIT 3.5 R source pinned as the `GAPIT/` Git submodule at commit `8d6651c719484c9f6c844144783dca1e4ef85b3e`.

The suite is divided by function so a failure identifies the affected layer directly:

- `test_source.py`: pinned GAPIT 3.5 reference commit
- `test_kinship.py`: VanRaden relationship matrix, including invariant markers
- `test_hapmap.py`: one- and two-bit HapMap genotype numericalization
- `test_pca.py`: genotype PCA eigenvalues, explained variance, and scores
- `test_compression.py`: CMLM average-linkage groups and compressed kinship
- `test_multiple_testing.py`: Benjamini–Hochberg adjustment via `stats::p.adjust`, including boundary p-values
- `test_glm.py`: additive GLM effects and p-values, with and without a covariate
- `test_farmcpu.py`: static-bin selection plus the complete top-level iterative
  workflow's pseudo-QTNs, p-values, and effects
- `test_blink.py`: FDR candidate selection, LD pruning, prefix-BIC selection,
  and the complete top-level iterative workflow's pseudo-QTNs, p-values, and
  effects across zero, one, and multiple selected QTNs. SUB reward coverage
  includes one and multiple valid substitutes. It also characterizes GAPIT's
  infinite reward when no independent substitute exists; pyGAPIT intentionally
  normalizes that invalid p-value to the equivalent non-significant value `1.0`
  and an undefined duplicate-marker effect to `0.0`. Degenerate no-QTN
  calibration scales that make GAPIT return `NaN` retain pyGAPIT's original,
  valid p-values.
- `test_mlm.py`: EMMA null-model REML likelihood, variance components, and heritability, plus SNP-level EMMAX/P3D p-values, effects, standard errors, and test statistics
- `test_mlm_boundaries.py`: monomorphic and missing-genotype P3D behavior
- `test_workflows.py`: public GLM and MLM orchestration with PCA, covariates,
  supplied kinship, shuffled labels, and missing phenotype observations
- `test_official_data.py`: public GLM, MLM, CMLM, MLMM, FarmCPU, and BLINK
  workflows on GAPIT's bundled maize diversity panel. Comparisons cover every
  MAF-filtered marker, PCA, iterative-model pseudo-QTNs, effects and p-values,
  VanRaden kinship, MLM variance components, fixed 40-group CMLM compression,
  and covariate-free MLMM selection. The MLMM regression characterizes GAPIT's
  all-`NA` `seqQTN` result for a null-model optimum; the BLINK regression also
  characterizes GAPIT's omission of supplied PCA covariates from BIC selection
  and validates pyGAPIT against the corrected-CV R workflow.
- `test_cmlm.py`: public CMLM orchestration with fixed compression, native
  incidence-matrix REML/EMMAX statistics, and automatic group selection
- `test_cmlm_boundaries.py`: redundant incidence levels, near-collinear fixed
  effects, and singular-design behavior in native-incidence REML
- `test_mlmm.py`: public MLMM orchestration with PCA, covariates, supplied
  kinship, missing phenotypes, forward/backward QTN selection, and final
  marker p-values and effects
- `test_prediction.py`: direct and top-level gBLUP and cBLUP fixed effects,
  breeding values, prediction-error variances, phenotype predictions, variance
  components, and cBLUP compression selection

Each comparison sources the smallest relevant GAPIT R file instead of loading
the full package. The complete BLINK and FarmCPU tests execute GAPIT's ordinary
matrix branches. They replace the optional `bigmemory::is.big.matrix(...)`
probes with `FALSE` in memory, leaving the pinned source checkout unchanged and
avoiding a test-only `bigmemory` dependency. The official BLINK regression
additionally creates an in-memory corrected reference that forwards `CV1` into
both `Blink.BICselection()` calls; it runs the unmodified function first so
this intentional divergence remains characterized against the pinned source.

An R runtime is required in addition to the Python `rpy2` package. If either dependency is unavailable, tests are skipped with an explicit reason so the ordinary suite remains usable. Set `PYGAPIT_REQUIRE_R_ALIGNMENT=1` to turn an unavailable R runtime or rpy2 import into a hard failure in a parity CI job.

Initialize the pinned R source and use the Pixi development environment, which includes R, rpy2, the Python development tools, and the required R `MASS` package:

```bash
git submodule update --init --recursive
pixi install -e dev
pixi run -e dev pytest tests/cross_language -q
```

The tests use fixed genotypes, phenotypes, covariates, and p-values from shared
fixtures. They do not modify the R checkout or write output files. Missing
numeric genotypes are checked under GAPIT's `Middle`, `Major`, and `Minor`
imputation policies. The main remaining prediction workflow gap is GAPIT's
SUPER-based top-level sBLUP QTN selection.

Intentional divergences are kept explicit and tested. In particular, GAPIT
accepts PCA covariates in `Blink()` but omits `CV1` when it calls
`Blink.BICselection()`, so marker selection no longer controls for the supplied
population structure. pyGAPIT retains the covariates throughout BIC selection;
the official maize regression compares every marker with the corrected R
workflow and separately locks the different QTN set selected by unmodified
GAPIT.
GAPIT silently changes a CMLM group count that cannot support the fixed-effect
design to one and effectively switches to a GLM path. pyGAPIT excludes such counts
from a wider search and raises when the entire requested range is invalid,
rather than returning results from a different model without notice.
GAPIT also returns an all-zero REML sentinel for a singular fixed-effect
design; pyGAPIT raises `ValueError` because those values do not describe a
fitted model. Redundant or empty incidence levels remain accepted when their
marginal model is numerically defined and matches GAPIT.
For MLMM extended BIC, GAPIT includes ordinary covariates in the
combinatorial marker-selection penalty. pyGAPIT counts only selected marker
cofactors, preserving the intended statistical meaning when user covariates
or principal components are present. The public MLMM result is the extended
BIC model, so the generic `p_threshold` option does not truncate its forward
path; GAPIT's similarly named `thresh` selects a separate report rather than
controlling forward selection. GAPIT's ML likelihood can also return `-Inf`
with `NaN` warnings for a supplied indefinite kinship matrix; pyGAPIT rejects
such a matrix explicitly while tolerating insignificant floating-point
eigenvalue noise.
