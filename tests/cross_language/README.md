# Cross-language alignment tests

This directory is deliberately separate from the ordinary Python unit tests. It compares deterministic Python implementations with the GAPIT 3.5 R source pinned as the `GAPIT/` Git submodule at commit `8d6651c719484c9f6c844144783dca1e4ef85b3e`.

The suite is divided by function so a failure identifies the affected layer directly:

- `test_source.py`: pinned GAPIT 3.5 reference commit
- `test_kinship.py`: VanRaden relationship matrix, including invariant markers
- `test_pca.py`: genotype PCA eigenvalues, explained variance, and scores
- `test_compression.py`: CMLM average-linkage groups and compressed kinship
- `test_multiple_testing.py`: Benjamini–Hochberg adjustment via `stats::p.adjust`, including boundary p-values
- `test_glm.py`: additive GLM effects and p-values, with and without a covariate
- `test_farmcpu.py`: static-bin pseudo-QTN selection
- `test_blink.py`: LD pruning and BIC pseudo-QTN selection
- `test_mlm.py`: EMMA null-model REML likelihood, variance components, and heritability, plus SNP-level EMMAX/P3D p-values, effects, standard errors, and test statistics
- `test_mlm_boundaries.py`: monomorphic and missing-genotype P3D behavior
- `test_prediction.py`: gBLUP fixed effects, breeding values, prediction-error variances, and phenotype predictions

Each comparison sources the smallest relevant GAPIT R file instead of loading the full package.

An R runtime is required in addition to the Python `rpy2` package. If either dependency is unavailable, tests are skipped with an explicit reason so the ordinary suite remains usable. Set `PYGAPIT_REQUIRE_R_ALIGNMENT=1` to turn an unavailable R runtime or rpy2 import into a hard failure in a parity CI job.

Initialize the pinned R source and use the Pixi development environment, which includes R, rpy2, the Python development tools, and the required R `MASS` package:

```bash
git submodule update --init --recursive
pixi install -e dev
pixi run -e dev pytest tests/cross_language -q
```

The tests use fixed genotypes, phenotypes, covariates, and p-values from shared fixtures. They do not modify the R checkout or write output files. Missing numeric genotypes are checked under GAPIT's `Middle`, `Major`, and `Minor` imputation policies. Remaining model work is the complete iterative FarmCPU/BLINK workflow and SUPER-based top-level sBLUP selection; their independently testable numerical kernels and public Python dispatch contracts are already covered.
