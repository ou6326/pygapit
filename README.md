# pyGAPIT — Genome Association and Prediction Integrated Tool (Python)

**Compatibility target:** pyGAPIT currently targets R **GAPIT 3.5**, using the official [GAPIT3.5](https://github.com/jiabowang/GAPIT/tree/GAPIT3.5) by Jiabo Wang & Zhiwu Zhang as its upstream reference. That tag currently resolves to commit [`8d6651c`](https://github.com/jiabowang/GAPIT/tree/8d6651c719484c9f6c844144783dca1e4ef85b3e). This identifies the intended upstream baseline; exact numerical and interface parity has not yet been established. GAPIT v4 compatibility is not currently targeted.

It provides GAPIT-style inputs for the GWAS models **GLM, MLM, CMLM, MLMM, FarmCPU, and BLINK**, together with **gBLUP, cBLUP, and sBLUP** genomic-selection functions. It is not a drop-in replacement for every R GAPIT parameter.

### Compatibility policy

GAPIT 3.5 is a pinned behavioral reference, not a requirement to reproduce
every historical implementation detail. pyGAPIT follows its numerical results
when the upstream behavior is statistically valid and well defined. A safer
Python behavior is used when GAPIT produces invalid statistics, silently
changes the requested model, discards data labels, or relies on an obsolete
runtime convention. Each such intentional divergence must be documented and
covered by a characterization or regression test.

The `GAPIT()` entry point currently retains several R-derived parameter names
for migration compatibility. New public interfaces and replacement parameters
will use descriptive `snake_case` names, explicit option semantics, and typed
result objects. Legacy spellings will be deprecated gradually rather than
removed without a transition period.

### Compatibility evidence

The table below reports tested behavior, not an estimate of feature
completeness. **R-validated** means that an automated cross-language test
compares pyGAPIT with the pinned GAPIT 3.5 source. **Python-only** means that
the interface has ordinary regression tests but does not yet have a direct R
comparison. **Not yet** identifies an evidence gap rather than a known model
failure.

| Model | Public interface | GAPIT 3.5 numerical evidence | Official maize regression | Current boundary coverage |
|---|---|---|---|---|
| GLM | `GAPIT(model="GLM")` | R-validated top-level workflow | Full MAF-filtered marker set | PCA, covariates, supplied kinship alignment, shuffled labels, and missing phenotypes |
| MLM | `GAPIT(model="MLM")` | R-validated top-level workflow and EMMA/P3D statistics | Full MAF-filtered marker set | Variance components, monomorphic markers, missing genotypes, and data alignment |
| CMLM | `GAPIT(model="CMLM")` | R-validated top-level workflow | Full MAF-filtered marker set with fixed 40-group compression | Fixed and automatic compression, native incidence matrices, redundant levels, near-collinear covariates, and invalid designs |
| MLMM | `GAPIT(model="MLMM")` | R-validated top-level workflow | Full MAF-filtered marker set without added covariates | Forward/backward selection, final marker statistics, corrected extended BIC, and indefinite-kinship rejection |
| FarmCPU | `GAPIT(model="FarmCPU")` | R-validated complete iterative workflow | Full MAF-filtered marker set | Static-bin selection, pseudo-QTNs, final p-values, and effects |
| BLINK | `GAPIT(model="BLINK")` | R-validated iterative workflow with the upstream missing-CV BIC call characterized | Full MAF-filtered marker set against the corrected-CV reference | PCA-aware BIC, FDR candidates, LD pruning, zero/one/multiple-QTN paths, and invalid-statistic normalization |
| gBLUP | `gblup()` and `GAPIT(..., prediction_model="gBLUP")` | R-validated direct and prediction workflows | Full EarHT prediction set | BLUE, BLUP, PEV, predictions, and variance components |
| cBLUP | `cblup()` and `GAPIT(model="cBLUP")` | R-validated direct and top-level workflows | Not yet | Compression selection, native-incidence BLUE/BLUP/PEV, predictions, and variance components |
| sBLUP | `sblup()` or a GWAS prediction override | Python-only | Not yet | Explicit pseudo-QTN validation; standalone top-level `model="sBLUP"` and GAPIT SUPER-based QTN selection are not implemented |

The official-data column currently refers to GAPIT's bundled maize diversity
panel and the `EarHT` trait. GLM, MLM, CMLM, MLMM, FarmCPU, and BLINK
comparisons cover every marker retained by the shared MAF filter; gBLUP covers
the complete set of phenotyped taxa. The CMLM
regression uses a fixed 40-group compression; automatic compression selection
remains covered by the smaller cross-language workflow test. The MLMM
regression omits added covariates so GAPIT's extended-BIC penalty is valid; its
broken all-`NA` `seqQTN` output for this null-model optimum is characterized,
while pyGAPIT returns an empty QTN array. The BLINK reference forwards
the already supplied PCA covariates into GAPIT's two BIC calls; the unmodified
upstream path is also executed to lock its different QTN selection as an
intentional divergence. Other rows must not be interpreted as official-data
parity until a corresponding regression is added.

Intentional divergences are tested rather than hidden. They include replacing
GAPIT's invalid BLINK `NaN`/infinite statistics with documented valid outputs,
retaining PCA covariates during BLINK BIC selection,
rejecting CMLM requests that silently change the model or return a singular-fit
sentinel, correcting MLMM's extended-BIC marker penalty and invalid null-model
QTN sentinel, and rejecting
materially indefinite supplied kinship matrices. The detailed test inventory
and divergence rationale are maintained in
[`tests/cross_language/README.md`](tests/cross_language/README.md); the required
R-backed suite runs in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Installation

### Regular use

pyGAPIT requires Python 3.10 or newer. Install it from this source checkout.
This installs only the Python runtime dependencies; R and the GAPIT reference
repository are not required.

```bash
pip install .

pip install ".[bigdata]"  # include HDF5, Zarr, and Dask for larger datasets
```

**Runtime dependencies** are installed automatically: `numpy`, `scipy`,
`pandas`, `matplotlib`, `seaborn`, `plotly`, `joblib`,
`biopython`, and `jinja2`.

### Development

Use an editable installation when changing pyGAPIT itself. The development
dependencies provide pytest, Ruff, Pyrefly, BasedPyright, rpy2, and the type
stubs. The GAPIT reference repository is still not needed for ordinary Python
development.

```bash
pip install -e ".[dev]"

pip install -e ".[bigdata]"      # editable install with big-data support
pip install -e ".[dev,bigdata]"  # development tools plus big-data support
```

The equivalent Pixi environment is:

```bash
pixi install -e dev
pixi shell -e dev
```

Run the ordinary Python checks without the R reference repository:

```bash
pixi run -e dev ruff check .
pixi run -e dev ruff format --check .
pixi run -e dev pyrefly check -p all
pixi run -e dev basedpyright
pixi run -e dev pytest tests/test_pygapit.py
```

### GAPIT 3.5 alignment validation

This level is only for maintainers checking numerical behavior against the
pinned R GAPIT 3.5 source. It uses the same development environment, but also
initializes the `GAPIT/` Git submodule. The submodule is not used by the
installed pyGAPIT package at runtime.

```bash
# Fetch the pinned GAPIT 3.5 reference only when running alignment tests
git submodule update --init --recursive

pixi install -e dev
pixi run -e dev pytest tests/cross_language -q
```

For one environment containing development and big-data dependencies, use
`full`:

```bash
pixi install -e full
pixi shell -e full
```

If Pixi is unavailable, development and validation require an existing R
installation and its `MASS` package before installing the Python development
dependencies:

```bash
pip install -e ".[dev]"
pytest tests/cross_language -q
```

### Performance baseline

A deterministic manual benchmark covers PCA, VanRaden kinship, GLM, MLM,
FarmCPU, and BLINK without adding machine-dependent timing thresholds to CI:

```bash
pixi run -e full python benchmarks/run_baseline.py --output benchmarks/results/baseline.json
```

See [`benchmarks/README.md`](benchmarks/README.md) for workload and memory
measurement details. Benchmark reports are evidence for comparing commits on
the same machine; they are not universal performance guarantees.

---

## Quick start

```python
import pandas as pd
from pygapit import GAPIT

# Load GAPIT-style tabular data
Y = pd.read_csv("mdp_traits.txt", sep="\t")  # phenotype
GD = pd.read_csv("mdp_numeric.txt", sep="\t")  # numeric genotype
GM = pd.read_csv("mdp_SNP_information.txt", sep="\t")  # SNP map

# Select one trait so GAPIT returns one GAPITResult instead of a result dict
result = GAPIT(
    Y=Y,
    GD=GD,
    GM=GM,
    model="BLINK",
    trait="EarHT",
    PCA_total=3,
)

print(result.GWAS.head())  # full GWAS results table
print(f"h²    = {result.h2:.3f}")  # heritability
print(f"λ     = {result.lambda_gc:.3f}")  # genomic inflation factor
print(f"QTNs  = {0 if result.QTNs is None else len(result.QTNs)}")
```

**Comparable R GAPIT call when `Y` contains the selected trait:**
```r
myGAPIT <- GAPIT(Y=myY, GD=myGD, GM=myGM, model="Blink", PCA.total=3)
```

---

## Input data formats

pyGAPIT accepts GAPIT-style phenotype, numeric-genotype/map, and HapMap inputs:

### Phenotype file (`Y`)
Tab-delimited. First column = Taxa names, remaining columns = trait values.
```
Taxa    EarHT   dpoll
33-16   64.75   64.5
38-11   69.12   61.0
4226    65.5    59.5
```

### Numeric genotype (`GD`) + map (`GM`)
`GD`: First column = taxa names, remaining = SNP dosages (0/1/2).
```
taxa        PZB00859.1  PZA01271.1  ...
33-16       2           0           ...
38-11       2           2           ...
```
`GM`: Three columns: SNP name, Chromosome, Position (bp).
```
SNP         Chromosome  Position
PZB00859.1  1           157104
PZA01271.1  1           1947984
```

### HapMap genotype (`G`)
Standard HapMap format with IUPAC allele codes.
```python
result = GAPIT(Y=Y, G=hapmap_df, model="BLINK")
```

---

## GWAS models

| Model    | Method type  | Uses kinship | Multi-QTN | Power   | Speed    |
|----------|-------------|-------------|-----------|---------|----------|
| `GLM`    | Single-locus | No (PCs)    | No        | Low     | Fastest  |
| `MLM`    | Single-locus | Yes (global) | No        | Medium  | Fast     |
| `CMLM`   | Single-locus | Compressed  | No        | Medium+ | Fast     |
| `MLMM`   | Multi-locus  | Yes (global) | Yes       | High    | Moderate |
| `FarmCPU`| Multi-locus  | Pseudo-QTN  | Yes       | High    | Moderate |
| `BLINK`  | Multi-locus  | No          | Yes       | High    | Fast     |

```python
# Run multiple models simultaneously
result = GAPIT(
    Y=Y, GD=GD, GM=GM, model=["GLM", "MLM", "FarmCPU", "BLINK"], trait="EarHT"
)
# Returns a dict keyed by "EarHT_GLM", "EarHT_MLM", etc.
```

---

## Genomic selection

```python
# gBLUP — best for polygenic traits
result = GAPIT(Y=Y, GD=GD, GM=GM, model="gBLUP", trait="EarHT")

# cBLUP — compressed-kinship prediction
result = GAPIT(Y=Y, GD=GD, GM=GM, model="cBLUP", trait="EarHT")

# Run gBLUP prediction after BLINK
result = GAPIT(Y=Y, GD=GD, GM=GM, model="BLINK", trait="EarHT", buspred=True)

# FarmCPU + buspred uses sBLUP when FarmCPU identifies QTNs;
# otherwise prediction falls back to gBLUP
result = GAPIT(Y=Y, GD=GD, GM=GM, model="FarmCPU", trait="EarHT", buspred=True)

# Direct sBLUP requires pseudo-QTN column indices from a prior GWAS
from pygapit import sblup

prediction = sblup(y, X0, GD_array, qtn_indices=selected_qtns)

# Access prediction results
print(result.Pred)
#      Taxa    BLUE    BLUP     PEV   gBreedingValue  Prediction
# 0  33-16   67.4   -2.65   89.3      -2.65          64.75
```

---

## Output files

When `file_output=True` (default), pyGAPIT writes to `output_dir`:

| File | Content |
|------|---------|
| `GAPIT.BLINK.EarHT.GWAS.Results.csv` | Full GWAS table: SNP, Chr, Pos, P.value, maf, effect, FDR |
| `GAPIT.BLINK.EarHT.Prediction.csv` | BLUE, BLUP, PEV, GEBV per individual; written only when `buspred=True` and prediction succeeds |
| `GAPIT.EarHT.Kinship.csv` | Selected or supplied kinship matrix |
| `GAPIT.EarHT.PCA.csv` | PC scores per individual |
| `GAPIT.BLINK.EarHT.Manhattan.pdf` | Manhattan plot |
| `GAPIT.BLINK.EarHT.QQ.pdf` | QQ plot with λ annotation |
| `GAPIT.EarHT.Kinship.pdf` | Kinship heatmap |
| `GAPIT.EarHT.PCA.pdf` | 2D PCA scatter |

The returned `GAPITResult.output_files` records every file written for that
analysis. With `file_output=False`, `output_files` is `None` and `output_dir`
is not created.

---

## Parameter reference

The main supported GAPIT-style parameters are:

| R parameter | Python parameter | Default | Description |
|-------------|-----------------|---------|-------------|
| `model` | `model` | `"BLINK"` | Model(s) to run |
| `PCA.total` | `PCA_total` | `3` | Number of PCs as covariates |
| `maf.threshold` | `maf_threshold` | `0.05` | Minimum MAF filter |
| `SNP.impute` | `SNP_impute` | `"middle"` | Missing genotype imputation |
| `file.output` | `file_output` | `True` | Write result files |
| `cutOff` | `cutOff` | Bonferroni | Significance threshold |
| `LD` | `LD` | `0.7` | LD threshold for BLINK pruning |
| `group.from` | `group_from` | `1` | Min groups for CMLM |
| `group.to` | `group_to` | n | Max groups for CMLM |
| `bin.size` | `bin_size` | `5000000` | Bin size (bp) for FarmCPU |
| `h2` | `h2` | `None` | Heritability for simulation |
| `NQTN` | `NQTN` | `None` | QTNs for simulation |
| `buspred` | `buspred` | `False` | Run GS after GWAS |
| `FDRcut` | `FDRcut` | `False` | Use the GAPIT 3.5 FDR cutoff for BLINK pseudo-QTNs |
| `kinship.algorithm` | `kinship_algorithm` | `"VanRaden"` | `"VanRaden"` or `"Zhang"` |
| `Z` | `Z` | `None` | Incidence matrix; combines with `KI` as `Z @ KI @ Z.T` |
| — | `prediction_model` | `None` | Override prediction with `gBLUP`, `cBLUP`, or `sBLUP` |
| `Multiple_analysis` | `Multiple_analysis` | `False` | Write combined Manhattan and QQ plots by trait |

When `Z` is supplied, `KI` represents covariance among the random-effect
levels (the columns of `Z`). NumPy inputs are aligned positionally. DataFrame
inputs are aligned by phenotype taxa and random-effect labels, and mismatched
labels are rejected. An explicit `p_threshold` takes precedence over `FDRcut`
during BLINK candidate selection. Multiple-analysis plots join models by SNP,
chromosome, and position before drawing them on a shared genomic axis; they are
written only when `file_output=True`.

---

## Command-line interface

```bash
# Basic GWAS
pygapit --Y traits.txt --GD geno.txt --GM map.txt --model BLINK

# Multiple models, custom output directory
pygapit --Y traits.txt --GD geno.txt --GM map.txt \
        --model GLM MLM BLINK FarmCPU \
        --PCA_total 5 --output_dir results/

# Genomic prediction
pygapit --Y traits.txt --GD geno.txt --GM map.txt --model gBLUP

# Phenotype simulation
pygapit --Y traits.txt --GD geno.txt --GM map.txt \
        --model BLINK --h2 0.7 --NQTN 20
```

---

## Using individual functions

```python
import numpy as np

from pygapit import (
    blink_gwas,
    bonferroni_threshold,
    build_covariate_matrix,
    compute_pca,
    emma_remle,
    farmcpu_gwas,
    gblup,
    genomic_inflation_factor,
    glm_gwas,
    manhattan_plot,
    mlm_gwas,
    qq_plot,
    vanraden_kinship,
)

# Compute kinship
K = vanraden_kinship(GD_array)  # (n, n) VanRaden matrix

# PCA for structure control
pca = compute_pca(GD_array, n_components=3)
X0 = build_covariate_matrix(pca, n_pcs=3)

# REML variance components
remle = emma_remle(y, X0, K)
print(f"h² = {remle.h2:.3f}")

# Run BLINK GWAS
result = blink_gwas(y, X0, GD_array, max_iterations=10, ld_threshold=0.7)
lam = genomic_inflation_factor(result.p_values)
thresh = bonferroni_threshold(len(result.p_values))
sig = (result.p_values <= thresh).sum()
print(f"λ = {lam:.3f},  {sig} significant SNPs")

# Genomic prediction
gs = gblup(y, X0, K)
print(f"Prediction accuracy (r): {np.corrcoef(y, gs.prediction)[0, 1]:.3f}")

# Plots
manhattan_plot(
    snp_names, chromosomes, positions, result.p_values, save_path="manhattan.pdf"
)
qq_plot(result.p_values, save_path="qq.pdf")
```

---

## Mathematical models

### Mixed Linear Model (MLM)
```
y = X·β + u + e
u ~ N(0, K·σ²g),   e ~ N(0, I·σ²e)
```
Variance components estimated by **REML via EMMA** (Kang et al. 2008):
spectral decomposition of K → grid search + Brent's method for optimal δ = σ²e/σ²g.
**P3D approximation**: δ estimated once from null model, fixed for all m SNP tests.

### VanRaden Kinship (2009)
```
K = ZZ' / [2 · Σⱼ pⱼ(1-pⱼ)]
Z = GD - 2p          (column-centered 0/1/2 coding)
p = alternate-allele frequencies
```

### BLINK iteration
```
Loop until convergence:
  1. GLM-1: sort markers by p-value
             LD-prune candidates (r² > threshold)
             select cofactors by BIC minimization
  2. GLM-2: test all m markers with cofactor set as fixed effects
             → updated p-values
```
BIC = -2·logL + k·log(n)  — replaces expensive REML from FarmCPU.

### Henderson's MME (gBLUP)
```
[X'X        X'Z        ] [β]   [X'y]
[Z'X   Z'Z + δ·K⁻¹     ] [u] = [Z'y]

BLUP = û,   BLUE = X·β̂
PEV  = diag(C⁻¹)ᵤᵤ · σ²g
```

---

## Citation

If you use pyGAPIT, please also cite the original GAPIT papers:

- Wang J., Zhang Z. (2021) GAPIT Version 3. *Genomics, Proteomics & Bioinformatics* https://doi.org/10.1016/j.gpb.2021.08.005
- Huang M. et al. (2019) BLINK. *GigaScience* https://doi.org/10.1093/gigascience/giy154
- Liu X. et al. (2016) FarmCPU. *PLOS Genetics* https://doi.org/10.1371/journal.pgen.1005767
- Kang H.M. et al. (2008) EMMA. *Genetics* 178:1709–1723
- VanRaden P.M. (2009) Kinship. *J. Dairy Sci.* 91:4414–4423

---

## License

GPL-3.0 — consistent with original R GAPIT license.
