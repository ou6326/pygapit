#!/usr/bin/env python3
"""
pyGAPIT Demo — Maize GWAS and Genomic Selection
================================================
Mirrors the GAPIT R tutorial exactly:

    R:  source("gapit_functions.txt")
        myY  <- read.table("mdp_traits.txt",        head=TRUE)
        myGD <- read.table("mdp_numeric.txt",        head=TRUE)
        myGM <- read.table("mdp_SNP_information.txt",head=TRUE)
        myGAPIT <- GAPIT(Y=myY[,c(1,2)], GD=myGD, GM=myGM,
                         model=c("GLM","MLM","FarmCPU","Blink"),
                         PCA.total=3)

Dataset: 281 maize inbred lines, 3093 SNPs, 3 traits (EarHT, dpoll, EarDia).
All output files are written to ./pygapit_demo_output/
"""

import sys
import time
import warnings
from collections.abc import Iterator, Mapping
from os import environ
from pathlib import Path
from typing import Any, Protocol, cast

from numpy import ndarray

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class _SpineLike(Protocol):
    def set_visible(self, visible: bool) -> None: ...


class _AxesLike(Protocol):
    spines: Mapping[str, _SpineLike]

    def scatter(self, *args: Any, **kwargs: Any) -> object: ...
    def plot(self, *args: Any, **kwargs: Any) -> object: ...
    def axhline(self, *args: Any, **kwargs: Any) -> object: ...
    def set_title(self, *args: Any, **kwargs: Any) -> object: ...
    def set_xlabel(self, *args: Any, **kwargs: Any) -> object: ...
    def set_ylabel(self, *args: Any, **kwargs: Any) -> object: ...
    def set_xticks(self, *args: Any, **kwargs: Any) -> object: ...
    def set_xticklabels(self, *args: Any, **kwargs: Any) -> object: ...


class _AxesSequence(Protocol):
    def __getitem__(self, index: int) -> _AxesLike: ...
    def __iter__(self) -> Iterator[_AxesLike]: ...
    def __len__(self) -> int: ...


from pygapit import (
    GAPIT,
    GAPITResult,
    align_taxa,
    blink_gwas,
    bonferroni_threshold,
    build_covariate_matrix,
    compute_pca,
    emma_remle,
    farmcpu_gwas,
    gblup,
    genomic_inflation_factor,
    glm_gwas,
    gs_scatter,
    kinship_heatmap,
    maf_filter,
    mlm_gwas,
    pca_plot_2d,
    read_numeric,
    read_phenotype,
    sblup,
    vanraden_kinship,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(
    environ.get(
        "GAPIT_DATA",
        Path(__file__).resolve().parents[2] / "GAPIT" / "Documents",
    )
)
OUT = Path("pygapit_demo_output")
OUT.mkdir(parents=True, exist_ok=True)

PHENO_FILE = BASE / "mdp_traits.txt"
GD_FILE = BASE / "mdp_numeric.txt"
GM_FILE = BASE / "mdp_SNP_information.txt"

# ── Check data availability ───────────────────────────────────────────────────
for fp in [PHENO_FILE, GD_FILE, GM_FILE]:
    if not fp.exists():
        print(f"ERROR: Demo data not found at {fp}")
        print(
            "Clone GAPIT repo first: git clone https://github.com/jiabowang/GAPIT.git"
        )
        sys.exit(1)

print("=" * 65)
print("  pyGAPIT Demo — Maize GWAS (281 lines × 3093 SNPs)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load and inspect data
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 1: Loading data ──────────────────────────────────")

Y = pd.read_csv(PHENO_FILE, sep="\t")
GD = pd.read_csv(GD_FILE, sep="\t")
GM = pd.read_csv(GM_FILE, sep="\t")

print(
    f"Phenotype:  {Y.shape[0]} individuals, {Y.shape[1] - 1} traits: {Y.columns[1:].tolist()}"
)
print(f"Genotype:   {GD.shape[0]} individuals, {GD.shape[1] - 1} SNPs")
print(
    f"Map:        {GM.shape[0]} SNPs across chromosomes {sorted(GM['Chromosome'].unique())}"
)
print(
    f"Missing phenotype: EarHT={Y['EarHT'].isna().sum()}, "
    f"dpoll={Y['dpoll'].isna().sum()}, EarDia={Y['EarDia'].isna().sum()}"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GAPIT() one-liner — matches R tutorial exactly
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 2: Multi-model GWAS via GAPIT() ─────────────────")
print("Running GLM, MLM, FarmCPU, BLINK on EarHT (mirrors R tutorial)...")

t0 = time.time()
results = GAPIT(
    Y=Y[["Taxa", "EarHT"]],
    GD=GD,
    GM=GM,
    model=["GLM", "MLM", "FarmCPU", "BLINK"],
    PCA_total=3,
    trait="EarHT",
    file_output=True,
    output_dir=OUT,
)
elapsed = time.time() - t0
print(f"All 4 models completed in {elapsed:.1f}s")

print("\nResults summary:")
if isinstance(results, GAPITResult):
    results = {results.model: results}
for r in results.values():
    n_sig = len(r.significant) if r.significant is not None else 0
    qtns = len(r.QTNs) if r.QTNs is not None else "n/a"
    print(
        f"  {r.model:10s}  h²={r.h2:.3f}  λ={r.lambda_gc:.3f}  "
        f"sig_SNPs={n_sig}  QTNs={qtns}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Low-level API — step by step, mimicking R GAPIT internals
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 3: Low-level API (step by step) ─────────────────")

pheno = read_phenotype(PHENO_FILE)
geno = read_numeric(GD_FILE, GM_FILE)
aligned = align_taxa(pheno, geno)

y_full = aligned["Y"]["EarHT"].values.astype(float)
valid = ~np.isnan(y_full)
y = y_full[valid]
GD_arr, kept = maf_filter(aligned["GD"][valid, :], threshold=0.05)
GM_arr = aligned["GM"].iloc[kept].reset_index(drop=True)
taxa = aligned["taxa"][valid]

print(
    f"Working set: n={len(y)} individuals, m={GD_arr.shape[1]} SNPs (after MAF≥0.05 filter)"
)

# Kinship
print("\nComputing VanRaden kinship...")
K = vanraden_kinship(GD_arr)
print(f"  K shape={K.shape}, diagonal mean={np.diag(K).mean():.3f}")

# PCA
print("Computing PCA (k=3)...")
pca = compute_pca(GD_arr, n_components=3)
X0 = build_covariate_matrix(pca, 3)
print(
    f"  Variance explained: PC1={pca.var_explained[0]:.1%}, "
    f"PC2={pca.var_explained[1]:.1%}, PC3={pca.var_explained[2]:.1%}"
)

# REML
print("Estimating variance components (REML)...")
remle = emma_remle(y, X0, K)
print(f"  h²={remle.h2:.4f}, vg={remle.vg:.2f}, ve={remle.ve:.2f}, δ={remle.delta:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GWAS model comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 4: GWAS model comparison ────────────────────────")

chromosomes = GM_arr["Chromosome"].values
positions = GM_arr["Position"].values.astype(float)
snp_names = GM_arr["SNP"].values.astype(str)
thresh_bon = bonferroni_threshold(len(snp_names))

gwas_results: dict[str, ndarray] = {}

# GLM
print("GLM...", end=" ", flush=True)
t0 = time.time()
glm_r = glm_gwas(y, X0, GD_arr)
gwas_results["GLM"] = glm_r.p_values
lam = genomic_inflation_factor(glm_r.p_values)
sig = (glm_r.p_values <= thresh_bon).sum()
print(f"λ={lam:.3f}, {sig} sig SNPs [{time.time() - t0:.1f}s]")

# MLM
print("MLM...", end=" ", flush=True)
t0 = time.time()
mlm_r = mlm_gwas(y, X0, GD_arr, K)
gwas_results["MLM"] = mlm_r.p_values
lam = genomic_inflation_factor(mlm_r.p_values)
sig = (mlm_r.p_values <= thresh_bon).sum()
print(f"λ={lam:.3f}, h²={mlm_r.h2:.3f}, {sig} sig SNPs [{time.time() - t0:.1f}s]")

# FarmCPU
print("FarmCPU...", end=" ", flush=True)
t0 = time.time()
farm_r = farmcpu_gwas(
    y, X0, GD_arr, chromosomes=chromosomes, positions=positions, max_iterations=10
)
gwas_results["FarmCPU"] = farm_r.p_values
non_qtn = np.ones(len(farm_r.p_values), dtype=bool)
if len(farm_r.selected_qtns) > 0:
    non_qtn[farm_r.selected_qtns] = False
lam = genomic_inflation_factor(farm_r.p_values[non_qtn])
sig = (farm_r.p_values <= thresh_bon).sum()
print(
    f"λ={lam:.3f}, {len(farm_r.selected_qtns)} QTNs, {sig} sig [{time.time() - t0:.1f}s]"
)

# BLINK
print("BLINK...", end=" ", flush=True)
t0 = time.time()
blink_r = blink_gwas(y, X0, GD_arr, max_iterations=10, ld_threshold=0.7)
gwas_results["BLINK"] = blink_r.p_values
non_qtn_b = np.ones(len(blink_r.p_values), dtype=bool)
if len(blink_r.selected_qtns) > 0:
    non_qtn_b[blink_r.selected_qtns] = False
lam = genomic_inflation_factor(blink_r.p_values[non_qtn_b])
sig = (blink_r.p_values <= thresh_bon).sum()
print(
    f"λ={lam:.3f}, {len(blink_r.selected_qtns)} QTNs, {sig} sig [{time.time() - t0:.1f}s]"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Genomic selection
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 5: Genomic selection (gBLUP) ────────────────────")

print("Running gBLUP...", end=" ", flush=True)
t0 = time.time()
gs = gblup(y, X0, K, taxa=taxa)
r_acc = np.corrcoef(y, gs.prediction)[0, 1]
print(f"h²={gs.h2:.3f}, r={r_acc:.4f} [{time.time() - t0:.1f}s]")

print("\nTop 5 individuals by GEBV:")
top5 = np.argsort(gs.gebv)[-5:][::-1]
for i in top5:
    print(
        f"  {taxa[i]:12s}  GEBV={gs.gebv[i]:+7.2f}  PEV={gs.pev[i]:.1f}  "
        f"Pred={gs.prediction[i]:.1f}"
    )

print("\nsBLUP (using BLINK QTNs)...", end=" ", flush=True)
t0 = time.time()
gs_s = sblup(y, X0, GD_arr, qtn_indices=blink_r.selected_qtns, taxa=taxa)
r_acc_s = np.corrcoef(y, gs_s.prediction)[0, 1]
print(f"h²={gs_s.h2:.3f}, r={r_acc_s:.4f} [{time.time() - t0:.1f}s]")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Visualizations
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 6: Generating plots ─────────────────────────────")

# Manhattan plots for all models
fig, raw_axes = plt.subplots(4, 1, figsize=(14, 18))
axes = cast(_AxesSequence, raw_axes)
for ax, (model_name, pvals) in zip(axes, gwas_results.items()):
    valid_p = np.where((pvals > 0) & ~np.isnan(pvals), pvals, 1.0)
    log_p = -np.log10(valid_p)
    chroms = np.array([str(c) for c in chromosomes])
    unique_chr: list[str] = []
    seen: set[str] = set()
    for c in chroms:
        if c not in seen:
            unique_chr.append(c)
            seen.add(c)
    offset: dict[str, float] = {}
    x_vals = np.zeros(len(pvals))
    chr_centers: dict[str, float] = {}
    cum = 0
    for ch in unique_chr:
        mask = chroms == ch
        mx = positions[mask].max() if mask.any() else 0
        offset[ch] = cum
        chr_centers[ch] = cum + mx / 2
        cum += mx + 5_000_000
    for i in range(len(pvals)):
        x_vals[i] = positions[i] + offset.get(chroms[i], 0)
    colors = [
        "#3C5587" if i % 2 == 0 else "#89A8D0"
        for i, ch in enumerate(unique_chr)
        for _ in np.where(chroms == ch)[0]
    ]
    col_arr = np.empty(len(pvals), dtype=object)
    for i, ch in enumerate(unique_chr):
        col_arr[chroms == ch] = "#3C5587" if i % 2 == 0 else "#89A8D0"
    ax.scatter(x_vals, log_p, c=col_arr, s=1.2, linewidths=0, rasterized=True)
    sig_line = -np.log10(thresh_bon)
    ax.axhline(sig_line, color="#E41A1C", linestyle="--", linewidth=0.8)
    ax.set_xticks([chr_centers[c] for c in unique_chr])
    ax.set_xticklabels(unique_chr, fontsize=7)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=9)
    ax.set_title(
        f"{model_name}  (λ={genomic_inflation_factor(pvals):.3f})",
        fontsize=10,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
cast(Any, fig).savefig(
    f"{OUT}/demo_manhattan_all_models.pdf", dpi=120, bbox_inches="tight"
)
plt.close()
print(f"  Saved: {OUT}/demo_manhattan_all_models.pdf")

# QQ plots
fig, raw_axes = plt.subplots(1, 4, figsize=(18, 4))
axes = cast(_AxesSequence, raw_axes)
for ax, (model_name, pvals) in zip(axes, gwas_results.items()):
    valid = pvals[(pvals > 0) & ~np.isnan(pvals)]
    n = len(valid)
    expected = -np.log10(np.arange(1, n + 1) / n)  # pyright: ignore[reportOperatorIssue]
    observed = -np.log10(np.sort(valid)[::-1])
    max_v = max(observed.max(), expected.max()) * 1.1
    ax.plot([0, max_v], [0, max_v], "k--", lw=0.8, alpha=0.6)
    ax.scatter(
        np.sort(expected)[::-1], observed, c="#3C5587", s=3, alpha=0.6, linewidths=0
    )
    lam = genomic_inflation_factor(pvals)
    ax.set_title(f"{model_name} (λ={lam:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Expected", fontsize=8)
    ax.set_ylabel("Observed", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
cast(Any, fig).savefig(f"{OUT}/demo_qq_all_models.pdf", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/demo_qq_all_models.pdf")

# Kinship heatmap
fig = kinship_heatmap(
    K[:50, :50],
    taxa=taxa[:50],
    title="Kinship (first 50 lines)",
    save_path=f"{OUT}/demo_kinship.pdf",
)
plt.close(fig)
print(f"  Saved: {OUT}/demo_kinship.pdf")

# PCA
fig = pca_plot_2d(
    pca.scores,
    pca.var_explained,
    title="PCA — Maize inbred lines",
    save_path=f"{OUT}/demo_pca.pdf",
)
plt.close(fig)
print(f"  Saved: {OUT}/demo_pca.pdf")

# GS scatter
fig = gs_scatter(
    y, gs.prediction, trait_name="EarHT", save_path=f"{OUT}/demo_gs_scatter.pdf"
)
plt.close(fig)
print(f"  Saved: {OUT}/demo_gs_scatter.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Simulation study (power comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 7: Simulation (h²=0.7, 20 QTNs) ─────────────────")
print("Simulating phenotype from genotype data...")

np.random.seed(198521)  # same seed as GAPIT's demo
m_total = GD_arr.shape[1]
qtn_idx = np.random.choice(m_total, size=20, replace=False)
alpha_q = np.random.normal(0, 1, 20)

GD_std = (GD_arr[:, qtn_idx] - GD_arr[:, qtn_idx].mean(0)) / (
    GD_arr[:, qtn_idx].std(0) + 1e-8
)
g = GD_std @ alpha_q
g_var = g.var()
e_var = g_var * (1 - 0.7) / 0.7
y_sim = g + np.random.normal(0, np.sqrt(e_var), len(y))

print("  True QTN positions (chr:pos): ", end="")
for idx in qtn_idx[:5]:
    snp = GM_arr.iloc[idx]
    print(f"chr{snp['Chromosome']}:{int(snp['Position']):,}", end=" ")
print("...")

# Quick GLM and BLINK on simulated phenotype
print("  Running GLM on simulated phenotype...", end=" ", flush=True)
glm_sim = glm_gwas(y_sim, X0, GD_arr)
top_glm = np.argsort(glm_sim.p_values)[:10]
overlap_glm = len(set(top_glm) & set(qtn_idx))
print(f"Top-10 overlap with true QTNs: {overlap_glm}/10")

print("  Running BLINK on simulated phenotype...", end=" ", flush=True)
blink_sim = blink_gwas(y_sim, X0, GD_arr, max_iterations=10)
overlap_blink = len(set(blink_sim.selected_qtns) & set(qtn_idx))
total_selected = len(blink_sim.selected_qtns)
print(
    f"QTN overlap: {overlap_blink}/{min(total_selected, 20)} "
    f"({total_selected} total selected)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Demo complete!")
print("=" * 65)
print(f"\nOutput files written to: {OUT.resolve()}/")
print("\nKey results (EarHT trait):")
print(f"  Heritability (h²):     {remle.h2:.4f}")
print(
    f"  Genomic inflation (λ): {genomic_inflation_factor(glm_r.p_values):.3f} [GLM]  "
    f"{genomic_inflation_factor(mlm_r.p_values):.3f} [MLM]"
)
print(f"  BLINK QTNs selected:   {len(blink_r.selected_qtns)}")
print(f"  gBLUP prediction r:    {r_acc:.4f}")
