"""
Visualization module.
Translates GAPIT.Manhattan.R, GAPIT.QQ.R, GAPIT.PCA.R,
GAPIT.GS.Visualization.R, GAPIT.Phenotype.View.R

All plots are publication-ready and match GAPIT's visual style.
Static plots use matplotlib/seaborn.
Interactive plots use plotly (same package as GAPIT's plotly R).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Protocol, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")

from importlib.util import find_spec

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .._typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    LabelVector,
    NumericVector,
    StrVector,
    Vector,
)


class _Spine(Protocol):
    def set_visible(self, visible: bool) -> None: ...


class _Axes(Protocol):
    spines: Mapping[str, _Spine]

    def scatter(self, *args: Any, **kwargs: Any) -> object: ...
    def plot(self, *args: Any, **kwargs: Any) -> object: ...
    def hist(self, *args: Any, **kwargs: Any) -> object: ...
    def imshow(self, *args: Any, **kwargs: Any) -> object: ...
    def fill_between(self, *args: Any, **kwargs: Any) -> object: ...
    def legend(self, *args: Any, **kwargs: Any) -> object: ...
    def axhline(self, *args: Any, **kwargs: Any) -> object: ...
    def axvline(self, *args: Any, **kwargs: Any) -> object: ...
    def set_facecolor(self, *args: Any, **kwargs: Any) -> object: ...
    def set_title(self, *args: Any, **kwargs: Any) -> object: ...
    def set_xlabel(self, *args: Any, **kwargs: Any) -> object: ...
    def set_ylabel(self, *args: Any, **kwargs: Any) -> object: ...
    def set_xlim(self, *args: Any, **kwargs: Any) -> object: ...
    def set_ylim(self, *args: Any, **kwargs: Any) -> object: ...
    def set_xticks(self, *args: Any, **kwargs: Any) -> object: ...
    def set_yticks(self, *args: Any, **kwargs: Any) -> object: ...
    def set_xticklabels(self, *args: Any, **kwargs: Any) -> object: ...
    def set_yticklabels(self, *args: Any, **kwargs: Any) -> object: ...


class _FigureWriter(Protocol):
    def savefig(self, *args: Any, **kwargs: Any) -> None: ...


class _PlotlyFigure(Protocol):
    def add_trace(self, *args: Any, **kwargs: Any) -> object: ...
    def add_hline(self, *args: Any, **kwargs: Any) -> object: ...
    def update_layout(self, *args: Any, **kwargs: Any) -> object: ...
    def write_html(self, *args: Any, **kwargs: Any) -> None: ...


class _PlotlyModule(Protocol):
    def Figure(self, *args: Any, **kwargs: Any) -> _PlotlyFigure: ...
    def Scatter(self, *args: Any, **kwargs: Any) -> object: ...
    def Scatter3d(self, *args: Any, **kwargs: Any) -> object: ...


def _axes(value: object) -> _Axes:
    return cast(_Axes, value)


def _savefig(figure: Figure, *args: Any, **kwargs: Any) -> None:
    cast(_FigureWriter, figure).savefig(*args, **kwargs)


HAS_SEABORN = find_spec("seaborn") is not None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

HAS_PLOTLY = go is not None


# ── Color palette matching GAPIT's default ───────────────────────────────
CHR_COLORS = [
    "#3C5587",
    "#89A8D0",  # alternating blue shades for chromosomes
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
    "#3C5587",
    "#89A8D0",
]
SIG_COLOR = "#E41A1C"  # red for significant hits
SUGGEST_COLOR = "#FF7F00"  # orange for suggestive


def _genomic_axis(
    chromosomes: LabelVector,
    positions: NumericVector,
    chromosome_gap: float = 5_000_000.0,
) -> tuple[FloatVector, tuple[str, ...], FloatVector]:
    """Map chromosome-local positions onto one cumulative genomic axis."""
    chroms = np.asarray(chromosomes, dtype=str)
    pos = np.asarray(positions, dtype=np.float64)
    if chroms.ndim != 1 or pos.ndim != 1 or len(chroms) != len(pos):
        raise ValueError("chromosomes and positions must be equal-length vectors")
    if len(pos) == 0:
        raise ValueError("chromosomes and positions must not be empty")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must contain only finite values")

    unique_chroms = tuple(dict.fromkeys(str(chrom) for chrom in chroms))
    x_values = np.empty(len(pos), dtype=np.float64)
    centers = np.empty(len(unique_chroms), dtype=np.float64)
    cumulative = 0.0
    for index, chrom in enumerate(unique_chroms):
        mask = chroms == chrom
        chrom_positions = pos[mask]
        minimum = float(np.min(chrom_positions))
        maximum = float(np.max(chrom_positions))
        span = maximum - minimum
        x_values[mask] = cumulative + chrom_positions - minimum
        centers[index] = cumulative + span / 2.0
        cumulative += span + chromosome_gap
    return x_values, unique_chroms, centers


def manhattan_plot(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    title: str = "Manhattan Plot",
    significance_threshold: float | None = None,
    suggestive_threshold: float | None = None,
    highlight_snps: IntVector | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 5),
    point_size: float = 1.5,
) -> Figure:
    """
    Manhattan plot.
    Translates GAPIT.Manhattan.R

    Parameters
    ----------
    snp_names            : SNP identifiers
    chromosomes          : chromosome labels
    positions            : genomic positions (bp)
    p_values             : association p-values
    significance_threshold : genome-wide significance line (default: Bonferroni)
    suggestive_threshold : suggestive line (default: 1e-5)
    highlight_snps       : indices of SNPs to highlight red
    save_path            : if provided, save to this path
    """
    # ── Data prep ────────────────────────────────────────────────────────
    positions = np.asarray(positions, dtype=np.float64)
    p_values = np.asarray(p_values, dtype=np.float64)
    valid = ~np.isnan(p_values) & (p_values > 0) & (p_values <= 1)
    p_vals = np.where(valid, p_values, 1.0)
    log_p = -np.log10(np.where(p_vals > 0, p_vals, 1e-300))

    m = len(p_values)
    if significance_threshold is None:
        significance_threshold = 0.05 / m
    if suggestive_threshold is None:
        suggestive_threshold = 1.0 / m

    sig_line = -np.log10(significance_threshold)
    sug_line = -np.log10(suggestive_threshold)

    chroms = np.asarray(chromosomes, dtype=str)
    x_vals, unique_chroms, chrom_centers = _genomic_axis(chroms, positions)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, raw_ax = plt.subplots(figsize=figsize)
    ax = _axes(raw_ax)
    ax.set_facecolor("white")

    for ci, chrom in enumerate(unique_chroms):
        mask = chroms == chrom
        color = CHR_COLORS[ci % len(CHR_COLORS)]
        ax.scatter(
            x_vals[mask],
            log_p[mask],
            c=color,
            s=point_size,
            linewidths=0,
            rasterized=True,
            alpha=0.8,
        )

    # Highlight significant SNPs
    if highlight_snps is not None and len(highlight_snps) > 0:
        ax.scatter(
            x_vals[highlight_snps],
            log_p[highlight_snps],
            c=SIG_COLOR,
            s=point_size * 4,
            linewidths=0,
            zorder=5,
        )

    # Threshold lines
    ax.axhline(y=sig_line, color=SIG_COLOR, linestyle="--", linewidth=0.8, alpha=0.9)
    ax.axhline(
        y=sug_line, color=SUGGEST_COLOR, linestyle="--", linewidth=0.6, alpha=0.7
    )

    # Axis formatting
    ax.set_xlim(0, max(float(x_vals.max()) * 1.01, 1.0))
    ax.set_ylim(0, max(log_p.max() * 1.1, sig_line * 1.2))
    ax.set_xticks(chrom_centers)
    ax.set_xticklabels(unique_chroms, fontsize=7)
    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=150, bbox_inches="tight")

    return fig


def qq_plot(
    p_values: FloatVector,
    title: str = "QQ Plot",
    save_path: str | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> Figure:
    """
    Quantile-Quantile plot with genomic inflation factor.
    Translates GAPIT.QQ.R

    Diagonal = expected under null hypothesis (no association).
    Deviation upward at right tail = true associations.
    Uniform upward deviation = population stratification (λ > 1).
    """
    from ..stats.testing import genomic_inflation_factor

    valid = ~np.isnan(p_values) & (p_values > 0) & (p_values <= 1)
    p_obs = np.sort(p_values[valid])
    n = len(p_obs)

    if n == 0:
        fig, raw_ax = plt.subplots(figsize=figsize)
        ax = _axes(raw_ax)
        ax.set_title("No valid p-values")
        return fig

    # Expected quantiles
    expected = -np.log10(np.arange(1, n + 1) / n)
    observed = -np.log10(p_obs[::-1])

    # Lambda
    lam = genomic_inflation_factor(p_values)

    fig, raw_ax = plt.subplots(figsize=figsize)
    ax = _axes(raw_ax)

    # Confidence band
    ax.fill_between(
        np.sort(expected)[::-1],
        np.sort(expected)[::-1],
        np.sort(expected)[::-1] * 1.3,
        alpha=0.15,
        color="gray",
    )

    # Diagonal
    max_val = max(observed.max(), expected.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.7)

    # Points
    ax.scatter(
        np.sort(expected)[::-1],
        observed,
        c="#3C5587",
        s=4,
        linewidths=0,
        alpha=0.7,
    )

    ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=10)
    ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=10)
    ax.set_title(f"{title}\n(λ = {lam:.3f})", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=150, bbox_inches="tight")

    return fig


def kinship_heatmap(
    K: FloatMatrix,
    taxa: Vector | None = None,
    title: str = "Kinship Matrix",
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 7),
) -> Figure:
    """
    Heatmap of genomic kinship matrix.
    Translates GAPIT.Genotype.View.R (kinship heatmap section)

    Color scale: blue=low kinship, red=high kinship
    Sorted by hierarchical clustering (like GAPIT's heatmap.2)
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = K.shape[0]
    dist = np.clip(1.0 - K, 0, None)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    try:
        condensed = squareform(dist)
        Z = linkage(condensed, method="average")
        order = np.asarray(leaves_list(Z), dtype=int)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        order = np.array(list(range(n)), dtype=int)

    K_sorted = K[np.ix_(order, order)]

    fig, raw_ax = plt.subplots(figsize=figsize)
    ax = _axes(raw_ax)
    im = ax.imshow(K_sorted, aspect="auto", cmap="RdBu_r", vmin=K.min(), vmax=K.max())
    cast(Any, plt).colorbar(im, ax=raw_ax, shrink=0.8, label="Kinship")

    if taxa is not None and n <= 50:
        taxa_sorted = np.array(taxa)[order]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(taxa_sorted, rotation=90, fontsize=6)
        ax.set_yticklabels(taxa_sorted, fontsize=6)
    else:
        ax.set_xlabel("Individuals", fontsize=10)
        ax.set_ylabel("Individuals", fontsize=10)

    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=150, bbox_inches="tight")

    return fig


def pca_plot_2d(
    scores: FloatMatrix,
    var_explained: FloatVector,
    taxa: Vector | None = None,
    groups: Vector | None = None,
    title: str = "PCA Plot",
    save_path: str | None = None,
    figsize: tuple[float, float] = (7, 6),
) -> Figure:
    """
    2D PCA scatter plot (PC1 vs PC2).
    Translates GAPIT.PCA.R static plot.
    """
    fig, raw_ax = plt.subplots(figsize=figsize)
    ax = _axes(raw_ax)

    pc1, pc2 = scores[:, 0], scores[:, 1]

    if groups is not None:
        unique_groups = np.unique(groups)
        palette = cast(Any, plt).cm.tab10(np.linspace(0, 0.9, len(unique_groups)))
        for gi, g in enumerate(unique_groups):
            mask = groups == g
            ax.scatter(
                pc1[mask],
                pc2[mask],
                s=15,
                alpha=0.8,
                color=palette[gi],
                label=str(g),
                linewidths=0,
            )
        ax.legend(fontsize=8, markerscale=2, framealpha=0.5)
    else:
        ax.scatter(pc1, pc2, s=10, alpha=0.7, color="#3C5587", linewidths=0)

    pct1 = var_explained[0] * 100 if len(var_explained) > 0 else 0
    pct2 = var_explained[1] * 100 if len(var_explained) > 1 else 0
    ax.set_xlabel(f"PC1 ({pct1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pct2:.1f}%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=150, bbox_inches="tight")

    return fig


def pca_plot_3d_interactive(
    scores: FloatMatrix,
    var_explained: FloatVector,
    taxa: Vector | None = None,
    groups: Vector | None = None,
    title: str = "3D PCA",
    save_path: str | None = None,
) -> object:
    """
    Interactive 3D PCA using Plotly.
    Translates GAPIT.3D.PCA.python.R — same plotly calls, Python syntax.
    Identical to GAPIT's interactive HTML output.
    """
    if go is None:
        warnings.warn("plotly not installed; skipping 3D interactive PCA.")
        return None
    plotly = cast(_PlotlyModule, cast(Any, go))

    pc1 = scores[:, 0] if scores.shape[1] > 0 else np.zeros(len(scores))
    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
    pc3 = scores[:, 2] if scores.shape[1] > 2 else np.zeros(len(scores))

    pct = [v * 100 for v in var_explained[:3]] if len(var_explained) >= 3 else [0, 0, 0]

    hover_text = (
        taxa.tolist() if taxa is not None else [str(i) for i in range(len(pc1))]
    )

    fig = plotly.Figure(
        data=[
            plotly.Scatter3d(
                x=pc1,
                y=pc2,
                z=pc3,
                mode="markers",
                marker={
                    "size": 4,
                    "color": pc1,
                    "colorscale": "Viridis",
                    "opacity": 0.85,
                },
                text=hover_text,
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": f"PC1 ({pct[0]:.1f}%)",
            "yaxis_title": f"PC2 ({pct[1]:.1f}%)",
            "zaxis_title": f"PC3 ({pct[2]:.1f}%)",
        },
        width=700,
        height=600,
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def manhattan_interactive(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    effects: NumericVector | None = None,
    maf: NumericVector | None = None,
    title: str = "Interactive Manhattan",
    save_path: str | None = None,
) -> object:
    """
    Interactive Manhattan plot with hover info.
    Translates GAPIT.Interactive.Manhattan.R
    Hover shows: SNP name, chromosome, position, MAF, p-value, effect.
    """
    if go is None:
        warnings.warn("plotly not installed; skipping interactive Manhattan.")
        return None
    plotly = cast(_PlotlyModule, cast(Any, go))

    snp_names = np.asarray(snp_names, dtype=str)
    chromosomes = np.asarray(chromosomes, dtype=str)
    positions = np.asarray(positions, dtype=np.float64)
    p_values = np.asarray(p_values, dtype=np.float64)
    valid = ~np.isnan(p_values) & (p_values > 0)
    m = len(p_values)

    chroms = chromosomes
    x_vals, unique_chroms, _ = _genomic_axis(chroms, positions)
    log_p = -np.log10(np.where(valid, np.maximum(p_values, 1e-300), 1.0))

    # Build hover text
    hover: list[str] = []
    for i in range(m):
        txt = (
            f"<b>{snp_names[i]}</b><br>"
            f"Chr: {chromosomes[i]}, Pos: {int(positions[i]):,}<br>"
            f"P-value: {p_values[i]:.2e}<br>"
        )
        if effects is not None:
            txt += f"Effect: {effects[i]:.4f}<br>"
        if maf is not None:
            txt += f"MAF: {maf[i]:.3f}"
        hover.append(txt)

    sig_threshold = 0.05 / m
    sig_line = -np.log10(sig_threshold)

    fig = plotly.Figure()
    for ci, chrom in enumerate(unique_chroms):
        mask = chroms == chrom
        color = CHR_COLORS[ci % len(CHR_COLORS)]
        fig.add_trace(
            plotly.Scatter(
                x=x_vals[mask],
                y=log_p[mask],
                mode="markers",
                marker={"size": 3, "color": color, "opacity": 0.7},
                text=np.array(hover)[mask],
                hovertemplate="%{text}<extra></extra>",
                name=f"Chr {chrom}",
                showlegend=False,
            )
        )

    fig.add_hline(
        y=sig_line,
        line_dash="dash",
        line_color=SIG_COLOR,
        annotation_text="Bonferroni",
        annotation_position="right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Genomic Position",
        yaxis_title="-log₁₀(p)",
        hovermode="closest",
        width=900,
        height=400,
        plot_bgcolor="white",
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def gs_scatter(
    observed: NumericVector,
    predicted: NumericVector,
    taxa: Vector | None = None,
    trait_name: str = "Trait",
    save_path: str | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> Figure:
    """
    Genomic Selection scatter: predicted vs observed.
    Translates GAPIT.GS.Visualization.R
    Pearson r = prediction accuracy.
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = ~(np.isnan(observed) | np.isnan(predicted))
    obs_v = observed[valid]
    pred_v = predicted[valid]

    if len(obs_v) < 2:
        fig, raw_ax = plt.subplots(figsize=figsize)
        ax = _axes(raw_ax)
        ax.set_title("Insufficient data for GS scatter")
        return fig

    r = np.corrcoef(obs_v, pred_v)[0, 1]

    fig, raw_ax = plt.subplots(figsize=figsize)
    ax = _axes(raw_ax)
    ax.scatter(obs_v, pred_v, s=15, alpha=0.6, color="#3C5587", linewidths=0)

    # Regression line
    m_coef = np.asarray(np.polyfit(obs_v, pred_v, 1), dtype=np.float64)
    x_min, x_max = float(obs_v.min()), float(obs_v.max())
    x_line = np.array(
        [x_min + (x_max - x_min) * i / 99 for i in range(100)], dtype=np.float64
    )
    y_line = np.asarray(m_coef[0] * x_line + m_coef[1], dtype=np.float64)
    ax.plot(x_line, y_line, "r-", linewidth=1.2, alpha=0.8)

    ax.set_xlabel(f"Observed {trait_name}", fontsize=10)
    ax.set_ylabel(f"Predicted {trait_name}", fontsize=10)
    ax.set_title(f"GS Accuracy (r = {r:.3f})", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=150, bbox_inches="tight")

    return fig


def phenotype_distribution(
    y: FloatVector,
    trait_name: str = "Trait",
    significant_snp_geno: Vector | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (6, 4),
) -> Figure:
    """
    Phenotype distribution histogram.
    Translates GAPIT.Phenotype.View.R
    Optionally split by genotype at top significant SNP.
    """
    fig, raw_ax = plt.subplots(figsize=figsize)
    ax = _axes(raw_ax)
    valid_y = y[~np.isnan(y)]

    ax.hist(
        valid_y, bins=30, color="#3C5587", alpha=0.75, edgecolor="white", linewidth=0.3
    )

    if significant_snp_geno is not None:
        for geno_val, label, color in [
            (0, "Ref/Ref", "#2166AC"),
            (1, "Ref/Alt", "#74ADD1"),
            (2, "Alt/Alt", "#D73027"),
        ]:
            mask = (significant_snp_geno == geno_val) & ~np.isnan(y)
            if mask.any():
                ax.hist(
                    y[mask],
                    bins=20,
                    alpha=0.5,
                    color=color,
                    label=label,
                    edgecolor="none",
                )
        ax.legend(fontsize=8)

    ax.set_xlabel(trait_name, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Distribution of {trait_name}", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=150, bbox_inches="tight")

    return fig
