"""
Visualization module.
Translates GAPIT.Manhattan.R, GAPIT.QQ.R, GAPIT.PCA.R,
GAPIT.GS.Visualization.R, GAPIT.Phenotype.View.R

All plots are publication-ready and match GAPIT's visual style.
Plots are returned as backend-neutral HoloViews objects. Callers choose a
renderer with ``holoviews.render`` or an output format with ``holoviews.save``.
"""

from __future__ import annotations

import typing as t
from functools import cache
from typing import Literal, Protocol, TypeVar, cast

import datashader as ds
import holoviews as hv
import numpy as np
from holoviews import opts as hv_opts
from holoviews.core import Dimensioned
from holoviews.operation.datashader import rasterize

if t.TYPE_CHECKING:
    from typing import Self

    from datashader.reductions import Reduction
    from holoviews.core.options import Options

from .._typing import (
    BoolVector,
    FloatMatrix,
    FloatVector,
    IntVector,
    LabelVector,
    NumericVector,
    StrVector,
    Vector,
    as_float_vector,
    require_length,
)
from .data import ManhattanPlotData, prepare_genomic_axis, prepare_manhattan_data

LargeDataMode = Literal["auto", "points", "aggregate"]

_AGGREGATE_MARKER_THRESHOLD = 250_000

_RasterizeInputT_contra = TypeVar("_RasterizeInputT_contra", contravariant=True)
_RasterizeOutputT_co = TypeVar("_RasterizeOutputT_co", covariant=True)


class _HoloViewsObject(Protocol):
    def __mul__(self, other: _HoloViewsObject) -> _HoloViewsObject: ...
    def collate(self) -> _HoloViewsObject: ...
    def dimensions(self) -> list[_HoloViewsDimension]: ...
    def opts(self, *options: Options) -> Self: ...


class _HoloViewsDimension(Protocol):
    name: str


class _HoloViewsStore(Protocol):
    current_backend: str

    def set_current_backend(self, backend: str) -> None: ...


class _HoloViewsModule(Protocol):
    Store: _HoloViewsStore

    def extension(self, *backends: str) -> None: ...

    def Points(
        self,
        data: tuple[Vector, ...],
        *,
        kdims: list[str],
        vdims: list[str],
    ) -> _HoloViewsObject: ...

    def Scatter(
        self,
        data: tuple[Vector, ...],
        *,
        kdims: list[str],
        vdims: list[str],
        label: str = "",
    ) -> _HoloViewsObject: ...

    def Curve(
        self,
        data: tuple[Vector, ...],
        *,
        kdims: list[str],
        vdims: list[str],
        label: str = "",
    ) -> _HoloViewsObject: ...

    def Area(
        self,
        data: tuple[Vector, ...],
        *,
        kdims: list[str],
        vdims: list[str],
    ) -> _HoloViewsObject: ...

    def Image(
        self,
        data: FloatMatrix,
        *,
        bounds: tuple[float, float, float, float],
        kdims: list[str],
        vdims: list[str],
    ) -> _HoloViewsObject: ...

    def Histogram(
        self,
        data: tuple[FloatVector, FloatVector],
        *,
        kdims: list[str],
        vdims: list[str],
        label: str = "",
    ) -> _HoloViewsObject: ...

    def Scatter3D(
        self,
        data: tuple[Vector, ...],
        *,
        kdims: list[str],
        vdims: list[str],
    ) -> _HoloViewsObject: ...

    def Overlay(self, items: list[_HoloViewsObject]) -> _HoloViewsObject: ...
    def HLine(self, y: float) -> _HoloViewsObject: ...
    def VLine(self, x: float) -> _HoloViewsObject: ...


class _HoloViewsOptions(Protocol):
    def Points(self, **kwargs: object) -> Options: ...
    def Scatter(self, **kwargs: object) -> Options: ...
    def Curve(self, **kwargs: object) -> Options: ...
    def Area(self, **kwargs: object) -> Options: ...
    def Image(self, **kwargs: object) -> Options: ...
    def Histogram(self, **kwargs: object) -> Options: ...
    def Overlay(self, **kwargs: object) -> Options: ...
    def HLine(self, **kwargs: object) -> Options: ...
    def VLine(self, **kwargs: object) -> Options: ...
    def Scatter3D(self, **kwargs: object) -> Options: ...


class _DatashaderModule(Protocol):
    def max(self, column: str) -> Reduction: ...


class _Rasterize(Protocol[_RasterizeInputT_contra, _RasterizeOutputT_co]):
    def __call__(
        self,
        obj: _RasterizeInputT_contra,
        *,
        aggregator: Reduction,
        width: int,
        height: int,
        precompute: bool,
    ) -> _RasterizeOutputT_co: ...


def _runtime_object(value: object) -> object:
    """Cross a third-party stub boundary without weakening the result type."""
    return value


def _as_dimensioned(value: object) -> Dimensioned:
    """Narrow a HoloViews object across its incomplete typing boundary."""
    return cast(Dimensioned, value)


def _use_aggregate(marker_count: int, large_data: LargeDataMode) -> bool:
    match large_data:
        case "auto":
            return marker_count >= _AGGREGATE_MARKER_THRESHOLD
        case "points":
            return False
        case "aggregate":
            return True
    raise ValueError("large_data must be 'auto', 'points', or 'aggregate'")


def _validate_plot_geometry(
    figsize: tuple[float, ...],
    point_size: float,
) -> None:
    if len(figsize) != 2:
        raise ValueError("figsize must contain exactly two dimensions")
    if not all(np.isfinite(dimension) and dimension > 0.0 for dimension in figsize):
        raise ValueError("figsize dimensions must be finite and positive")
    if not np.isfinite(point_size) or point_size <= 0.0:
        raise ValueError("point_size must be finite and positive")


# ── Color palette matching GAPIT's default ───────────────────────────────
CHR_COLORS = (
    "#3C5587",
    "#89A8D0",  # alternating blue shades for chromosomes
)
SIG_COLOR = "#E41A1C"  # red for significant hits
SUGGEST_COLOR = "#FF7F00"  # orange for suggestive


def manhattan(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    *,
    title: str = "Manhattan Plot",
    significance_threshold: float | None = None,
    suggestive_threshold: float | None = None,
    highlight_snps: IntVector | None = None,
    effects: NumericVector | None = None,
    maf: NumericVector | None = None,
    figsize: tuple[float, float] = (14, 5),
    point_size: float = 1.5,
    large_data: LargeDataMode = "auto",
) -> Dimensioned:
    """Build a backend-neutral HoloViews Manhattan plot.

    Use :func:`holoviews.render` to obtain a backend figure or
    :func:`holoviews.save` to write a static or interactive artifact.
    """
    data = prepare_manhattan_data(
        snp_names,
        chromosomes,
        positions,
        p_values,
        significance_threshold=significance_threshold,
        suggestive_threshold=suggestive_threshold,
    )
    _validate_plot_geometry(figsize, point_size)
    return _as_dimensioned(
        _build_manhattan_plot(
            data,
            aggregate=_use_aggregate(len(data.p_values), large_data),
            title=title,
            highlight_snps=highlight_snps,
            effects=effects,
            maf=maf,
            figsize=figsize,
            point_size=point_size,
        ),
    )


def _build_manhattan_plot(
    data: ManhattanPlotData,
    *,
    aggregate: bool,
    title: str,
    highlight_snps: IntVector | None,
    effects: NumericVector | None,
    maf: NumericVector | None,
    figsize: tuple[float, float],
    point_size: float,
) -> _HoloViewsObject:
    """Build one HoloViews graph that can be rendered by any loaded backend."""
    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))
    effect_values, maf_values = _interactive_manhattan_metadata(data, effects, maf)
    layers: list[_HoloViewsObject] = []

    for chromosome_index, chromosome in enumerate(data.chromosome_labels):
        mask = data.chromosomes == chromosome
        color = CHR_COLORS[chromosome_index % len(CHR_COLORS)]
        points = _manhattan_points(
            holoviews,
            options,
            data,
            mask,
            effect_values=effect_values,
            maf_values=maf_values,
            color=color,
            point_size=point_size,
        )
        layers.append(
            _rasterize_manhattan_layer(
                points,
                options,
                color,
                figsize,
            )
            if aggregate
            else points
        )

    plot = holoviews.Overlay(layers)
    if aggregate:
        # ``rasterize`` returns DynamicMaps.  Collating them lifts the dynamic
        # operation above the Overlay, which is the composition HoloViews can
        # update correctly when an interactive viewport changes.
        plot = plot.collate()
    emphasized = (
        data.p_values <= data.significance_threshold
        if aggregate
        else np.zeros(len(data.p_values), dtype=np.bool_)
    )
    if highlight_snps is not None:
        emphasized = emphasized.copy()
        emphasized[highlight_snps] = True
    if emphasized.any():
        plot *= _manhattan_points(
            holoviews,
            options,
            data,
            emphasized,
            effect_values=effect_values,
            maf_values=maf_values,
            color=SIG_COLOR,
            point_size=point_size * 2,
        )

    plot *= _threshold_line(
        holoviews,
        options,
        -np.log10(data.significance_threshold),
        color=SIG_COLOR,
        width=0.8,
    )
    plot *= _threshold_line(
        holoviews,
        options,
        -np.log10(data.suggestive_threshold),
        color=SUGGEST_COLOR,
        width=0.6,
    )
    x_limit = max(data.x_values.max() * 1.01, 1.0)
    y_limit = max(
        data.log_p_values.max() * 1.1,
        -np.log10(data.significance_threshold) * 1.2,
    )
    ticks = list(
        zip(
            data.chromosome_centers.tolist(),
            data.chromosome_labels,
            strict=True,
        )
    )
    common = {
        "title": title,
        "xlabel": "Chromosome",
        "ylabel": "-log10(p)",
        "xlim": (0.0, x_limit),
        "ylim": (0.0, y_limit),
        "xticks": ticks,
    }
    return plot.opts(
        options.Overlay(
            backend="matplotlib",
            fig_inches=figsize,
            show_frame=False,
            **common,
        ),
        options.Overlay(
            backend="bokeh",
            width=max(int(figsize[0] * 100), 1),
            height=max(int(figsize[1] * 100), 1),
            **common,
        ),
        options.Overlay(
            backend="plotly",
            width=max(int(figsize[0] * 100), 1),
            height=max(int(figsize[1] * 100), 1),
            **common,
        ),
    )


@cache
def _register_holoviews_backends() -> None:
    """Register supported renderers while preserving the caller's selection."""
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    previous_backend = holoviews.Store.current_backend
    holoviews.extension("matplotlib", "bokeh", "plotly")
    holoviews.Store.set_current_backend(previous_backend)


def _manhattan_points(
    holoviews: _HoloViewsModule,
    options: _HoloViewsOptions,
    data: ManhattanPlotData,
    mask: BoolVector,
    *,
    effect_values: FloatVector | None,
    maf_values: FloatVector | None,
    color: str,
    point_size: float,
) -> _HoloViewsObject:
    columns: list[Vector] = [
        data.x_values[mask],
        data.log_p_values[mask],
        data.snp_names[mask],
        data.chromosomes[mask],
        data.positions[mask],
        data.p_values[mask],
    ]
    dimensions = ["snp", "chromosome", "position", "p_value"]
    if effect_values is not None:
        columns.append(effect_values[mask])
        dimensions.append("effect")
    if maf_values is not None:
        columns.append(maf_values[mask])
        dimensions.append("maf")

    points = holoviews.Points(
        tuple(columns),
        kdims=["genomic_position", "log_p"],
        vdims=dimensions,
    )
    return points.opts(
        options.Points(
            backend="matplotlib",
            color=color,
            s=point_size,
            alpha=0.8,
            linewidth=0,
        ),
        options.Points(
            backend="bokeh",
            color=color,
            size=point_size,
            alpha=0.7,
            tools=["hover"],
        ),
        options.Points(
            backend="plotly",
            color=color,
            size=point_size,
            alpha=0.7,
            marker="circle",
        ),
    )


def _rasterize_manhattan_layer(
    points: _HoloViewsObject,
    options: _HoloViewsOptions,
    color: str,
    figsize: tuple[float, float],
) -> _HoloViewsObject:
    datashader = cast(_DatashaderModule, _runtime_object(ds))
    rasterizer = cast(
        "_Rasterize[_HoloViewsObject, _HoloViewsObject]",
        cast(object, rasterize),
    )
    image = rasterizer(
        points,
        aggregator=datashader.max("log_p"),
        width=max(int(figsize[0] * 100), 1),
        height=max(int(figsize[1] * 100), 1),
        precompute=True,
    )
    color_map = ["#FFFFFF", color]
    return image.opts(
        options.Image(backend="matplotlib", cmap=color_map, colorbar=False),
        options.Image(backend="bokeh", cmap=color_map, colorbar=False),
        options.Image(backend="plotly", cmap=color_map, colorbar=False),
    )


def _threshold_line(
    holoviews: _HoloViewsModule,
    options: _HoloViewsOptions,
    y: float,
    *,
    color: str,
    width: float,
) -> _HoloViewsObject:
    return holoviews.HLine(y).opts(
        options.HLine(
            backend="matplotlib",
            color=color,
            linestyle="--",
            linewidth=width,
        ),
        options.HLine(
            backend="bokeh",
            color=color,
            line_dash="dashed",
            line_width=width,
        ),
        options.HLine(
            backend="plotly",
            line_color=color,
            line_dash="dash",
            line_width=width,
        ),
    )


def qq_plot(
    p_values: FloatVector,
    title: str = "QQ Plot",
    figsize: tuple[float, float] = (5, 5),
) -> Dimensioned:
    """Build a backend-neutral HoloViews QQ plot."""
    return _as_dimensioned(_qq_plot(p_values, title, figsize))


def _qq_plot(
    p_values: FloatVector,
    title: str = "QQ Plot",
    figsize: tuple[float, float] = (5, 5),
) -> _HoloViewsObject:
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

    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))

    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return holoviews.Scatter(
            (empty, empty),
            kdims=["Expected -log10(p)"],
            vdims=["Observed -log10(p)"],
        ).opts(
            options.Scatter(
                backend="matplotlib",
                fig_inches=figsize,
                title="No valid p-values",
            )
        )

    # Expected quantiles
    expected = np.asarray(
        -np.log10(np.arange(1, n + 1, dtype=np.float64) / n),
        dtype=np.float64,
    )[::-1]
    observed = -np.log10(p_obs[::-1])

    # Lambda
    lam = genomic_inflation_factor(p_values)

    max_val = max(observed.max(), expected.max()) * 1.1
    diagonal = np.asarray([0.0, max_val], dtype=np.float64)
    confidence = holoviews.Area(
        (expected, expected, expected * 1.3),
        kdims=["Expected -log10(p)"],
        vdims=["Null", "Upper confidence"],
    ).opts(
        options.Area(
            backend="matplotlib",
            color="gray",
            alpha=0.15,
            linewidth=0.0,
        )
    )
    null_line = holoviews.Curve(
        (diagonal, diagonal),
        kdims=["Expected -log10(p)"],
        vdims=["Observed -log10(p)"],
    ).opts(
        options.Curve(
            backend="matplotlib",
            color="black",
            linestyle="dashed",
            linewidth=0.8,
            alpha=0.7,
        )
    )
    points = holoviews.Scatter(
        (expected, observed),
        kdims=["Expected -log10(p)"],
        vdims=["Observed -log10(p)"],
    ).opts(
        options.Scatter(
            backend="matplotlib",
            color="#3C5587",
            s=4,
            alpha=0.7,
        )
    )
    return (confidence * null_line * points).opts(
        options.Overlay(
            backend="matplotlib",
            fig_inches=figsize,
            show_legend=False,
            title=f"{title}\n(λ = {lam:.3f})",
            xlim=(0.0, max_val),
            ylim=(0.0, max_val),
        )
    )


def kinship_heatmap(
    K: FloatMatrix,
    taxa: Vector | None = None,
    title: str = "Kinship Matrix",
    figsize: tuple[float, float] = (8, 7),
) -> Dimensioned:
    """Build a backend-neutral HoloViews kinship heatmap."""
    return _as_dimensioned(_kinship_heatmap(K, taxa, title, figsize))


def _kinship_heatmap(
    K: FloatMatrix,
    taxa: Vector | None = None,
    title: str = "Kinship Matrix",
    figsize: tuple[float, float] = (8, 7),
) -> _HoloViewsObject:
    """
    Heatmap of genomic kinship matrix.
    Translates GAPIT.Genotype.View.R (kinship heatmap section)

    Color scale: blue=low kinship, red=high kinship
    Sorted by hierarchical clustering (like GAPIT's heatmap.2)
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    if K.ndim != 2 or K.shape[0] == 0 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a non-empty square matrix")
    if not np.all(np.isfinite(K)):
        raise ValueError("K must contain only finite values")
    n = K.shape[0]
    if taxa is not None:
        require_length(taxa, n, name="taxa")
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
    ticks: list[tuple[float, str]] | None = None
    if taxa is not None and n <= 50:
        taxa_sorted = np.asarray(taxa)[order]
        ticks = [(index + 0.5, str(taxon)) for index, taxon in enumerate(taxa_sorted)]

    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))
    return holoviews.Image(
        K_sorted,
        bounds=(0.0, 0.0, float(n), float(n)),
        kdims=["Individuals X", "Individuals Y"],
        vdims=["Kinship"],
    ).opts(
        options.Image(
            backend="matplotlib",
            fig_inches=figsize,
            title=title,
            cmap="RdBu_r",
            clim=(float(K.min()), float(K.max())),
            colorbar=True,
            invert_yaxis=True,
            xticks=ticks,
            yticks=ticks,
            xrotation=90 if ticks is not None else 0,
        )
    )


def pca_plot_2d(
    scores: FloatMatrix,
    var_explained: FloatVector,
    taxa: Vector | None = None,
    groups: Vector | None = None,
    title: str = "PCA Plot",
    figsize: tuple[float, float] = (7, 6),
) -> Dimensioned:
    """Build a backend-neutral HoloViews PC1/PC2 plot."""
    return _as_dimensioned(
        _pca_plot_2d(scores, var_explained, taxa, groups, title, figsize),
    )


def _pca_plot_2d(
    scores: FloatMatrix,
    var_explained: FloatVector,
    taxa: Vector | None = None,
    groups: Vector | None = None,
    title: str = "PCA Plot",
    figsize: tuple[float, float] = (7, 6),
) -> _HoloViewsObject:
    """
    2D PCA scatter plot (PC1 vs PC2).
    Translates GAPIT.PCA.R static plot.
    """
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must be a matrix with at least two components")
    pc1, pc2 = scores[:, 0], scores[:, 1]
    sample_count = len(scores)
    taxa_values = (
        np.asarray([str(index) for index in range(sample_count)])
        if taxa is None
        else taxa
    )
    if taxa is not None:
        require_length(taxa, sample_count, name="taxa")
    if groups is not None:
        require_length(groups, sample_count, name="groups")

    columns: list[Vector] = [pc1, pc2, taxa_values]
    dimensions = ["taxa"]
    if groups is not None:
        columns.append(groups)
        dimensions.append("group")

    pct1 = var_explained[0] * 100 if len(var_explained) > 0 else 0
    pct2 = var_explained[1] * 100 if len(var_explained) > 1 else 0
    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))
    points = holoviews.Points(
        tuple(columns),
        kdims=["PC1", "PC2"],
        vdims=dimensions,
    ).opts(
        options.Points(
            backend="matplotlib",
            color="group" if groups is not None else "#3C5587",
            cmap="Category10",
            s=15 if groups is not None else 10,
            alpha=0.8 if groups is not None else 0.7,
            show_legend=groups is not None,
        )
    )
    horizontal = holoviews.HLine(0.0).opts(
        options.HLine(
            backend="matplotlib",
            color="gray",
            linewidth=0.4,
            alpha=0.5,
        )
    )
    vertical = holoviews.VLine(0.0).opts(
        options.VLine(
            backend="matplotlib",
            color="gray",
            linewidth=0.4,
            alpha=0.5,
        )
    )
    return (points * horizontal * vertical).opts(
        options.Overlay(
            backend="matplotlib",
            fig_inches=figsize,
            title=title,
            xlabel=f"PC1 ({pct1:.1f}%)",
            ylabel=f"PC2 ({pct2:.1f}%)",
            legend_position="right",
        )
    )


def pca_plot_3d(
    scores: FloatMatrix,
    var_explained: FloatVector,
    taxa: Vector | None = None,
    groups: Vector | None = None,
    title: str = "3D PCA",
) -> Dimensioned:
    """Build a backend-neutral HoloViews 3D PCA plot."""
    if scores.ndim != 2 or scores.shape[1] < 3:
        raise ValueError("scores must be a matrix with at least three components")
    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))
    pc1, pc2, pc3 = scores[:, 0], scores[:, 1], scores[:, 2]
    sample_count = len(scores)
    taxa_values = (
        np.asarray([str(index) for index in range(sample_count)])
        if taxa is None
        else taxa
    )
    if taxa is not None:
        require_length(taxa, sample_count, name="taxa")
    if groups is not None:
        require_length(groups, sample_count, name="groups")
    columns: list[Vector] = [pc1, pc2, pc3, taxa_values]
    dimensions = ["taxa"]
    if groups is not None:
        columns.append(groups)
        dimensions.append("group")

    pct = np.zeros(3, dtype=np.float64)
    available_components = min(len(var_explained), 3)
    pct[:available_components] = var_explained[:available_components] * 100.0
    plot = holoviews.Scatter3D(
        tuple(columns),
        kdims=["PC1", "PC2", "PC3"],
        vdims=dimensions,
    ).opts(
        options.Scatter3D(
            backend="plotly",
            color="group" if groups is not None else "PC1",
            cmap="Category10" if groups is not None else "Viridis",
            marker="circle",
            size=4,
            alpha=0.85,
            width=700,
            height=600,
            title=title,
            xlabel=f"PC1 ({pct[0]:.1f}%)",
            ylabel=f"PC2 ({pct[1]:.1f}%)",
            zlabel=f"PC3 ({pct[2]:.1f}%)",
        )
    )
    return _as_dimensioned(plot)


def _interactive_manhattan_metadata(
    data: ManhattanPlotData,
    effects: NumericVector | None,
    maf: NumericVector | None,
) -> tuple[FloatVector | None, FloatVector | None]:
    marker_count = len(data.p_values)
    effect_values = (
        None if effects is None else as_float_vector(effects, name="effects")
    )
    maf_values = None if maf is None else as_float_vector(maf, name="MAF")
    if effect_values is not None:
        require_length(effect_values, marker_count, name="effects")
    if maf_values is not None:
        require_length(maf_values, marker_count, name="MAF")
    return effect_values, maf_values


def gs_scatter(
    observed: NumericVector,
    predicted: NumericVector,
    taxa: Vector | None = None,
    trait_name: str = "Trait",
    figsize: tuple[float, float] = (6, 5),
) -> Dimensioned:
    """Build backend-neutral genomic-selection diagnostics."""
    return _as_dimensioned(_gs_scatter(observed, predicted, taxa, trait_name, figsize))


def _gs_scatter(
    observed: NumericVector,
    predicted: NumericVector,
    taxa: Vector | None = None,
    trait_name: str = "Trait",
    figsize: tuple[float, float] = (6, 5),
) -> _HoloViewsObject:
    """
    Genomic Selection scatter: predicted vs observed.
    Translates GAPIT.GS.Visualization.R
    Pearson r = prediction accuracy.
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    require_length(predicted, len(observed), name="predicted")
    if taxa is not None:
        require_length(taxa, len(observed), name="taxa")
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs_v = observed[valid]
    pred_v = predicted[valid]

    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))

    if len(obs_v) < 2:
        empty = np.empty(0, dtype=np.float64)
        return holoviews.Scatter(
            (empty, empty),
            kdims=[f"Observed {trait_name}"],
            vdims=[f"Predicted {trait_name}"],
        ).opts(
            options.Scatter(
                backend="matplotlib",
                fig_inches=figsize,
                title="Insufficient data for GS scatter",
            )
        )

    r = np.corrcoef(obs_v, pred_v)[0, 1]

    # Regression line
    m_coef = np.asarray(np.polyfit(obs_v, pred_v, 1), dtype=np.float64)
    x_min, x_max = obs_v.min(), obs_v.max()
    x_line = np.linspace(x_min, x_max, 100, dtype=np.float64)
    y_line = np.asarray(m_coef[0] * x_line + m_coef[1], dtype=np.float64)
    columns: list[Vector] = [obs_v, pred_v]
    dimensions: list[str] = []
    if taxa is not None:
        columns.append(np.asarray(taxa)[valid])
        dimensions.append("taxa")
    points = holoviews.Scatter(
        tuple(columns),
        kdims=[f"Observed {trait_name}"],
        vdims=[f"Predicted {trait_name}", *dimensions],
    ).opts(
        options.Scatter(
            backend="matplotlib",
            color="#3C5587",
            s=15,
            alpha=0.6,
        )
    )
    regression = holoviews.Curve(
        (x_line, y_line),
        kdims=[f"Observed {trait_name}"],
        vdims=[f"Predicted {trait_name}"],
    ).opts(
        options.Curve(
            backend="matplotlib",
            color="red",
            linewidth=1.2,
            alpha=0.8,
        )
    )
    return (points * regression).opts(
        options.Overlay(
            backend="matplotlib",
            fig_inches=figsize,
            show_legend=False,
            title=f"GS Accuracy (r = {r:.3f})",
        )
    )


def phenotype_distribution(
    y: FloatVector,
    trait_name: str = "Trait",
    significant_snp_geno: Vector | None = None,
    figsize: tuple[float, float] = (6, 4),
) -> Dimensioned:
    """Build a backend-neutral phenotype distribution."""
    return _as_dimensioned(
        _phenotype_distribution(y, trait_name, significant_snp_geno, figsize),
    )


def _phenotype_distribution(
    y: FloatVector,
    trait_name: str = "Trait",
    significant_snp_geno: Vector | None = None,
    figsize: tuple[float, float] = (6, 4),
) -> _HoloViewsObject:
    """
    Phenotype distribution histogram.
    Translates GAPIT.Phenotype.View.R
    Optionally split by genotype at top significant SNP.
    """
    finite = np.isfinite(y)
    valid_y = y[finite]
    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))
    frequencies, edges = np.histogram(valid_y, bins=30)
    layers = [
        holoviews.Histogram(
            (
                np.asarray(edges, dtype=np.float64),
                np.asarray(frequencies, dtype=np.float64),
            ),
            kdims=[trait_name],
            vdims=["Count"],
        ).opts(
            options.Histogram(
                backend="matplotlib",
                color="#3C5587",
                alpha=0.75,
                edgecolor="white",
                linewidth=0.3,
            )
        )
    ]

    if significant_snp_geno is not None:
        require_length(significant_snp_geno, len(y), name="significant_snp_geno")
        for geno_val, label, color in [
            (0, "Ref/Ref", "#2166AC"),
            (1, "Ref/Alt", "#74ADD1"),
            (2, "Alt/Alt", "#D73027"),
        ]:
            mask = (significant_snp_geno == geno_val) & finite
            if mask.any():
                group_frequencies, group_edges = np.histogram(y[mask], bins=20)
                layers.append(
                    holoviews.Histogram(
                        (
                            np.asarray(group_edges, dtype=np.float64),
                            np.asarray(group_frequencies, dtype=np.float64),
                        ),
                        kdims=[trait_name],
                        vdims=["Count"],
                        label=label,
                    ).opts(
                        options.Histogram(
                            backend="matplotlib",
                            alpha=0.5,
                            color=color,
                            linewidth=0.0,
                        )
                    )
                )
    return holoviews.Overlay(layers).opts(
        options.Overlay(
            backend="matplotlib",
            fig_inches=figsize,
            title=f"Distribution of {trait_name}",
            show_legend=significant_snp_geno is not None,
            legend_position="right",
        )
    )


def multiple_manhattan(
    chromosomes: LabelVector,
    positions: NumericVector,
    model_p_values: list[tuple[str, FloatVector]],
    *,
    title: str = "Multiple Manhattan",
    figsize: tuple[float, float] = (12, 5),
) -> Dimensioned:
    """Build a multi-model Manhattan comparison as a HoloViews overlay."""
    x_values, labels, centers = prepare_genomic_axis(chromosomes, positions)
    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))
    layers: list[_HoloViewsObject] = []
    for model, p_values in model_p_values:
        require_length(p_values, len(x_values), name=f"{model} p_values")
        valid = np.isfinite(p_values) & (p_values > 0.0) & (p_values <= 1.0)
        layers.append(
            holoviews.Scatter(
                (x_values[valid], -np.log10(p_values[valid])),
                kdims=["genomic_position"],
                vdims=["log_p"],
                label=model,
            ).opts(
                options.Scatter(
                    backend="matplotlib",
                    s=12,
                    alpha=0.65,
                )
            )
        )
    ticks = list(zip(centers.tolist(), labels, strict=True))
    return _as_dimensioned(
        holoviews.Overlay(layers).opts(
            options.Overlay(
                backend="matplotlib",
                fig_inches=figsize,
                title=title,
                xlabel="Chromosome",
                ylabel="-log10(p)",
                xticks=ticks,
                show_legend=True,
            )
        ),
    )


def multiple_qq(
    model_p_values: list[tuple[str, FloatVector]],
    *,
    title: str = "Multiple QQ",
    figsize: tuple[float, float] = (6, 6),
) -> Dimensioned:
    """Build a multi-model QQ comparison as a HoloViews overlay."""
    _register_holoviews_backends()
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    options = cast(_HoloViewsOptions, _runtime_object(hv_opts))
    layers: list[_HoloViewsObject] = []
    upper = 1.0
    for model, p_values in model_p_values:
        valid = p_values[np.isfinite(p_values) & (p_values > 0.0) & (p_values <= 1.0)]
        observed = -np.log10(np.sort(valid))
        expected = (
            -np.log10(
                (np.arange(1, len(observed) + 1, dtype=np.float64) - 0.5)
                / len(observed)
            )
            if len(observed)
            else np.empty(0, dtype=np.float64)
        )
        if len(observed):
            upper = max(upper, observed.max(), expected.max())
        layers.append(
            holoviews.Curve(
                (expected, observed),
                kdims=["Expected -log10(p)"],
                vdims=["Observed -log10(p)"],
                label=model,
            ).opts(options.Curve(backend="matplotlib", marker="o", linewidth=1))
        )
    diagonal = np.asarray([0.0, upper], dtype=np.float64)
    layers.append(
        holoviews.Curve(
            (diagonal, diagonal),
            kdims=["Expected -log10(p)"],
            vdims=["Observed -log10(p)"],
        ).opts(
            options.Curve(
                backend="matplotlib",
                color="grey",
                linestyle="dashed",
            )
        )
    )
    return _as_dimensioned(
        holoviews.Overlay(layers).opts(
            options.Overlay(
                backend="matplotlib",
                fig_inches=figsize,
                title=title,
                show_legend=True,
            )
        ),
    )
