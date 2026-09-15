"""
Visualization module.
Translates GAPIT.Manhattan.R, GAPIT.QQ.R, GAPIT.PCA.R,
GAPIT.GS.Visualization.R, GAPIT.Phenotype.View.R

All plots are publication-ready and match GAPIT's visual style.
Static plots use Matplotlib. Interactive plots use Plotly or Bokeh, with
HoloViews and Datashader providing bounded rendering for dense point clouds.
"""

from __future__ import annotations

import typing as t
from functools import cache
from typing import Literal, Protocol, TypedDict, TypeVar, cast, overload

import datashader as ds
import holoviews as hv
import numpy as np
from holoviews import opts as hv_opts
from holoviews.operation.datashader import rasterize

if t.TYPE_CHECKING:
    from typing import Self

    from bokeh.plotting import figure as BokehFigure
    from datashader.reductions import Reduction
    from holoviews.core.options import Options
    from matplotlib.figure import Figure
    from plotly.graph_objs._figure import Figure as PlotlyFigure

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
from ._backends import StaticStyle as _StaticStyle
from ._backends import matplotlib_style, plotly_figure
from .data import ManhattanPlotData, prepare_manhattan_data

StaticStyle: t.TypeAlias = _StaticStyle
PlotMode = Literal["static", "interactive"]
PlotBackend = Literal["auto", "matplotlib", "plotly", "bokeh"]
LargeDataMode = Literal["auto", "points", "aggregate"]
StaticBackend = Literal["auto", "matplotlib"]
PlotlyBackend = Literal["auto", "plotly"]
BokehBackend = Literal["bokeh"]

_AGGREGATE_MARKER_THRESHOLD = 250_000

_RasterizeInputT_contra = TypeVar("_RasterizeInputT_contra", contravariant=True)
_RasterizeOutputT_co = TypeVar("_RasterizeOutputT_co", covariant=True)


class _HoloViewsObject(Protocol):
    def __mul__(self, other: _HoloViewsObject) -> _HoloViewsObject: ...
    def collate(self) -> _HoloViewsObject: ...
    def opts(self, *options: Options) -> Self: ...


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
    ) -> _HoloViewsObject: ...

    def Curve(
        self,
        data: tuple[Vector, ...],
        *,
        kdims: list[str],
        vdims: list[str],
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

    @overload
    def render(
        self,
        obj: _HoloViewsObject,
        *,
        backend: Literal["matplotlib"],
    ) -> Figure: ...

    @overload
    def render(
        self,
        obj: _HoloViewsObject,
        *,
        backend: Literal["plotly"],
    ) -> dict[str, object]: ...

    @overload
    def render(
        self,
        obj: _HoloViewsObject,
        *,
        backend: Literal["bokeh"],
    ) -> BokehFigure: ...

    def save(
        self,
        obj: _HoloViewsObject,
        filename: str,
        *,
        backend: Literal["matplotlib", "plotly", "bokeh"],
    ) -> object: ...


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


class _PlotlyTraceState(TypedDict, total=False):
    type: str
    text: StrVector
    hovertemplate: str


class _PlotlyRenderState(TypedDict, total=False):
    data: list[_PlotlyTraceState]


class _PlotlyHookPlot(Protocol):
    state: _PlotlyRenderState


class _HoloViewsHook(Protocol):
    def __call__(self, plot: object, _element: object) -> None: ...


def _runtime_object(value: object) -> object:
    """Cross a third-party stub boundary without weakening the result type."""
    return value


def _resolve_manhattan_backend(
    mode: PlotMode,
    backend: PlotBackend,
) -> Literal["matplotlib", "plotly", "bokeh"]:
    match mode, backend:
        case "static", "auto" | "matplotlib":
            return "matplotlib"
        case "interactive", "auto" | "plotly":
            return "plotly"
        case "interactive", "bokeh":
            return "bokeh"
        case "static", _:
            raise ValueError("static Manhattan plots require the matplotlib backend")
        case "interactive", _:
            raise ValueError("interactive Manhattan plots require plotly or bokeh")
    raise ValueError("unsupported Manhattan plot mode or backend")


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


@overload
def manhattan(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    *,
    mode: Literal["static"] = "static",
    backend: StaticBackend = "auto",
    style: StaticStyle = "pygapit",
    title: str = "Manhattan Plot",
    significance_threshold: float | None = None,
    suggestive_threshold: float | None = None,
    highlight_snps: IntVector | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 5),
    point_size: float = 1.5,
    large_data: LargeDataMode = "auto",
) -> Figure: ...


@overload
def manhattan(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    *,
    mode: Literal["interactive"],
    backend: PlotlyBackend = "auto",
    style: Literal["pygapit"] = "pygapit",
    title: str = "Interactive Manhattan",
    significance_threshold: float | None = None,
    suggestive_threshold: float | None = None,
    effects: NumericVector | None = None,
    maf: NumericVector | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (9, 4),
    point_size: float = 3.0,
    large_data: LargeDataMode = "auto",
) -> PlotlyFigure: ...


@overload
def manhattan(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    *,
    mode: Literal["interactive"],
    backend: BokehBackend,
    style: Literal["pygapit"] = "pygapit",
    title: str = "Interactive Manhattan",
    significance_threshold: float | None = None,
    suggestive_threshold: float | None = None,
    effects: NumericVector | None = None,
    maf: NumericVector | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (9, 4),
    point_size: float = 3.0,
    large_data: LargeDataMode = "auto",
) -> BokehFigure: ...


def manhattan(
    snp_names: StrVector,
    chromosomes: LabelVector,
    positions: NumericVector,
    p_values: NumericVector,
    *,
    mode: PlotMode = "static",
    backend: PlotBackend = "auto",
    style: StaticStyle = "pygapit",
    title: str | None = None,
    significance_threshold: float | None = None,
    suggestive_threshold: float | None = None,
    highlight_snps: IntVector | None = None,
    effects: NumericVector | None = None,
    maf: NumericVector | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
    point_size: float | None = None,
    large_data: LargeDataMode = "auto",
) -> Figure | PlotlyFigure | BokehFigure:
    """Render a Manhattan plot through a statically valid backend/mode pair."""
    resolved_backend = _resolve_manhattan_backend(mode, backend)
    if mode == "interactive" and style != "pygapit":
        raise ValueError("seaborn and science styles require static matplotlib output")
    if mode == "static" and (effects is not None or maf is not None):
        raise ValueError("effects and maf are available only for interactive output")
    if mode == "interactive" and highlight_snps is not None:
        raise ValueError("highlight_snps is available only for static output")

    data = prepare_manhattan_data(
        snp_names,
        chromosomes,
        positions,
        p_values,
        significance_threshold=significance_threshold,
        suggestive_threshold=suggestive_threshold,
    )
    resolved_figsize = (
        figsize if figsize is not None else ((14, 5) if mode == "static" else (9, 4))
    )
    resolved_point_size = (
        point_size if point_size is not None else (1.5 if mode == "static" else 3.0)
    )
    _validate_plot_geometry(resolved_figsize, resolved_point_size)
    plot = _build_manhattan_plot(
        data,
        backend=resolved_backend,
        aggregate=_use_aggregate(len(data.p_values), large_data),
        title=title
        or ("Manhattan Plot" if mode == "static" else "Interactive Manhattan"),
        highlight_snps=highlight_snps,
        effects=effects,
        maf=maf,
        figsize=resolved_figsize,
        point_size=resolved_point_size,
    )
    if resolved_backend == "matplotlib":
        with matplotlib_style(style):
            return _render_holoviews(
                plot,
                backend="matplotlib",
                save_path=save_path,
            )
    if resolved_backend == "plotly":
        return _render_holoviews(
            plot,
            backend="plotly",
            save_path=save_path,
        )
    return _render_holoviews(
        plot,
        backend="bokeh",
        save_path=save_path,
    )


def _build_manhattan_plot(
    data: ManhattanPlotData,
    *,
    backend: Literal["matplotlib", "plotly", "bokeh"],
    aggregate: bool,
    title: str,
    highlight_snps: IntVector | None,
    effects: NumericVector | None,
    maf: NumericVector | None,
    figsize: tuple[float, float],
    point_size: float,
) -> _HoloViewsObject:
    """Build a HoloViews graph configured only for the selected renderer."""
    _register_holoviews_backend(backend)
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
            backend=backend,
            hover=not aggregate,
        )
        layers.append(
            _rasterize_manhattan_layer(
                points,
                options,
                color,
                figsize,
                backend=backend,
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
            backend=backend,
            hover=True,
        )

    plot *= _threshold_line(
        holoviews,
        options,
        -np.log10(data.significance_threshold),
        backend=backend,
        color=SIG_COLOR,
        width=0.8,
    )
    plot *= _threshold_line(
        holoviews,
        options,
        -np.log10(data.suggestive_threshold),
        backend=backend,
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
    match backend:
        case "matplotlib":
            overlay_options = options.Overlay(
                backend=backend,
                fig_inches=figsize,
                title=title,
                xlabel="Chromosome",
                ylabel="-log10(p)",
                xlim=(0.0, x_limit),
                ylim=(0.0, y_limit),
                xticks=ticks,
                show_frame=False,
            )
        case "bokeh" | "plotly":
            overlay_options = options.Overlay(
                backend=backend,
                width=max(int(figsize[0] * 100), 1),
                height=max(int(figsize[1] * 100), 1),
                title=title,
                xlabel="Chromosome",
                ylabel="-log10(p)",
                xlim=(0.0, x_limit),
                ylim=(0.0, y_limit),
                xticks=ticks,
            )
    return plot.opts(overlay_options)


@cache
def _register_holoviews_backend(
    backend: Literal["matplotlib", "plotly", "bokeh"],
) -> None:
    """Register one renderer while preserving the caller's selected backend."""
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    previous_backend = holoviews.Store.current_backend
    holoviews.extension(backend)
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
    backend: Literal["matplotlib", "plotly", "bokeh"],
    hover: bool,
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
    match backend:
        case "matplotlib":
            point_options = options.Points(
                backend=backend,
                color=color,
                s=point_size,
                alpha=0.8,
                linewidth=0,
            )
        case "bokeh":
            if hover:
                tooltips = [
                    ("SNP", "@snp"),
                    ("Chromosome", "@chromosome"),
                    ("Position", "@position{0,0}"),
                    ("P-value", "@p_value{0.00e}"),
                ]
                if effect_values is not None:
                    tooltips.append(("Effect", "@effect{0.0000}"))
                if maf_values is not None:
                    tooltips.append(("MAF", "@maf{0.000}"))
                point_options = options.Points(
                    backend=backend,
                    color=color,
                    size=point_size,
                    alpha=0.7,
                    tools=["hover"],
                    hover_tooltips=tooltips,
                )
            else:
                point_options = options.Points(
                    backend=backend,
                    color=color,
                    size=point_size,
                    alpha=0.7,
                )
        case "plotly":
            hooks: list[_HoloViewsHook] = (
                [_plotly_manhattan_hook(data, mask, effect_values, maf_values)]
                if hover
                else []
            )
            point_options = options.Points(
                backend=backend,
                color=color,
                size=point_size,
                alpha=0.7,
                marker="circle",
                hooks=hooks,
            )
    return points.opts(point_options)


def _plotly_manhattan_hook(
    data: ManhattanPlotData,
    mask: BoolVector,
    effect_values: FloatVector | None,
    maf_values: FloatVector | None,
) -> _HoloViewsHook:
    marker_names = data.snp_names[mask]
    chromosomes = data.chromosomes[mask]
    positions = data.positions[mask]
    p_values = data.p_values[mask]
    effects = None if effect_values is None else effect_values[mask]
    frequencies = None if maf_values is None else maf_values[mask]
    labels: list[str] = []
    for index, marker_name in enumerate(marker_names):
        label = (
            f"<b>{marker_name}</b><br>Chromosome: {chromosomes[index]}<br>"
            f"Position: {positions[index]:,.0f}<br>P-value: {p_values[index]:.2e}"
        )
        if effects is not None:
            label += f"<br>Effect: {effects[index]:.4f}"
        if frequencies is not None:
            label += f"<br>MAF: {frequencies[index]:.3f}"
        labels.append(label)
    hover_text = np.asarray(labels, dtype=np.str_)

    def apply_hover(plot: object, _element: object) -> None:
        state = cast(_PlotlyHookPlot, _runtime_object(plot)).state
        traces = state.get("data", [])
        if not traces:
            return
        trace = traces[0]
        trace["type"] = "scattergl"
        trace["text"] = hover_text
        trace["hovertemplate"] = "%{text}<extra></extra>"

    return apply_hover


def _rasterize_manhattan_layer(
    points: _HoloViewsObject,
    options: _HoloViewsOptions,
    color: str,
    figsize: tuple[float, float],
    *,
    backend: Literal["matplotlib", "plotly", "bokeh"],
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
    return image.opts(options.Image(backend=backend, cmap=color_map, colorbar=False))


def _threshold_line(
    holoviews: _HoloViewsModule,
    options: _HoloViewsOptions,
    y: float,
    *,
    backend: Literal["matplotlib", "plotly", "bokeh"],
    color: str,
    width: float,
) -> _HoloViewsObject:
    match backend:
        case "matplotlib":
            line_options = options.HLine(
                backend=backend,
                color=color,
                linestyle="--",
                linewidth=width,
            )
        case "bokeh":
            line_options = options.HLine(
                backend=backend,
                color=color,
                line_dash="dashed",
                line_width=width,
            )
        case "plotly":
            line_options = options.HLine(
                backend=backend,
                line_color=color,
                line_dash="dash",
                line_width=width,
            )
    return holoviews.HLine(y).opts(line_options)


@overload
def _render_holoviews(
    obj: _HoloViewsObject,
    *,
    backend: Literal["matplotlib"],
    save_path: str | None,
) -> Figure: ...


@overload
def _render_holoviews(
    obj: _HoloViewsObject,
    *,
    backend: Literal["plotly"],
    save_path: str | None,
) -> PlotlyFigure: ...


@overload
def _render_holoviews(
    obj: _HoloViewsObject,
    *,
    backend: Literal["bokeh"],
    save_path: str | None,
) -> BokehFigure: ...


def _render_holoviews(
    obj: _HoloViewsObject,
    *,
    backend: Literal["matplotlib", "plotly", "bokeh"],
    save_path: str | None,
) -> Figure | PlotlyFigure | BokehFigure:
    """Render and optionally export through HoloViews' public entry points."""
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    if save_path is not None:
        holoviews.save(obj, save_path, backend=backend)
    match backend:
        case "matplotlib":
            return holoviews.render(obj, backend="matplotlib")
        case "bokeh":
            return holoviews.render(obj, backend="bokeh")
        case "plotly":
            # Plotly's public Figure model ignores HoloViews' private axis
            # metadata while retaining its fully rendered trace state.
            return plotly_figure(holoviews.render(obj, backend="plotly"))


def qq_plot(
    p_values: FloatVector,
    title: str = "QQ Plot",
    save_path: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    *,
    style: StaticStyle = "pygapit",
) -> Figure:
    """Render a QQ plot through HoloViews' Matplotlib backend."""
    with matplotlib_style(style):
        return _render_holoviews(
            _qq_plot(p_values, title, figsize),
            backend="matplotlib",
            save_path=save_path,
        )


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

    _register_holoviews_backend("matplotlib")
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
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 7),
    *,
    style: StaticStyle = "pygapit",
) -> Figure:
    """Render a kinship heatmap through HoloViews' Matplotlib backend."""
    with matplotlib_style(style):
        return _render_holoviews(
            _kinship_heatmap(K, taxa, title, figsize),
            backend="matplotlib",
            save_path=save_path,
        )


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

    _register_holoviews_backend("matplotlib")
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
    save_path: str | None = None,
    figsize: tuple[float, float] = (7, 6),
    *,
    style: StaticStyle = "pygapit",
) -> Figure:
    """Render static PC1/PC2 scores through HoloViews."""
    with matplotlib_style(style):
        return _render_holoviews(
            _pca_plot_2d(scores, var_explained, taxa, groups, title, figsize),
            backend="matplotlib",
            save_path=save_path,
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
    _register_holoviews_backend("matplotlib")
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


def pca_plot_3d_interactive(
    scores: FloatMatrix,
    var_explained: FloatVector,
    taxa: Vector | None = None,
    groups: Vector | None = None,
    title: str = "3D PCA",
    save_path: str | None = None,
) -> PlotlyFigure:
    """Render interactive PCA scores through HoloViews' Plotly backend."""
    if scores.ndim != 2 or scores.shape[1] < 3:
        raise ValueError("scores must be a matrix with at least three components")
    _register_holoviews_backend("plotly")
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
            hooks=[_pca_plotly_hover_hook(taxa_values, groups)],
        )
    )
    return _render_holoviews(
        plot,
        backend="plotly",
        save_path=save_path,
    )


def _pca_plotly_hover_hook(
    taxa: Vector,
    groups: Vector | None,
) -> _HoloViewsHook:
    labels = np.asarray(taxa, dtype=np.str_)
    if groups is not None:
        labels = np.asarray(
            [
                f"{taxon}<br>Group: {group}"
                for taxon, group in zip(labels, groups, strict=True)
            ],
            dtype=np.str_,
        )

    def apply_hover(plot: object, _element: object) -> None:
        state = cast(_PlotlyHookPlot, _runtime_object(plot)).state
        traces = state.get("data", [])
        if not traces:
            return
        trace = traces[0]
        trace["text"] = labels
        trace["hovertemplate"] = (
            "<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>"
            "PC3: %{z:.3f}<extra></extra>"
        )

    return apply_hover


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
    save_path: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    *,
    style: StaticStyle = "pygapit",
) -> Figure:
    """Render genomic-selection diagnostics through HoloViews."""
    with matplotlib_style(style):
        return _render_holoviews(
            _gs_scatter(observed, predicted, taxa, trait_name, figsize),
            backend="matplotlib",
            save_path=save_path,
        )


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

    _register_holoviews_backend("matplotlib")
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
    save_path: str | None = None,
    figsize: tuple[float, float] = (6, 4),
    *,
    style: StaticStyle = "pygapit",
) -> Figure:
    """Render a phenotype distribution through HoloViews."""
    with matplotlib_style(style):
        return _render_holoviews(
            _phenotype_distribution(y, trait_name, significant_snp_geno, figsize),
            backend="matplotlib",
            save_path=save_path,
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
    _register_holoviews_backend("matplotlib")
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
