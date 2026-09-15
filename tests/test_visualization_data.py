"""Backend-independent visualization data contracts."""

from __future__ import annotations

import importlib
from importlib.util import find_spec
from types import ModuleType
from typing import Protocol, cast

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pytest
from bokeh.models import Image as BokehImage
from bokeh.plotting import figure as BokehFigure
from matplotlib.figure import Figure
from plotly.graph_objs._figure import Figure as PlotlyFigure

from pygapit.visualization import prepare_genomic_axis, prepare_manhattan_data
from pygapit.visualization.plots import (
    StaticStyle,
    gs_scatter,
    kinship_heatmap,
    manhattan,
    pca_plot_2d,
    pca_plot_3d_interactive,
    phenotype_distribution,
    qq_plot,
)


class _PlotlyTrace(Protocol):
    type: str
    text: np.ndarray
    hovertemplate: str


class _PlotlyFigureData(Protocol):
    data: tuple[object, ...]


class _PlotlyTypedTrace(Protocol):
    type: str


class _BokehGlyphRenderer(Protocol):
    glyph: object


class _HoloViewsStore(Protocol):
    current_backend: str

    def set_current_backend(self, backend: str) -> None: ...


class _HoloViewsExtension(Protocol):
    def __call__(self, *backends: str) -> None: ...


def test_prepare_manhattan_data_reuses_compatible_inputs() -> None:
    snps = np.asarray(["s1", "s2", "s3", "s4"])
    chromosomes = np.asarray(["1", "1", "2", "2"])
    positions = np.asarray([10.0, 20.0, 5.0, 15.0])
    p_values = np.asarray([0.05, np.nan, 0.0, 1e-4])

    prepared = prepare_manhattan_data(
        snps,
        chromosomes,
        positions,
        p_values,
    )

    assert np.shares_memory(prepared.snp_names, snps)
    assert np.shares_memory(prepared.chromosomes, chromosomes)
    assert np.shares_memory(prepared.positions, positions)
    assert np.shares_memory(prepared.p_values, p_values)
    np.testing.assert_allclose(prepared.x_values, [0.0, 10.0, 5_000_010.0, 5_000_020.0])
    np.testing.assert_allclose(
        prepared.log_p_values,
        [-np.log10(0.05), 0.0, 0.0, 4.0],
    )
    assert prepared.chromosome_labels == ("1", "2")
    np.testing.assert_allclose(prepared.chromosome_centers, [5.0, 5_000_015.0])
    assert prepared.significance_threshold == pytest.approx(0.0125)
    assert prepared.suggestive_threshold == pytest.approx(0.25)


def test_prepare_genomic_axis_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="positions must have length 2"):
        prepare_genomic_axis(
            np.asarray(["1", "2"]),
            np.asarray([1.0]),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        prepare_genomic_axis(
            np.asarray(["1"]),
            np.asarray([1.0]),
            chromosome_gap=np.inf,
        )


def test_prepare_genomic_axis_preserves_interleaved_chromosome_semantics() -> None:
    x_values, labels, centers = prepare_genomic_axis(
        np.asarray(["1", "2", "1"]),
        np.asarray([10.0, 5.0, 20.0]),
    )

    assert labels == ("1", "2")
    np.testing.assert_allclose(x_values, [0.0, 5_000_010.0, 10.0])
    np.testing.assert_allclose(centers, [5.0, 5_000_010.0])


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.inf, np.nan, 1.1])
def test_prepare_manhattan_data_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="significance_threshold"):
        prepare_manhattan_data(
            np.asarray(["s1"]),
            np.asarray(["1"]),
            np.asarray([1.0]),
            np.asarray([0.5]),
            significance_threshold=threshold,
        )


def test_manhattan_preserves_selected_holoviews_backend() -> None:
    store = cast(_HoloViewsStore, cast(object, hv.Store))
    previous_backend = store.current_backend
    extension = cast(_HoloViewsExtension, cast(object, hv.extension))
    extension("plotly")
    store.set_current_backend("plotly")
    try:
        figure = manhattan(
            np.asarray(["s1"]),
            np.asarray(["1"]),
            np.asarray([10.0]),
            np.asarray([0.05]),
            backend="matplotlib",
            large_data="points",
        )
        assert store.current_backend == "plotly"
        plt.close(figure)
    finally:
        store.set_current_backend(previous_backend)


def test_interactive_manhattan_uses_webgl_and_hover_metadata() -> None:
    figure = manhattan(
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "1", "2"]),
        np.asarray([10.0, 20.0, 5.0]),
        np.asarray([0.05, 1e-4, 0.2]),
        mode="interactive",
        effects=np.asarray([0.1, -0.2, 0.3]),
        maf=np.asarray([0.2, 0.3, 0.4]),
    )

    assert figure is not None
    first_trace = cast(_PlotlyTrace, cast(object, figure.data[0]))
    assert first_trace.type == "scattergl"
    hover_text = np.asarray(first_trace.text, dtype=np.str_)
    assert "<b>s1</b>" in hover_text[0]
    assert "Position: 10" in hover_text[0]
    assert "Effect: 0.1000" in hover_text[0]
    assert "MAF: 0.200" in hover_text[0]
    assert first_trace.hovertemplate == "%{text}<extra></extra>"


def test_manhattan_backend_returns_are_precise() -> None:
    arguments = (
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "1", "2"]),
        np.asarray([10.0, 20.0, 5.0]),
        np.asarray([0.05, 1e-4, 0.2]),
    )

    static = manhattan(*arguments, backend="matplotlib", large_data="points")
    plotly = manhattan(
        *arguments,
        mode="interactive",
        backend="plotly",
        large_data="points",
    )
    bokeh = manhattan(
        *arguments,
        mode="interactive",
        backend="bokeh",
        large_data="points",
    )

    assert isinstance(static, Figure)
    assert isinstance(plotly, PlotlyFigure)
    assert isinstance(bokeh, BokehFigure)
    plt.close(static)


def test_manhattan_aggregate_renders_significant_points() -> None:
    arguments = (
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "1", "2"]),
        np.asarray([10.0, 20.0, 5.0]),
        np.asarray([0.05, 1e-4, 0.2]),
    )
    static = manhattan(*arguments, large_data="aggregate")
    plotly = manhattan(
        *arguments,
        mode="interactive",
        backend="plotly",
        large_data="aggregate",
    )
    bokeh = manhattan(
        *arguments,
        mode="interactive",
        backend="bokeh",
        large_data="aggregate",
    )

    assert isinstance(static, Figure)
    assert isinstance(plotly, PlotlyFigure)
    assert isinstance(bokeh, BokehFigure)
    plotly_data = cast(_PlotlyFigureData, cast(object, plotly))
    assert len(plotly_data.data) >= 2
    assert len(static.axes[0].images) == 2
    assert any(
        cast(_PlotlyTypedTrace, trace).type == "heatmap" for trace in plotly_data.data
    )
    assert (
        sum(
            isinstance(
                cast(_BokehGlyphRenderer, cast(object, renderer)).glyph, BokehImage
            )
            for renderer in bokeh.renderers
            if hasattr(renderer, "glyph")
        )
        == 2
    )
    plt.close(static)


def test_manhattan_auto_uses_datashader_at_marker_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pygapit.visualization.plots as visualization_plots

    monkeypatch.setattr(visualization_plots, "_AGGREGATE_MARKER_THRESHOLD", 3)
    figure = manhattan(
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "1", "2"]),
        np.asarray([10.0, 20.0, 5.0]),
        np.asarray([0.05, 1e-4, 0.2]),
        large_data="auto",
    )

    assert len(figure.axes[0].images) == 2
    plt.close(figure)


def test_pca_3d_uses_holoviews_scatter_with_group_hover() -> None:
    figure = pca_plot_3d_interactive(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.asarray([0.5, 0.3, 0.2]),
        taxa=np.asarray(["sample-a", "sample-b"]),
        groups=np.asarray(["group-1", "group-2"]),
    )

    trace = cast(_PlotlyTrace, cast(object, figure.data[0]))
    assert trace.type == "scatter3d"
    np.testing.assert_array_equal(
        trace.text,
        ["sample-a<br>Group: group-1", "sample-b<br>Group: group-2"],
    )


def test_pca_3d_validates_hover_metadata_lengths() -> None:
    with pytest.raises(ValueError, match="taxa must have length 2"):
        pca_plot_3d_interactive(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.asarray([0.5, 0.3, 0.2]),
            taxa=np.asarray(["sample-a"]),
        )


def test_static_holoviews_diagnostics_cover_specialized_layers() -> None:
    qq = qq_plot(np.asarray([0.5, 0.1, 0.01]))
    kinship = kinship_heatmap(
        np.asarray([[1.0, 0.2], [0.2, 1.0]]),
        taxa=np.asarray(["sample-a", "sample-b"]),
    )
    pca = pca_plot_2d(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([0.6, 0.3]),
        taxa=np.asarray(["sample-a", "sample-b"]),
        groups=np.asarray(["group-1", "group-2"]),
    )
    gs = gs_scatter(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.1, 1.9, 3.2]),
    )
    phenotype = phenotype_distribution(
        np.asarray([1.0, 1.5, 2.0, 2.5]),
        significant_snp_geno=np.asarray([0, 1, 2, 2]),
    )

    assert all(
        isinstance(figure, Figure) for figure in (qq, kinship, pca, gs, phenotype)
    )
    assert len(qq.axes[0].collections) == 2
    assert len(qq.axes[0].lines) == 1
    assert len(kinship.axes[0].images) == 1
    assert pca.axes[0].get_legend() is not None
    assert len(pca.axes[0].lines) == 2
    assert len(gs.axes[0].collections) == 1
    assert len(gs.axes[0].lines) == 1
    assert len(phenotype.axes[0].patches) > 30
    assert phenotype.axes[0].get_legend() is not None
    for figure in (qq, kinship, pca, gs, phenotype):
        plt.close(figure)


def test_static_manhattan_does_not_prepare_plotly_hover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pygapit.visualization.plots as visualization_plots

    def reject_plotly_hook(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("static rendering must not prepare Plotly hover text")

    monkeypatch.setattr(
        visualization_plots,
        "_plotly_manhattan_hook",
        reject_plotly_hook,
    )
    figure = manhattan(
        np.asarray(["s1", "s2"]),
        np.asarray(["1", "1"]),
        np.asarray([10.0, 20.0]),
        np.asarray([0.05, 0.1]),
        large_data="points",
    )

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_visualization_contracts_reject_ambiguous_shapes() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        kinship_heatmap(np.ones((2, 3)))
    with pytest.raises(ValueError, match="taxa must have length 51"):
        kinship_heatmap(np.eye(51), taxa=np.asarray(["sample"] * 50))
    with pytest.raises(ValueError, match="at least two components"):
        pca_plot_2d(np.ones((2, 1)), np.asarray([1.0]))
    with pytest.raises(ValueError, match="at least three components"):
        pca_plot_3d_interactive(np.ones((2, 2)), np.asarray([0.6, 0.4]))


@pytest.mark.parametrize(
    ("style", "package"),
    [("seaborn", "seaborn"), ("science", "scienceplots")],
)
def test_optional_static_styles_do_not_leak_rc_changes(
    style: str,
    package: str,
) -> None:
    if find_spec(package) is None:
        pytest.skip(f"{package} is not installed")

    facecolor = plt.rcParams["axes.facecolor"]
    font_family = list(plt.rcParams["font.family"])
    figure = manhattan(
        np.asarray(["s1"]),
        np.asarray(["1"]),
        np.asarray([10.0]),
        np.asarray([0.05]),
        style=cast("StaticStyle", style),
        large_data="points",
    )

    assert plt.rcParams["axes.facecolor"] == facecolor
    assert list(plt.rcParams["font.family"]) == font_family
    plt.close(figure)


def test_optional_style_falls_back_without_global_rc_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = importlib.import_module

    def import_without_seaborn(
        name: str,
        package: str | None = None,
    ) -> ModuleType:
        if name == "seaborn":
            raise ImportError(name)
        return original_import_module(name, package)

    facecolor = plt.rcParams["axes.facecolor"]
    monkeypatch.setattr(importlib, "import_module", import_without_seaborn)
    with pytest.warns(RuntimeWarning, match=r"pygapit-ng\[styles\]"):
        figure = manhattan(
            np.asarray(["s1"]),
            np.asarray(["1"]),
            np.asarray([10.0]),
            np.asarray([0.05]),
            style="seaborn",
            large_data="points",
        )

    assert plt.rcParams["axes.facecolor"] == facecolor
    plt.close(figure)


def test_manhattan_rejects_invalid_geometry() -> None:
    arguments = (
        np.asarray(["s1"]),
        np.asarray(["1"]),
        np.asarray([10.0]),
        np.asarray([0.05]),
    )

    with pytest.raises(ValueError, match="exactly two"):
        manhattan(
            *arguments,
            figsize=cast("tuple[float, float]", (8.0, 5.0, 3.0)),
        )
    with pytest.raises(ValueError, match="point_size"):
        manhattan(*arguments, point_size=0.0)


def test_interactive_manhattan_validates_effect_length() -> None:
    with pytest.raises(ValueError, match="effects"):
        manhattan(
            np.asarray(["s1", "s2"]),
            np.asarray(["1", "1"]),
            np.asarray([10.0, 20.0]),
            np.asarray([0.05, 1e-4]),
            mode="interactive",
            effects=np.asarray([0.1]),
        )


def test_interactive_manhattan_validates_maf_length() -> None:
    with pytest.raises(ValueError, match="MAF"):
        manhattan(
            np.asarray(["s1", "s2"]),
            np.asarray(["1", "1"]),
            np.asarray([10.0, 20.0]),
            np.asarray([0.05, 1e-4]),
            mode="interactive",
            maf=np.asarray([0.1]),
        )
