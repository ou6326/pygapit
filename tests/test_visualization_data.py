"""Backend-independent visualization data contracts."""

from __future__ import annotations

import importlib
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, cast, overload

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pytest
from bokeh.models import ImageRGBA as BokehImageRGBA
from bokeh.plotting import figure as BokehFigure
from holoviews.core import Dimensioned
from matplotlib.figure import Figure

from pygapit.visualization import prepare_genomic_axis, prepare_manhattan_data
from pygapit.visualization.output import StaticStyle, save_plot
from pygapit.visualization.plots import (
    _iter_chromosome_selections,
    gs_scatter,
    kinship_heatmap,
    manhattan,
    pca_plot_2d,
    pca_plot_3d,
    phenotype_distribution,
    qq_plot,
)


class _HoloViewsRuntime(Protocol):
    @overload
    def render(self, obj: Dimensioned, *, backend: Literal["matplotlib"]) -> Figure: ...

    @overload
    def render(
        self, obj: Dimensioned, *, backend: Literal["plotly"]
    ) -> dict[str, object]: ...

    @overload
    def render(self, obj: Dimensioned, *, backend: Literal["bokeh"]) -> BokehFigure: ...


class _Dimension(Protocol):
    name: str


class _DimensionedView(Protocol):
    def dimensions(self) -> list[_Dimension]: ...


def _runtime_object(value: object) -> object:
    return value


holoviews = cast(_HoloViewsRuntime, _runtime_object(hv))


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


def test_manhattan_chromosome_selections_use_slices_only_for_contiguous_runs() -> None:
    contiguous = prepare_manhattan_data(
        np.asarray(["s1", "s2", "s3", "s4"]),
        np.asarray(["1", "1", "2", "2"]),
        np.asarray([10.0, 20.0, 5.0, 15.0]),
        np.asarray([0.5, 0.1, 0.05, 0.01]),
    )
    assert list(_iter_chromosome_selections(contiguous)) == [
        slice(0, 2),
        slice(2, 4),
    ]

    interleaved = prepare_manhattan_data(
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "2", "1"]),
        np.asarray([10.0, 5.0, 20.0]),
        np.asarray([0.5, 0.1, 0.01]),
    )
    selections = list(_iter_chromosome_selections(interleaved))
    assert len(selections) == 2
    assert all(isinstance(selection, np.ndarray) for selection in selections)
    np.testing.assert_array_equal(selections[0], [True, False, True])
    np.testing.assert_array_equal(selections[1], [False, True, False])


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


def test_manhattan_returns_one_holoviews_object_for_all_backends() -> None:
    store = cast(_HoloViewsStore, cast(object, hv.Store))
    previous_backend = store.current_backend
    extension = cast(_HoloViewsExtension, cast(object, hv.extension))
    extension("plotly")
    store.set_current_backend("plotly")
    try:
        plot = manhattan(
            np.asarray(["s1"]),
            np.asarray(["1"]),
            np.asarray([10.0]),
            np.asarray([0.05]),
            large_data="points",
        )
        assert store.current_backend == "plotly"
        matplotlib_figure = holoviews.render(plot, backend="matplotlib")
        plotly_figure = holoviews.render(plot, backend="plotly")
        bokeh_figure = holoviews.render(plot, backend="bokeh")
        assert isinstance(matplotlib_figure, Figure)
        assert isinstance(plotly_figure, dict)
        assert isinstance(bokeh_figure, BokehFigure)
        plt.close(matplotlib_figure)
    finally:
        store.set_current_backend(previous_backend)


def test_manhattan_keeps_hover_metadata_in_dimensions() -> None:
    plot = manhattan(
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "1", "2"]),
        np.asarray([10.0, 20.0, 5.0]),
        np.asarray([0.05, 1e-4, 0.2]),
        effects=np.asarray([0.1, -0.2, 0.3]),
        maf=np.asarray([0.2, 0.3, 0.4]),
        large_data="points",
    )
    dimensions = {
        dimension.name
        for dimension in cast(_DimensionedView, _runtime_object(plot)).dimensions()
    }
    assert {"snp", "chromosome", "position", "p_value", "effect", "maf"} <= dimensions


def test_manhattan_rendering_is_selected_after_construction() -> None:
    arguments = (
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "1", "2"]),
        np.asarray([10.0, 20.0, 5.0]),
        np.asarray([0.05, 1e-4, 0.2]),
    )

    plot = manhattan(*arguments, large_data="points")
    static = holoviews.render(plot, backend="matplotlib")
    plotly = holoviews.render(plot, backend="plotly")
    bokeh = holoviews.render(plot, backend="bokeh")

    assert isinstance(static, Figure)
    assert isinstance(plotly, dict)
    assert isinstance(bokeh, BokehFigure)
    plt.close(static)


def test_manhattan_aggregate_uses_one_color_raster_and_keeps_hits() -> None:
    arguments = (
        np.asarray(["s1", "s2", "s3", "s4"]),
        np.asarray(["1", "2", "3", "3"]),
        np.asarray([10.0, 5.0, 5.0, 15.0]),
        np.asarray([0.05, 1e-4, 0.2, 0.3]),
    )
    plot = manhattan(*arguments, large_data="aggregate")
    static = holoviews.render(plot, backend="matplotlib")
    plotly = holoviews.render(plot, backend="plotly")
    bokeh = holoviews.render(plot, backend="bokeh")

    assert isinstance(static, Figure)
    assert isinstance(plotly, dict)
    assert isinstance(bokeh, BokehFigure)
    plotly_data = cast("list[dict[str, object]]", plotly["data"])
    assert len(plotly_data) >= 2
    assert len(static.axes[0].images) == 1
    plotly_layout = cast("dict[str, object]", plotly["layout"])
    plotly_images = cast("list[dict[str, object]]", plotly_layout["images"])
    assert len(plotly_images) == 1
    assert (
        sum(
            isinstance(
                cast(_BokehGlyphRenderer, cast(object, renderer)).glyph,
                BokehImageRGBA,
            )
            for renderer in bokeh.renderers
            if hasattr(renderer, "glyph")
        )
        == 1
    )
    plt.close(static)


def test_manhattan_auto_uses_datashader_at_marker_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pygapit.visualization.plots as visualization_plots

    monkeypatch.setattr(visualization_plots, "_AGGREGATE_MARKER_THRESHOLD", 3)
    plot = manhattan(
        np.asarray(["s1", "s2", "s3"]),
        np.asarray(["1", "1", "2"]),
        np.asarray([10.0, 20.0, 5.0]),
        np.asarray([0.05, 1e-4, 0.2]),
        large_data="auto",
    )

    figure = holoviews.render(plot, backend="matplotlib")
    assert len(figure.axes[0].images) == 1
    plt.close(figure)


def test_pca_3d_returns_holoviews_scatter_with_metadata() -> None:
    plot = pca_plot_3d(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.asarray([0.5, 0.3, 0.2]),
        taxa=np.asarray(["sample-a", "sample-b"]),
        groups=np.asarray(["group-1", "group-2"]),
    )

    dimensions = {
        dimension.name
        for dimension in cast(_DimensionedView, _runtime_object(plot)).dimensions()
    }
    assert dimensions >= {"taxa", "group"}
    figure = holoviews.render(plot, backend="plotly")
    assert isinstance(figure, dict)


def test_pca_3d_validates_hover_metadata_lengths() -> None:
    with pytest.raises(ValueError, match="taxa must have length 2"):
        pca_plot_3d(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.asarray([0.5, 0.3, 0.2]),
            taxa=np.asarray(["sample-a"]),
        )


def test_static_holoviews_diagnostics_cover_specialized_layers() -> None:
    qq = holoviews.render(qq_plot(np.asarray([0.5, 0.1, 0.01])), backend="matplotlib")
    kinship = holoviews.render(
        kinship_heatmap(
            np.asarray([[1.0, 0.2], [0.2, 1.0]]),
            taxa=np.asarray(["sample-a", "sample-b"]),
        ),
        backend="matplotlib",
    )
    pca = holoviews.render(
        pca_plot_2d(
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            np.asarray([0.6, 0.3]),
            taxa=np.asarray(["sample-a", "sample-b"]),
            groups=np.asarray(["group-1", "group-2"]),
        ),
        backend="matplotlib",
    )
    gs = holoviews.render(
        gs_scatter(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.1, 1.9, 3.2]),
        ),
        backend="matplotlib",
    )
    phenotype = holoviews.render(
        phenotype_distribution(
            np.asarray([1.0, 1.5, 2.0, 2.5]),
            significant_snp_geno=np.asarray([0, 1, 2, 2]),
        ),
        backend="matplotlib",
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


def test_visualization_contracts_reject_ambiguous_shapes() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        kinship_heatmap(np.ones((2, 3)))
    with pytest.raises(ValueError, match="taxa must have length 51"):
        kinship_heatmap(np.eye(51), taxa=np.asarray(["sample"] * 50))
    with pytest.raises(ValueError, match="at least two components"):
        pca_plot_2d(np.ones((2, 1)), np.asarray([1.0]))
    with pytest.raises(ValueError, match="at least three components"):
        pca_plot_3d(np.ones((2, 2)), np.asarray([0.6, 0.4]))


@pytest.mark.parametrize(
    ("style", "package"),
    [("seaborn", "seaborn"), ("science", "scienceplots")],
)
def test_optional_static_styles_do_not_leak_rc_changes(
    style: str,
    package: str,
    tmp_path: Path,
) -> None:
    if find_spec(package) is None:
        pytest.skip(f"{package} is not installed")

    facecolor = plt.rcParams["axes.facecolor"]
    font_family = list(plt.rcParams["font.family"])
    plot = manhattan(
        np.asarray(["s1"]),
        np.asarray(["1"]),
        np.asarray([10.0]),
        np.asarray([0.05]),
        large_data="points",
    )
    save_plot(plot, tmp_path / "styled.pdf", style=cast("StaticStyle", style))

    assert plt.rcParams["axes.facecolor"] == facecolor
    assert list(plt.rcParams["font.family"]) == font_family


def test_optional_style_falls_back_without_global_rc_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    plot = manhattan(
        np.asarray(["s1"]),
        np.asarray(["1"]),
        np.asarray([10.0]),
        np.asarray([0.05]),
        large_data="points",
    )
    with pytest.warns(RuntimeWarning, match=r"pygapit-ng\[styles\]"):
        save_plot(plot, tmp_path / "fallback.pdf", style="seaborn")

    assert plt.rcParams["axes.facecolor"] == facecolor


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


def test_manhattan_validates_effect_length() -> None:
    with pytest.raises(ValueError, match="effects"):
        manhattan(
            np.asarray(["s1", "s2"]),
            np.asarray(["1", "1"]),
            np.asarray([10.0, 20.0]),
            np.asarray([0.05, 1e-4]),
            effects=np.asarray([0.1]),
        )


def test_manhattan_validates_maf_length() -> None:
    with pytest.raises(ValueError, match="MAF"):
        manhattan(
            np.asarray(["s1", "s2"]),
            np.asarray(["1", "1"]),
            np.asarray([10.0, 20.0]),
            np.asarray([0.05, 1e-4]),
            maf=np.asarray([0.1]),
        )
