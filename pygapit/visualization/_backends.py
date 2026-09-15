"""Narrow adapters around backend-specific plotting runtimes."""

from __future__ import annotations

import importlib
import typing as t
import warnings
from contextlib import contextmanager
from typing import Literal, Protocol, cast

if t.TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from contextlib import AbstractContextManager
    from os import PathLike

    from matplotlib.axis import Tick
    from matplotlib.collections import FillBetweenPolyCollection, PathCollection
    from matplotlib.colorbar import Colorbar
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.spines import Spine
    from matplotlib.text import Text
    from plotly.graph_objs._figure import Figure as PlotlyFigure

    from .._typing import FloatMatrix, FloatVector

__all__ = (
    "Axes",
    "PyplotModule",
    "StaticStyle",
    "axes",
    "plotly_figure",
    "pyplot",
    "savefig",
)

StaticStyle = Literal["pygapit", "seaborn", "science"]


class Axes(Protocol):
    spines: Mapping[str, Spine]

    def scatter(self, *args: object, **kwargs: object) -> PathCollection: ...
    def plot(self, *args: object, **kwargs: object) -> list[Line2D]: ...
    def hist(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[FloatVector, FloatVector, BarContainer]: ...
    def imshow(self, *args: object, **kwargs: object) -> AxesImage: ...
    def fill_between(
        self,
        *args: object,
        **kwargs: object,
    ) -> FillBetweenPolyCollection: ...
    def legend(self, *args: object, **kwargs: object) -> Legend: ...
    def axhline(self, *args: object, **kwargs: object) -> Line2D: ...
    def axvline(self, *args: object, **kwargs: object) -> Line2D: ...
    def set_facecolor(self, *args: object, **kwargs: object) -> None: ...
    def set_title(self, *args: object, **kwargs: object) -> Text: ...
    def set_xlabel(self, *args: object, **kwargs: object) -> Text: ...
    def set_ylabel(self, *args: object, **kwargs: object) -> Text: ...
    def set_xlim(self, *args: object, **kwargs: object) -> tuple[float, float]: ...
    def set_ylim(self, *args: object, **kwargs: object) -> tuple[float, float]: ...
    def set_xticks(self, *args: object, **kwargs: object) -> list[Tick]: ...
    def set_yticks(self, *args: object, **kwargs: object) -> list[Tick]: ...
    def set_xticklabels(self, *args: object, **kwargs: object) -> list[Text]: ...
    def set_yticklabels(self, *args: object, **kwargs: object) -> list[Text]: ...


class _FigureWriter(Protocol):
    def savefig(
        self,
        path: str | PathLike[str],
        *,
        dpi: int,
        bbox_inches: str,
    ) -> None: ...


class _SeabornModule(Protocol):
    def axes_style(self, style: str) -> Mapping[str, object]: ...


class _MatplotlibStyleLibrary(Protocol):
    def context(self, style: list[str]) -> AbstractContextManager[None]: ...


class _MatplotlibContext(Protocol):
    style: _MatplotlibStyleLibrary

    def rc_context(
        self,
        rc: Mapping[str, object],
    ) -> AbstractContextManager[None]: ...


class _ColorMap(Protocol):
    def __call__(self, values: FloatVector) -> FloatMatrix: ...


class _ColorMapRegistry(Protocol):
    tab10: _ColorMap


class PyplotModule(_MatplotlibContext, Protocol):
    cm: _ColorMapRegistry

    def subplots(self, **kwargs: object) -> tuple[Figure, object]: ...
    def tight_layout(self) -> None: ...
    def close(self, figure: Figure) -> None: ...
    def colorbar(
        self,
        mappable: AxesImage,
        *,
        ax: Axes,
        shrink: float,
        label: str,
    ) -> Colorbar: ...


def _runtime_object(value: object) -> object:
    return value


def axes(value: object) -> Axes:
    return cast(Axes, value)


def savefig(
    figure: Figure,
    path: str | PathLike[str],
    *,
    dpi: int = 150,
    bbox_inches: str,
) -> None:
    cast(_FigureWriter, cast(object, figure)).savefig(
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
    )


def pyplot() -> PyplotModule:
    import matplotlib.pyplot as plt

    return cast(PyplotModule, _runtime_object(plt))


def plotly_figure(state: Mapping[str, object]) -> PlotlyFigure:
    """Convert HoloViews Plotly renderer state to its public figure type."""
    import plotly.graph_objects as go

    return go.Figure(state)


@contextmanager
def matplotlib_style(style: StaticStyle) -> Generator[None]:
    """Apply an optional style without changing process-wide defaults."""
    if style == "pygapit":
        yield
        return

    try:
        if style == "seaborn":
            seaborn = cast(
                _SeabornModule,
                cast(object, importlib.import_module("seaborn")),
            )
            with pyplot().rc_context(seaborn.axes_style("white")):
                yield
            return

        if style == "science":
            importlib.import_module("scienceplots")
            with pyplot().style.context(["science", "no-latex"]):
                yield
            return

        raise ValueError("style must be 'pygapit', 'seaborn', or 'science'")
    except ImportError:
        warnings.warn(
            f"{style} style is unavailable; install pygapit-ng[styles]. "
            "Falling back to the pyGAPIT style.",
            RuntimeWarning,
            stacklevel=3,
        )
        yield
