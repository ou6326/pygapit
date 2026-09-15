"""Output-boundary helpers for HoloViews plots."""

from __future__ import annotations

import importlib
import typing as t
import warnings
from contextlib import contextmanager
from typing import Literal, Protocol, cast

import holoviews as hv

if t.TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from contextlib import AbstractContextManager
    from os import PathLike

    from .plots import HoloViewsPlot

StaticStyle = Literal["pygapit", "seaborn", "science"]
OutputBackend = Literal["matplotlib", "bokeh", "plotly"]


class _SeabornModule(Protocol):
    def axes_style(self, style: str) -> Mapping[str, object]: ...


class _MatplotlibStyleLibrary(Protocol):
    def context(self, style: list[str]) -> AbstractContextManager[None]: ...


class _PyplotModule(Protocol):
    style: _MatplotlibStyleLibrary

    def rc_context(
        self,
        rc: Mapping[str, object],
    ) -> AbstractContextManager[None]: ...


class _HoloViewsModule(Protocol):
    def save(
        self,
        obj: HoloViewsPlot,
        filename: str | PathLike[str],
        *,
        backend: OutputBackend,
    ) -> object: ...


def _runtime_object(value: object) -> object:
    return value


def _pyplot() -> _PyplotModule:
    import matplotlib.pyplot as plt

    return cast(_PyplotModule, _runtime_object(plt))


@contextmanager
def matplotlib_style(style: StaticStyle) -> Generator[None]:
    """Apply an optional style without changing process-wide defaults."""
    if style == "pygapit":
        yield
        return

    try:
        match style:
            case "seaborn":
                seaborn = cast(
                    _SeabornModule,
                    cast(object, importlib.import_module("seaborn")),
                )
                with _pyplot().rc_context(seaborn.axes_style("white")):
                    yield
            case "science":
                importlib.import_module("scienceplots")
                with _pyplot().style.context(["science", "no-latex"]):
                    yield
    except ImportError:
        warnings.warn(
            f"{style} style is unavailable; install pygapit-ng[styles]. "
            "Falling back to the pyGAPIT style.",
            RuntimeWarning,
            stacklevel=3,
        )
        yield


def save_plot(
    plot: HoloViewsPlot,
    path: str | PathLike[str],
    *,
    backend: OutputBackend = "matplotlib",
    style: StaticStyle = "pygapit",
) -> None:
    """Save a HoloViews object at pyGAPIT's automatic-output boundary."""
    holoviews = cast(_HoloViewsModule, _runtime_object(hv))
    if backend == "matplotlib":
        with matplotlib_style(style):
            holoviews.save(plot, path, backend=backend)
        return
    holoviews.save(plot, path, backend=backend)
