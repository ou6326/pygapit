"""Output-boundary helpers for HoloViews plots."""

from __future__ import annotations

import importlib
import typing as t
import warnings
from contextlib import contextmanager
from os import fspath
from typing import Literal, cast, overload

import holoviews as hv
import matplotlib.pyplot as plt
from holoviews.core import Dimensioned

if t.TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager
    from os import PathLike

StaticStyle = Literal["pygapit", "seaborn", "science"]
OutputBackend = Literal["matplotlib", "bokeh", "plotly"]


@contextmanager
def matplotlib_style(style: StaticStyle) -> Generator[None]:
    """Apply an optional style without changing process-wide defaults."""
    if style == "pygapit":
        yield
        return

    try:
        match style:
            case "seaborn":
                axes_style = cast(
                    "Callable[[str], AbstractContextManager[None]]",
                    importlib.import_module("seaborn").axes_style,
                )
                with axes_style("white"):
                    yield
            case "science":
                importlib.import_module("scienceplots")
                with plt.style.context(["science", "no-latex"]):
                    yield
    except ImportError:
        warnings.warn(
            f"{style} style is unavailable; install pygapit-ng[styles]. "
            "Falling back to the pyGAPIT style.",
            RuntimeWarning,
            stacklevel=3,
        )
        yield


@overload
def save_plot(
    plot: Dimensioned,
    path: str | PathLike[str],
    *,
    backend: Literal["matplotlib"] = "matplotlib",
    style: StaticStyle = "pygapit",
) -> None: ...


@overload
def save_plot(
    plot: Dimensioned,
    path: str | PathLike[str],
    *,
    backend: Literal["bokeh", "plotly"],
    style: Literal["pygapit"] = "pygapit",
) -> None: ...


def save_plot(
    plot: Dimensioned,
    path: str | PathLike[str],
    *,
    backend: OutputBackend = "matplotlib",
    style: StaticStyle = "pygapit",
) -> None:
    """Save a HoloViews object at pyGAPIT's automatic-output boundary."""
    if backend == "matplotlib":
        with matplotlib_style(style):
            hv.save(  # pyright: ignore[reportUnknownMemberType]
                plot, fspath(path), backend=backend
            )
        return
    if style != "pygapit":
        raise ValueError("style is available only for the matplotlib backend")
    hv.save(  # pyright: ignore[reportUnknownMemberType]
        plot, fspath(path), backend=backend
    )
