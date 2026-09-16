"""Output-boundary helpers for HoloViews plots."""

from __future__ import annotations

import importlib
import typing as t
import warnings
from contextlib import contextmanager
from os import fspath
from os.path import splitext
from typing import Literal, cast, overload

import holoviews as hv
import matplotlib.pyplot as plt

from .view import BackendT, OutputBackend, Visualization

if t.TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager
    from os import PathLike

StaticStyle = Literal["pygapit", "seaborn", "science"]
_STATIC_SUFFIXES = frozenset({
    ".eps",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".ps",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
})


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
    plot: Visualization[BackendT],
    path: str | PathLike[str],
    *,
    backend: None = None,
    style: StaticStyle = "pygapit",
) -> None: ...


@overload
def save_plot(
    plot: Visualization[BackendT],
    path: str | PathLike[str],
    *,
    backend: Literal["matplotlib"],
    style: StaticStyle = "pygapit",
) -> None: ...


@overload
def save_plot(
    plot: Visualization[BackendT],
    path: str | PathLike[str],
    *,
    backend: BackendT,
    style: Literal["pygapit"] = "pygapit",
) -> None: ...


def save_plot(
    plot: Visualization[BackendT],
    path: str | PathLike[str],
    *,
    backend: OutputBackend | None = None,
    style: StaticStyle = "pygapit",
) -> None:
    """Save using an explicit backend or a format-appropriate default.

    Static image extensions and Matplotlib styles select Matplotlib. Other
    formats use the visualization's current notebook backend.
    """
    path_string = fspath(path)
    if backend is not None:
        selected = backend
    elif style != "pygapit" or splitext(path_string)[1].lower() in _STATIC_SUFFIXES:
        selected = "matplotlib"
    else:
        selected = plot.backend
    if selected not in plot.supported_backends:
        raise ValueError(f"backend {selected!r} is not supported by this plot")
    if selected == "matplotlib":
        with matplotlib_style(style):
            hv.save(  # pyright: ignore[reportUnknownMemberType]
                plot.specification, path_string, backend=selected
            )
        return
    if style != "pygapit":
        raise ValueError("style is available only for the matplotlib backend")
    hv.save(  # pyright: ignore[reportUnknownMemberType]
        plot.specification, path_string, backend=selected
    )
