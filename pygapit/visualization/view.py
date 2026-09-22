"""A HoloViews graph with a per-object notebook backend preference."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Literal, Protocol, cast, final, overload

import holoviews as hv
from holoviews.core import Dimensioned

if TYPE_CHECKING:
    from bokeh.plotting import figure as BokehFigure
    from matplotlib.figure import Figure as MatplotlibFigure
    from plotly.graph_objs._figure import Figure as PlotlyFigure

OutputBackend = Literal["matplotlib", "bokeh", "plotly"]
ThreeDBackend = Literal["matplotlib", "plotly"]

ALL_BACKENDS: tuple[OutputBackend, ...] = ("matplotlib", "bokeh", "plotly")
THREE_D_BACKENDS: tuple[ThreeDBackend, ...] = ("matplotlib", "plotly")

MimeBundle = tuple[dict[str, object], dict[str, object]]


class _NotebookRenderer(Protocol):
    def components(self, obj: Dimensioned) -> MimeBundle: ...


@final
class Visualization[BackendT: OutputBackend]:
    """Display one HoloViews specification with a per-object backend.

    Assigning ``backend`` changes only this object's notebook representation.
    ``render`` obtains a backend-native figure without exposing the wrapped
    HoloViews object during ordinary use.
    """

    __slots__: tuple[str, ...] = (
        "_backend",
        "_specification",
        "_supported_backends",
    )

    _backend: BackendT
    _specification: Dimensioned
    _supported_backends: tuple[BackendT, ...]

    def __init__(
        self,
        specification: Dimensioned,
        backend: BackendT,
        supported_backends: tuple[BackendT, ...],
    ) -> None:
        if backend not in supported_backends:
            raise ValueError(f"backend {backend!r} is not supported by this plot")
        self._specification = specification
        self._supported_backends = supported_backends
        self._backend = backend

    @property
    def backend(self) -> BackendT:
        """Backend used for notebook display and default interactive output."""
        return self._backend

    @backend.setter
    def backend(self, backend: BackendT) -> None:
        if backend not in self.supported_backends:
            raise ValueError(f"backend {backend!r} is not supported by this plot")
        self._backend = backend

    @property
    def specification(self) -> Dimensioned:
        """Underlying HoloViews graph for advanced composition or operations."""
        return self._specification

    @property
    def supported_backends(self) -> tuple[BackendT, ...]:
        """Backends accepted by this visualization."""
        return self._supported_backends

    @overload
    def render(self, backend: Literal["matplotlib"]) -> MatplotlibFigure: ...

    @overload
    def render(self, backend: Literal["plotly"]) -> PlotlyFigure: ...

    @overload
    def render(
        self: Visualization[OutputBackend], backend: Literal["bokeh"]
    ) -> BokehFigure: ...

    @overload
    def render(self, backend: BackendT | None = None) -> object: ...

    def render(  # pyright: ignore[reportInconsistentOverload]
        self, backend: OutputBackend | None = None
    ) -> object:
        """Render a backend-native figure without changing notebook display."""
        selected = self.backend if backend is None else backend
        if selected not in self.supported_backends:
            raise ValueError(f"backend {selected!r} is not supported by this plot")
        return hv.render(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            self._specification,
            backend=selected,
        )

    def _repr_mimebundle_(  # pyright: ignore[reportUnusedFunction]
        self,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
    ) -> MimeBundle:
        renderer = cast(Callable[[str], _NotebookRenderer], hv.renderer)(self.backend)
        data, metadata = renderer.components(self._specification)
        if include is not None:
            data = {key: value for key, value in data.items() if key in include}
        if exclude is not None:
            data = {key: value for key, value in data.items() if key not in exclude}
        return data, {key: value for key, value in metadata.items() if key in data}
