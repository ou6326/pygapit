"""Check backend capabilities at actual call sites, including rejected calls."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import pytest


class _Position(TypedDict):
    line: int


class _Range(TypedDict):
    start: _Position


class _Diagnostic(TypedDict):
    severity: str
    range: _Range


class _Report(TypedDict):
    generalDiagnostics: list[_Diagnostic]


def test_backend_capability_typechecking(tmp_path: Path) -> None:
    pytest.importorskip("basedpyright")
    source = """from typing import assert_type
import numpy as np
from bokeh.plotting import figure as BokehFigure
from plotly.graph_objects import Figure as PlotlyFigure
from pygapit import pca_plot_3d, qq_plot, save_plot
from pygapit.visualization.view import OutputBackend, ThreeDBackend, Visualization
p = qq_plot(np.asarray([.5, .1]), backend="bokeh")
q = pca_plot_3d(np.ones((3, 3)), np.asarray([.5, .3, .2]))
assert_type(p, Visualization[OutputBackend])
assert_type(q, Visualization[ThreeDBackend])
assert_type(p.render("bokeh"), BokehFigure)
assert_type(p.render("plotly"), PlotlyFigure)
save_plot(p, "plot.html")
save_plot(p, "plot.pdf", style="science")
save_plot(q, "pca.html", backend="plotly")
p.backend = "plotly"
q.backend = "matplotlib"
pca_plot_3d(np.ones((3, 3)), np.ones(3), backend="bokeh")  # invalid
q.backend = "bokeh"  # invalid
q.render("bokeh")  # invalid
save_plot(q, "pca.html", backend="bokeh")  # invalid
save_plot(p, "plot.html", backend="bokeh", style="science")  # invalid
"""
    probe = tmp_path / "visualization_calls.py"
    probe.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "basedpyright",
            "--outputjson",
            "--pythonpath",
            sys.executable,
            str(probe),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    report: _Report = json.loads(result.stdout)
    errors = {
        diagnostic["range"]["start"]["line"]
        for diagnostic in report["generalDiagnostics"]
        if diagnostic["severity"] == "error"
    }
    expected = {
        index for index, line in enumerate(source.splitlines()) if "# invalid" in line
    }
    assert result.returncode == 1, result.stdout + result.stderr
    assert errors == expected, result.stdout
