"""Contract tests for the large-marker benchmark's store-backend selection."""

from __future__ import annotations

from importlib.util import find_spec
from typing import cast

import pytest

from benchmarks import benchmark_large_scale
from benchmarks.benchmark_large_scale import (
    _STORE_BACKENDS,
    _STORE_SUFFIXES,
    _StoreBackend,
    run_large_scale_benchmark,
)

_STAGES = (
    "streaming_numeric_import",
    "vanraden_kinship_store",
    "pca_store",
    "glm_marker_scan_store",
)


def test_store_backend_suffixes_cover_every_backend() -> None:
    assert set(_STORE_SUFFIXES) == set(_STORE_BACKENDS)
    assert _STORE_SUFFIXES == {"numpy": "", "hdf5": ".h5", "zarr": ".zarr"}


@pytest.mark.parametrize(
    ("backend", "dependency"),
    [("numpy", None), ("hdf5", "h5py"), ("zarr", "zarr")],
)
def test_large_scale_benchmark_reports_each_store_backend(
    backend: str,
    dependency: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if dependency is not None and find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    monkeypatch.setattr(benchmark_large_scale, "_MARKER_SCALES", (200,))

    report = run_large_scale_benchmark(
        n_individuals=12,
        n_markers=200,
        seed=7,
        warmups=0,
        repeats=1,
        store_backend=cast(_StoreBackend, backend),
        render_backend="matplotlib",
    )

    workload = cast(dict[str, object], report["workload"])
    assert workload["store_backend"] == backend
    measurements = cast(list[dict[str, object]], report["numerical_measurements"])
    assert tuple(measurement["name"] for measurement in measurements) == _STAGES
