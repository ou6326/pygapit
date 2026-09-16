"""Formal storage-backend parity matrix for public model workflows."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import pandas as pd
import pytest

from pygapit import (
    GenotypeData,
    blink_gwas,
    cblup,
    cmlm_gwas,
    farmcpu_gwas,
    gblup,
    glm_gwas,
    mlm_gwas,
    mlmm_gwas,
    sblup,
    select_super_qtns,
)
from pygapit._typing import FloatMatrix, FloatVector, IntVector, StrVector
from pygapit.io.storage import (
    GenotypeStore,
    StorageBackend,
    open_genotype_store,
    write_genotype_store,
)
from pygapit.stats.kinship import vanraden_kinship

_MARKER_WORKSPACE_MIB = 0.0005

_Backend: TypeAlias = Literal["ndarray", "numpy", "hdf5", "zarr"]
_GenotypeInput: TypeAlias = FloatMatrix | GenotypeStore


@dataclass(frozen=True, slots=True)
class _ParityData:
    genotype: GenotypeData
    phenotype: FloatVector
    design: FloatMatrix
    chromosomes: StrVector
    positions: FloatVector
    kinship: FloatMatrix


@dataclass(frozen=True, slots=True)
class _ParityResult:
    continuous: dict[str, FloatVector]
    discrete: dict[str, IntVector]


_ModelRunner: TypeAlias = Callable[[_ParityData, _GenotypeInput], _ParityResult]


def _float_vector(*values: float) -> FloatVector:
    return np.asarray(values, dtype=np.float64)


@pytest.fixture(scope="module")
def parity_data() -> _ParityData:
    rng = np.random.default_rng(20260916)
    sample_count = 18
    marker_count = 12
    genotype_values = rng.integers(
        0,
        3,
        size=(sample_count, marker_count),
    ).astype(np.float64)
    chromosomes = np.asarray([str(index // 4 + 1) for index in range(marker_count)])
    positions = np.asarray(
        [(index % 4 + 1) * 100.0 for index in range(marker_count)],
        dtype=np.float64,
    )
    marker_map = pd.DataFrame({
        "SNP": [f"marker-{index}" for index in range(marker_count)],
        "Chromosome": chromosomes,
        "Position": positions,
    })
    taxa = np.asarray([f"sample-{index}" for index in range(sample_count)])
    genotype = GenotypeData(genotype_values, marker_map, taxa)
    trend = np.linspace(-1.0, 1.0, sample_count)
    design: FloatMatrix = np.column_stack([np.ones(sample_count), trend])
    phenotype = np.asarray(
        2.0
        + 0.4 * trend
        + 0.8 * genotype_values[:, 2]
        - 0.5 * genotype_values[:, 8]
        + rng.normal(0.0, 0.3, sample_count),
        dtype=np.float64,
    )
    kinship = vanraden_kinship(
        genotype_values,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityData(
        genotype=genotype,
        phenotype=phenotype,
        design=design,
        chromosomes=chromosomes,
        positions=positions,
        kinship=kinship,
    )


def _glm(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    result = glm_gwas(
        data.phenotype,
        data.design,
        genotype,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "p_values": result.p_values,
            "effects": result.effects,
            "se": result.se,
            "stats": result.t_stats,
            "r2_full": _float_vector(result.r2_full),
        },
        discrete={},
    )


def _mlm(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    result = mlm_gwas(
        data.phenotype,
        data.design,
        genotype,
        data.kinship,
        ngrids=20,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "p_values": result.p_values,
            "effects": result.effects,
            "se": result.se,
            "stats": result.stats,
            "variance": _float_vector(result.vg, result.ve, result.h2),
        },
        discrete={},
    )


def _cmlm(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    result = cmlm_gwas(
        data.phenotype,
        data.design,
        genotype,
        data.kinship,
        group_from=4,
        group_to=4,
        ngrids=20,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "p_values": result.p_values,
            "effects": result.effects,
            "se": result.se,
            "stats": result.stats,
            "variance": _float_vector(result.vg, result.ve, result.h2),
        },
        discrete={},
    )


def _mlmm(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    result = mlmm_gwas(
        data.phenotype,
        data.design,
        genotype,
        data.kinship,
        max_steps=3,
        ngrids=20,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "p_values": result.p_values,
            "effects": result.effects,
            "se": result.se,
            "stats": result.stats,
            "variance": _float_vector(result.vg, result.ve, result.h2),
        },
        discrete={
            "selected_qtns": result.selected_qtns,
            "n_steps": np.asarray([result.n_steps], dtype=np.int_),
        },
    )


def _farmcpu(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    result = farmcpu_gwas(
        data.phenotype,
        data.design,
        genotype,
        data.chromosomes,
        data.positions,
        max_iterations=3,
        p_threshold=0.2,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "p_values": result.p_values,
            "effects": result.effects,
            "se": result.se,
            "stats": result.t_stats,
            "variance": _float_vector(result.vg, result.ve, result.h2),
        },
        discrete={
            "selected_qtns": result.selected_qtns,
            "n_iterations": np.asarray([result.n_iterations], dtype=np.int_),
        },
    )


def _blink(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    result = blink_gwas(
        data.phenotype,
        data.design,
        genotype,
        max_iterations=3,
        p_threshold=0.2,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "p_values": result.p_values,
            "effects": result.effects,
            "se": result.se,
            "stats": result.t_stats,
        },
        discrete={
            "selected_qtns": result.selected_qtns,
            "n_iterations": np.asarray([result.n_iterations], dtype=np.int_),
        },
    )


def _super_sblup(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    scan = glm_gwas(
        data.phenotype,
        data.design,
        genotype,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    selection = select_super_qtns(
        data.phenotype,
        data.design,
        genotype,
        data.chromosomes,
        data.positions,
        scan.p_values,
        bin_size=50,
        candidate_counts=(1, 2, 3),
        ngrids=20,
    )
    prediction = sblup(
        data.phenotype,
        data.design,
        genotype,
        selection.qtn_indices,
        ngrids=20,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "selection_reml": selection.reml,
            "gebv": prediction.gebv,
            "prediction": prediction.prediction,
            "pev": prediction.pev,
            "variance": _float_vector(prediction.vg, prediction.ve, prediction.h2),
        },
        discrete={
            "selected_qtns": selection.qtn_indices,
            "candidate_counts": selection.candidate_counts,
        },
    )


def _gblup(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    kinship = vanraden_kinship(
        genotype,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    result = gblup(
        data.phenotype,
        data.design,
        kinship,
        ngrids=20,
    )
    return _ParityResult(
        continuous={
            "blue": result.blue,
            "blup": result.blup,
            "gebv": result.gebv,
            "prediction": result.prediction,
            "pev": result.pev,
            "variance": _float_vector(result.vg, result.ve, result.h2),
        },
        discrete={},
    )


def _cblup(data: _ParityData, genotype: _GenotypeInput) -> _ParityResult:
    result = cblup(
        data.phenotype,
        data.design,
        genotype,
        group_to=4,
        ngrids=20,
        marker_workspace_mib=_MARKER_WORKSPACE_MIB,
    )
    return _ParityResult(
        continuous={
            "blue": result.blue,
            "blup": result.blup,
            "gebv": result.gebv,
            "prediction": result.prediction,
            "pev": result.pev,
            "variance": _float_vector(result.vg, result.ve, result.h2),
        },
        discrete={},
    )


_MODEL_RUNNERS: tuple[tuple[str, _ModelRunner], ...] = (
    ("glm", _glm),
    ("mlm-emmax", _mlm),
    ("cmlm", _cmlm),
    ("mlmm", _mlmm),
    ("farmcpu", _farmcpu),
    ("blink", _blink),
    ("super-sblup", _super_sblup),
    ("gblup", _gblup),
    ("cblup", _cblup),
)

_BACKENDS = (
    pytest.param("ndarray", id="ndarray"),
    pytest.param("numpy", id="numpy-mmap"),
    pytest.param(
        "hdf5",
        id="hdf5",
        marks=pytest.mark.skipif(
            find_spec("h5py") is None,
            reason="h5py is not installed",
        ),
    ),
    pytest.param(
        "zarr",
        id="zarr",
        marks=pytest.mark.skipif(
            find_spec("zarr") is None,
            reason="zarr is not installed",
        ),
    ),
)


@contextmanager
def _backend_genotype(
    data: _ParityData,
    backend: _Backend,
    directory: Path,
) -> Generator[_GenotypeInput]:
    if backend == "ndarray":
        yield data.genotype.GD
        return

    storage_backend = cast(StorageBackend, backend)
    suffix = {"numpy": "", "hdf5": ".h5", "zarr": ".zarr"}[backend]
    path = directory / f"genotype{suffix}"
    write_genotype_store(
        path,
        data.genotype,
        backend=storage_backend,
        marker_chunk_size=3,
    )
    with open_genotype_store(path, backend=storage_backend) as store:
        np.testing.assert_array_equal(store.taxa, data.genotype.taxa)
        np.testing.assert_array_equal(
            store.marker_ids,
            data.genotype.GM["SNP"].to_numpy(dtype=str),
        )
        np.testing.assert_array_equal(store.chromosomes, data.chromosomes)
        np.testing.assert_array_equal(store.positions, data.positions)
        yield store


@pytest.fixture(scope="module")
def ndarray_results(parity_data: _ParityData) -> dict[str, _ParityResult]:
    return {
        name: runner(parity_data, parity_data.genotype.GD)
        for name, runner in _MODEL_RUNNERS
    }


@pytest.mark.parametrize(
    ("model_name", "runner"), _MODEL_RUNNERS, ids=lambda value: value
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_storage_backend_model_parity(
    tmp_path: Path,
    parity_data: _ParityData,
    ndarray_results: dict[str, _ParityResult],
    backend: _Backend,
    model_name: str,
    runner: _ModelRunner,
) -> None:
    expected = ndarray_results[model_name]
    with _backend_genotype(parity_data, backend, tmp_path) as genotype:
        actual = runner(parity_data, genotype)

    assert actual.continuous.keys() == expected.continuous.keys()
    assert actual.discrete.keys() == expected.discrete.keys()
    for name, expected_values in expected.continuous.items():
        np.testing.assert_allclose(
            actual.continuous[name],
            expected_values,
            rtol=1e-11,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"{model_name} {name} differs for {backend}",
        )
    for name, expected_values in expected.discrete.items():
        np.testing.assert_array_equal(
            actual.discrete[name],
            expected_values,
            err_msg=f"{model_name} {name} differs for {backend}",
        )
