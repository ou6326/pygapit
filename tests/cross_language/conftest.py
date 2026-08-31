"""Shared fixtures for GAPIT 3.5 cross-language alignment tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from tests.cross_language.r_bridge import RBridge, RUnavailableError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
R_ROOT = PROJECT_ROOT / "GAPIT" / "R"
REQUIRE_R = os.environ.get("PYGAPIT_REQUIRE_R_ALIGNMENT", "0") == "1"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the Python repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def r_root() -> Path:
    """Return the directory containing the pinned GAPIT R sources."""
    return R_ROOT


@pytest.fixture(scope="session")
def r_bridge() -> RBridge:
    """Return a typed R bridge or skip/fail with an actionable message."""
    try:
        return RBridge.connect()
    except RUnavailableError as exc:
        message = str(exc)
        if REQUIRE_R:
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(scope="session")
def fixed_genotypes() -> NDArray[np.float64]:
    """Return a deterministic 0/1/2 matrix with no monomorphic markers."""
    return np.array(
        [
            [0, 1, 2, 0, 1, 2],
            [1, 1, 2, 1, 0, 2],
            [2, 0, 1, 2, 1, 1],
            [0, 2, 0, 1, 2, 0],
            [1, 0, 1, 2, 0, 1],
            [2, 2, 2, 0, 2, 0],
            [0, 1, 0, 2, 1, 2],
            [1, 2, 1, 1, 2, 1],
            [2, 0, 2, 0, 0, 2],
            [0, 2, 1, 1, 1, 0],
            [1, 0, 0, 2, 2, 1],
            [2, 1, 2, 0, 0, 2],
        ],
        dtype=float,
    )


@pytest.fixture(scope="session")
def fixed_phenotype() -> NDArray[np.float64]:
    """Return a deterministic quantitative phenotype."""
    return np.array(
        [2.1, 2.8, 3.7, 1.4, 2.5, 3.2, 1.8, 3.0, 3.4, 1.6, 2.3, 3.6],
        dtype=float,
    )


@pytest.fixture(scope="session")
def fixed_covariate(fixed_phenotype: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a deterministic continuous covariate."""
    return np.linspace(-1.0, 1.0, len(fixed_phenotype), dtype=float)


@pytest.fixture(scope="session")
def fixed_gapit_inputs(
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return labeled top-level inputs sharing the numerical R fixtures."""
    taxa = [f"T{index + 1}" for index in range(len(fixed_phenotype))]
    marker_names = [f"s{index + 1}" for index in range(fixed_genotypes.shape[1])]
    phenotype = pd.DataFrame({"Taxa": taxa, "Trait": fixed_phenotype})
    genotype = pd.DataFrame(fixed_genotypes, columns=marker_names)
    genotype.insert(0, "Taxa", taxa)
    marker_map = pd.DataFrame(
        {
            "SNP": marker_names,
            "Chromosome": [1, 1, 1, 2, 2, 2],
            "Position": [100, 200, 1500, 100, 200, 1500],
        }
    )
    return phenotype, genotype, marker_map
