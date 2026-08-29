"""Cross-language GLM alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.gwas.glm import glm_gwas
from tests.cross_language.r_bridge import RBridge


@pytest.mark.parametrize(
    "with_covariate", [False, True], ids=["intercept-only", "covariate"]
)
def test_glm_matches_gapit_farmcpu_lm(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    fixed_covariate: NDArray[np.float64],
    with_covariate: bool,
) -> None:
    """Compare additive effects and p-values with GAPIT's GLM kernel."""
    farmcpu_lm = r_bridge.source_function(r_root, "GAPIT.FarmCPU.R", "FarmCPU.LM")
    if with_covariate:
        design = np.column_stack([np.ones(len(fixed_phenotype)), fixed_covariate])
        r_result = farmcpu_lm(
            r_bridge.float_vector(fixed_phenotype),
            w=r_bridge.matrix(fixed_covariate[:, np.newaxis]),
            GDP=r_bridge.matrix(fixed_genotypes),
            orientation="col",
            model="A",
            ncpus=1,
        )
    else:
        design = np.ones((len(fixed_phenotype), 1), dtype=float)
        r_result = farmcpu_lm(
            r_bridge.float_vector(fixed_phenotype),
            GDP=r_bridge.matrix(fixed_genotypes),
            orientation="col",
            model="A",
            ncpus=1,
        )
    r_p_values = r_bridge.float_array(r_bridge.component(r_result, "PF"))
    r_effects = r_bridge.float_array(r_bridge.component(r_result, "B")).reshape(-1)
    py_result = glm_gwas(fixed_phenotype, design, fixed_genotypes)

    nt.assert_allclose(py_result.p_values, r_p_values, rtol=1e-10, atol=1e-12)
    nt.assert_allclose(py_result.effects, r_effects, rtol=1e-10, atol=1e-12)
