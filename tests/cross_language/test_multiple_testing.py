"""Cross-language multiple-testing alignment tests."""

from __future__ import annotations

import numpy as np
import numpy.testing as nt
import pytest
from numpy.typing import NDArray

from pygapit.stats.testing import benjamini_hochberg
from tests.cross_language.r_bridge import RBridge


@pytest.mark.parametrize(
    "p_values",
    [
        np.array([0.91, 0.001, 0.20, 0.04, 0.04, 0.0001, 0.65, 0.12], dtype=float),
        np.array([0.0, 1.0, 0.5, 0.5, 1e-300], dtype=float),
    ],
    ids=["typical", "boundaries"],
)
def test_benjamini_hochberg_matches_r_base(
    r_bridge: RBridge, p_values: NDArray[np.float64]
) -> None:
    """Compare Python BH correction with R's base p.adjust implementation."""
    p_adjust = r_bridge.function("stats::p.adjust")
    r_adjusted = r_bridge.float_array(
        p_adjust(r_bridge.float_vector(p_values), method="BH")
    )
    py_adjusted = benjamini_hochberg(p_values)

    nt.assert_allclose(py_adjusted, r_adjusted, rtol=1e-14, atol=1e-14)
