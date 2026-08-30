"""Runtime contracts for public numerical boundaries."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pygapit._typing import as_float_matrix, as_float_vector
from pygapit.gwas.glm import GLMResult, glm_gwas
from pygapit.stats.kinship import scale_kinship


def test_float_array_helpers_normalize_dtype_and_dimension() -> None:
    vector = as_float_vector([1, 2], name="phenotype")
    matrix = as_float_matrix([[1, 2], [3, 4]], name="genotype matrix")

    assert vector.dtype == np.float64
    assert vector.shape == (2,)
    assert matrix.dtype == np.float64
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize(
    ("converter", "values", "message"),
    [
        (as_float_vector, [[1.0]], "one-dimensional"),
        (as_float_matrix, [1.0], "two-dimensional"),
    ],
)
def test_float_array_helpers_reject_wrong_dimensions(
    converter: Callable[..., object], values: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        converter(values)


def test_glm_rejects_mismatched_sample_rows() -> None:
    with pytest.raises(ValueError, match="genotype matrix must have 4 rows"):
        glm_gwas(np.arange(4.0), np.ones((4, 1)), np.ones((3, 2)))


def test_scale_kinship_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="kinship matrix must be square"):
        scale_kinship(np.ones((2, 3)))


def test_model_results_are_immutable() -> None:
    source = np.ones(1)
    result = GLMResult(
        p_values=source,
        effects=np.zeros(1),
        se=np.ones(1),
        t_stats=np.zeros(1),
        r2_full=0.0,
    )

    field_name = "r2_full"
    with pytest.raises(FrozenInstanceError):
        setattr(result, field_name, 1.0)
    assert result.p_values is not source
    with pytest.raises(ValueError, match="read-only"):
        np.put(result.p_values, [0], [0.5])

    source[0] = 0.25
    assert result.p_values[0] == 1.0
