"""Canonical ridge-regression BLUP fit."""

from __future__ import annotations

from dataclasses import dataclass

from .._typing import FloatVector, Matrix, Vector, readonly_copy
from .validation import _genotypes, _phenotype, _ridge_components


@dataclass(frozen=True, slots=True)
class RRBLUPResult:
    """Full-data RR-BLUP components under an unpenalized intercept."""

    gebv: FloatVector
    prediction: FloatVector
    intercept: float
    effects: FloatVector
    marker_means: FloatVector
    regularization: float
    method: str = "RR-BLUP"

    def __post_init__(self) -> None:
        for field in ("gebv", "prediction", "effects", "marker_means"):
            object.__setattr__(self, field, readonly_copy(getattr(self, field)))


def rrblup(
    phenotype: Vector,
    genotype: Matrix,
    *,
    lambda_: float | None = None,
) -> RRBLUPResult:
    """Fit full-data RR-BLUP with centered markers and an intercept.

    Missing marker values are imputed to their full-training column means.
    ``gebv`` is the centered genomic component; ``prediction`` adds the
    unpenalized intercept. When ``lambda_`` is omitted, EMMA REML estimates
    delta from ``K = Z Z' / m`` and the marker penalty is ``m * delta``.
    """
    y = _phenotype(phenotype)
    markers = _genotypes(genotype, len(y))
    gebv, prediction, intercept, effects, means, regularization = _ridge_components(
        y, markers, markers, lambda_
    )
    return RRBLUPResult(
        gebv=gebv,
        prediction=prediction,
        intercept=intercept,
        effects=effects,
        marker_means=means,
        regularization=regularization,
    )
