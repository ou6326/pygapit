"""Canonical genomic prediction fits and fold-local validation."""

from .blup import (
    GBLUPResult,
    SUPERSelectionResult,
    cblup,
    gblup,
    predict_new,
    sblup,
    select_super_qtns,
)
from .ridge import RRBLUPResult, rrblup
from .validation import (
    PredictionCVResult,
    cross_validate_gblup,
    cross_validate_rrblup,
)

__all__ = [
    "GBLUPResult",
    "PredictionCVResult",
    "RRBLUPResult",
    "SUPERSelectionResult",
    "cblup",
    "cross_validate_gblup",
    "cross_validate_rrblup",
    "gblup",
    "predict_new",
    "rrblup",
    "sblup",
    "select_super_qtns",
]
