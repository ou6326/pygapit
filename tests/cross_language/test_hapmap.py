"""Alignment tests for GAPIT HapMap genotype numericalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest

from pygapit.io.formats import _numericalize_snp
from tests.cross_language.r_bridge import RBridge


@pytest.mark.parametrize(
    ("alleles", "bit", "major_allele_zero"),
    [
        (["AA", "AT", "TT", "NN"], 2, False),
        (["AA", "AT", "TA", "TT"], 2, False),
        (["AA", "AA", "AT", "TT", "TT", "TT"], 2, True),
        (["A", "G", "R", "N"], 1, False),
        (["AA", "NN"], 2, False),
    ],
)
def test_numericalize_snp_matches_bundled_r_gapit(
    r_bridge: RBridge,
    r_root: Path,
    alleles: list[str],
    bit: int,
    major_allele_zero: bool,
) -> None:
    """One- and two-bit calls follow GAPIT 3.5 coding and edge semantics."""
    r_numericalize = r_bridge.source_function(
        r_root,
        "GAPIT.Numericalization.R",
        "GAPIT.Numericalization",
    )
    quoted = ", ".join(f'"{allele}"' for allele in alleles)
    r_alleles = r_bridge.evaluate(f"c({quoted})")
    r_result = r_numericalize(
        x=r_alleles,
        bit=bit,
        impute="None",
        Major_allele_zero=major_allele_zero,
    )
    expected = r_bridge.float_array(r_result).reshape(-1)

    actual = _numericalize_snp(
        np.asarray(alleles, dtype=str),
        major_allele_zero=major_allele_zero,
    )

    nt.assert_allclose(actual, expected, equal_nan=True)
