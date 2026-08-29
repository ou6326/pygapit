"""Checks for the pinned GAPIT 3.5 reference source."""

from __future__ import annotations

import subprocess
from pathlib import Path

EXPECTED_GAPIT_COMMIT = "8d6651c719484c9f6c844144783dca1e4ef85b3e"


def test_gapit_checkout_is_pinned(project_root: Path) -> None:
    """Ensure comparisons use the exact commit behind the GAPIT3.5 tag."""
    completed = subprocess.run(
        ["git", "-C", str(project_root / "GAPIT"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == EXPECTED_GAPIT_COMMIT
