"""Process-wide configuration for the headless test suite."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
