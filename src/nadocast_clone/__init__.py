"""Nadocast-style severe-weather ML forecaster."""
from __future__ import annotations

import os

# Force UTF-8 mode on Windows so herbie-data's first-run console output
# doesn't crash with UnicodeEncodeError under cp1252. Must run before any
# herbie import. setdefault preserves any explicit PYTHONUTF8 the user set.
os.environ.setdefault("PYTHONUTF8", "1")

__version__ = "0.1.0"
