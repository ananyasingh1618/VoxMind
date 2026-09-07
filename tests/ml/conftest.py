"""Ensures the repo root (where the `ml` package lives) is importable, and
that `voxmind` resolves via apps/api's editable install - these tests must
be run with that venv's Python:
    apps/api/.venv/bin/pytest tests/ml
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
