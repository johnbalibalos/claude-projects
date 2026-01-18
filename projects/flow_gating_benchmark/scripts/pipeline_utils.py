#!/usr/bin/env python3
"""
Shared utilities for pipeline scripts.

Provides common setup, I/O helpers, and progress callbacks used across
predict.py, score.py, and judge.py.
"""

import sys
from pathlib import Path

# Bootstrap libs path for shared utilities
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent.parent
sys.path.insert(0, str(_repo_root / "libs"))

# Setup project paths consistently
from paths import setup_project_paths  # noqa: E402

paths = setup_project_paths(__file__)
PROJECT_ROOT = paths["project_root"]


def print_phase(name: str):
    """Print a phase header."""
    print()
    print("=" * 60)
    print(f"PHASE: {name}")
    print("=" * 60)


def progress_callback(current: int, total: int, item):
    """Generic progress callback for collectors and scorers."""
    pct = (current / total) * 100 if total > 0 else 0
    if hasattr(item, "error") and item.error:
        print(f"  [{current}/{total}] ({pct:.0f}%) ERROR: {item.error[:50]}")
    else:
        print(f"  [{current}/{total}] ({pct:.0f}%) {getattr(item, 'test_case_id', 'unknown')}")


def ensure_parent_dir(path: Path) -> Path:
    """Ensure parent directory exists and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
