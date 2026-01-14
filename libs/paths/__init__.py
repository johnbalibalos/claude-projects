"""
Path resolution utilities for consistent imports across run scripts.

Solves the problem of different scripts using different sys.path.insert patterns,
which can cause class identity issues (same class imported via different paths
becomes two different classes).

Usage:
    from pathlib import Path

    # In a run script, add these lines at the top:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "libs"))

    from paths import setup_project_paths
    setup_project_paths(__file__)

    # Now imports work consistently:
    from curation.omip_extractor import load_all_test_cases
    from hypothesis_pipeline import HypothesisPipeline
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start_path: Path | str) -> Path:
    """
    Find the repository root by looking for .git directory.

    Args:
        start_path: Starting path to search from (usually __file__)

    Returns:
        Path to repository root

    Raises:
        FileNotFoundError: If no .git directory found
    """
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent

    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    raise FileNotFoundError(f"Could not find repository root from {start_path}")


def find_project_root(start_path: Path | str) -> Path:
    """
    Find the project root (directory containing src/).

    Walks up from start_path until it finds a directory with a src/ subdirectory.

    Args:
        start_path: Starting path to search from (usually __file__)

    Returns:
        Path to project root (parent of src/)

    Raises:
        FileNotFoundError: If no src/ directory found
    """
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent

    # Don't go past repo root
    try:
        repo_root = find_repo_root(start_path)
    except FileNotFoundError:
        repo_root = None

    while current != current.parent:
        if (current / "src").is_dir():
            return current
        if repo_root and current == repo_root:
            break
        current = current.parent

    raise FileNotFoundError(f"Could not find project root (directory with src/) from {start_path}")


def get_libs_path(start_path: Path | str) -> Path:
    """
    Get the path to the libs/ directory.

    Args:
        start_path: Starting path to search from (usually __file__)

    Returns:
        Path to libs/ directory
    """
    repo_root = find_repo_root(start_path)
    return repo_root / "libs"


def get_src_path(start_path: Path | str) -> Path:
    """
    Get the path to the project's src/ directory.

    Args:
        start_path: Starting path to search from (usually __file__)

    Returns:
        Path to src/ directory
    """
    project_root = find_project_root(start_path)
    return project_root / "src"


def setup_project_paths(
    script_path: Path | str,
    *,
    include_src: bool = True,
    include_libs: bool = True,
) -> dict[str, Path]:
    """
    Setup sys.path for consistent imports from a run script.

    This is the main function to call from run scripts. It adds the project's
    src/ directory and the repo's libs/ directory to sys.path in a consistent way.

    Args:
        script_path: Path to the calling script (pass __file__)
        include_src: Whether to add project src/ to path (default True)
        include_libs: Whether to add repo libs/ to path (default True)

    Returns:
        Dictionary with resolved paths:
        - repo_root: Path to repository root
        - project_root: Path to project root (may be same as repo_root)
        - src: Path to src/ directory (if include_src)
        - libs: Path to libs/ directory (if include_libs)

    Example:
        # At top of run script:
        import sys
        from pathlib import Path

        # Bootstrap: add libs to path so we can import paths module
        _libs = Path(__file__).resolve().parent
        while not (_libs / "libs").exists() and _libs != _libs.parent:
            _libs = _libs.parent
        sys.path.insert(0, str(_libs / "libs"))

        from paths import setup_project_paths
        paths = setup_project_paths(__file__)

        # Now imports work:
        from curation.schemas import TestCase
    """
    script_path = Path(script_path).resolve()

    result = {}

    try:
        result["repo_root"] = find_repo_root(script_path)
    except FileNotFoundError:
        pass

    try:
        result["project_root"] = find_project_root(script_path)
    except FileNotFoundError:
        pass

    # Add paths to sys.path (insert at position 0 for priority)
    # But check if already present to avoid duplicates

    if include_src and "project_root" in result:
        src_path = result["project_root"] / "src"
        result["src"] = src_path
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    if include_libs and "repo_root" in result:
        libs_path = result["repo_root"] / "libs"
        result["libs"] = libs_path
        libs_str = str(libs_path)
        if libs_str not in sys.path:
            sys.path.insert(0, libs_str)

    return result


__all__ = [
    "find_repo_root",
    "find_project_root",
    "get_libs_path",
    "get_src_path",
    "setup_project_paths",
]
