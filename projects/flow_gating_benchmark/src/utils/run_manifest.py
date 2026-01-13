"""
Run manifest for data provenance tracking.

Each experiment run creates a manifest.json with full provenance metadata.
Data files reference the run_id, avoiding redundant metadata in every row.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class RunManifest:
    """Immutable provenance metadata for an experiment run."""

    run_id: str
    schema_version: str
    pipeline_version: str
    git_commit: str  # Full 40-char hash
    git_dirty: bool
    git_branch: str
    config_hash: str  # SHA256 of frozen config
    started_at: str
    completed_at: str | None
    models: list[str]
    test_cases_dir: str
    n_bootstrap: int
    status: str  # "running", "completed", "failed"
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(**data)

    @classmethod
    def create(
        cls,
        models: list[str],
        test_cases_dir: str,
        n_bootstrap: int,
        config: dict[str, Any] | None = None,
    ) -> RunManifest:
        """Create a new manifest with current git info."""
        return cls(
            run_id=str(uuid.uuid4())[:8],  # Short UUID for readability
            schema_version=SCHEMA_VERSION,
            pipeline_version=_get_pipeline_version(),
            git_commit=_get_git_commit_full(),
            git_dirty=_is_git_dirty(),
            git_branch=_get_git_branch(),
            config_hash=_hash_config(config) if config else "",
            started_at=datetime.now().isoformat(),
            completed_at=None,
            models=models,
            test_cases_dir=test_cases_dir,
            n_bootstrap=n_bootstrap,
            status="running",
        )

    def save(self, output_dir: Path) -> Path:
        """Save manifest to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, output_dir: Path) -> RunManifest | None:
        """Load manifest from output directory."""
        path = output_dir / "manifest.json"
        if not path.exists():
            return None
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def mark_completed(
        self,
        total: int,
        successful: int,
        failed: int,
        output_dir: Path,
    ) -> None:
        """Update manifest as completed and save."""
        self.completed_at = datetime.now().isoformat()
        self.status = "completed"
        self.total_predictions = total
        self.successful_predictions = successful
        self.failed_predictions = failed
        self.save(output_dir)

    def mark_failed(self, output_dir: Path, error: str = "") -> None:
        """Update manifest as failed and save."""
        self.completed_at = datetime.now().isoformat()
        self.status = f"failed: {error}" if error else "failed"
        self.save(output_dir)


def _get_git_commit_full() -> str:
    """Get full 40-char git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_git_branch() -> str:
    """Get current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _is_git_dirty() -> bool:
    """Check if working directory has uncommitted changes."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_pipeline_version() -> str:
    """Get pipeline version from git tag or commit."""
    try:
        # Try to get most recent tag
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return tag
    except subprocess.CalledProcessError:
        # Fall back to commit short hash
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"


def _hash_config(config: dict[str, Any]) -> str:
    """Create SHA256 hash of config for drift detection."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def save_frozen_config(config: dict[str, Any], output_dir: Path) -> Path:
    """Save a frozen copy of the config used for this run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def validate_manifests_compatible(
    manifests: list[RunManifest],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """
    Check if manifests are compatible for merging.

    Args:
        manifests: List of manifests to compare
        strict: If True, require exact git commit match

    Returns:
        (is_compatible, list of warnings)
    """
    if not manifests:
        return True, []

    warnings = []

    # Check schema versions
    schemas = {m.schema_version for m in manifests}
    if len(schemas) > 1:
        warnings.append(f"Mixed schema versions: {schemas}")

    # Check git commits
    commits = {m.git_commit for m in manifests}
    if len(commits) > 1:
        msg = f"Different git commits: {[c[:8] for c in commits]}"
        if strict:
            return False, [msg]
        warnings.append(msg)

    # Check for dirty runs
    dirty_runs = [m.run_id for m in manifests if m.git_dirty]
    if dirty_runs:
        warnings.append(f"Runs with uncommitted changes: {dirty_runs}")

    # Check config hashes
    configs = {m.config_hash for m in manifests if m.config_hash}
    if len(configs) > 1:
        warnings.append(f"Different configs used: {configs}")

    return True, warnings
