"""
Experiment provenance tracking for reproducibility.

Captures git commit, dataset hash, and config at experiment start.
Enables reproducing exact conditions of any past experiment.

Usage:
    from utils.provenance import ExperimentContext

    # Create at experiment start
    ctx = ExperimentContext.create(
        ground_truth_dir=Path("data/ground_truth"),
        config={"models": ["claude-sonnet-cli"], "n_bootstrap": 3},
    )

    # Save to results directory
    ctx.save(output_dir)

    # Later: verify or reproduce
    # See experiment.json for git commit and dataset hash

Reproducing an experiment:
    1. Check out the code version:
       git checkout <commit_hash>

    2. Verify dataset hasn't changed:
       python -c "from utils.provenance import hash_directory; \\
                  print(hash_directory('data/ground_truth'))"
       # Should match dataset_hash in experiment.json

    3. If dataset changed, find when it matched:
       git log --oneline -- data/ground_truth/
       # Check out commit where ground truth had the right content

    4. Re-run the pipeline with same config from experiment.json

TODO: Add verify_experiment.py script to automate checking if current
      code/data matches a past experiment's provenance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> str | None:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty() -> bool:
    """Check if working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_directory(directory: Path) -> str:
    """
    Compute deterministic hash of directory contents.

    Hashes all files sorted by name, so result is reproducible.
    Returns first 12 chars of SHA256 for brevity.
    """
    directory = Path(directory)
    if not directory.exists():
        return "missing"

    sha256 = hashlib.sha256()

    # Sort files for deterministic ordering
    files = sorted(directory.glob("**/*"))
    for filepath in files:
        if filepath.is_file():
            # Include relative path in hash (so renames are detected)
            rel_path = filepath.relative_to(directory)
            sha256.update(str(rel_path).encode())
            sha256.update(hash_file(filepath).encode())

    return sha256.hexdigest()[:12]


@dataclass
class ExperimentContext:
    """
    Provenance context for an experiment run.

    Created once at experiment start, saved to experiment.json.
    All checkpoint files in the same experiment reference this context.
    """

    experiment_id: str
    started_at: datetime
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    dataset_hash: str
    dataset_path: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        ground_truth_dir: Path | str,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentContext:
        """
        Create experiment context capturing current state.

        Args:
            ground_truth_dir: Path to ground truth data directory
            config: Experiment configuration to record
            experiment_id: Optional custom ID (auto-generated if None)

        Returns:
            ExperimentContext with current git/data state
        """
        ground_truth_dir = Path(ground_truth_dir)
        now = datetime.now()

        return cls(
            experiment_id=experiment_id or f"exp_{now:%Y%m%d_%H%M%S}",
            started_at=now,
            git_commit=get_git_commit(),
            git_branch=get_git_branch(),
            git_dirty=is_git_dirty(),
            dataset_hash=hash_directory(ground_truth_dir),
            dataset_path=str(ground_truth_dir),
            config=config or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "started_at": self.started_at.isoformat(),
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
                # How to reproduce:
                "_reproduce": f"git checkout {self.git_commit}" if self.git_commit else None,
            },
            "dataset": {
                "path": self.dataset_path,
                "hash": self.dataset_hash,
                # How to verify:
                "_verify": f"python -c \"from utils.provenance import hash_directory; print(hash_directory('{self.dataset_path}'))\"",
            },
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentContext:
        """Load from dictionary."""
        git = data.get("git", {})
        dataset = data.get("dataset", {})

        return cls(
            experiment_id=data["experiment_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            git_commit=git.get("commit"),
            git_branch=git.get("branch"),
            git_dirty=git.get("dirty", False),
            dataset_hash=dataset.get("hash", "unknown"),
            dataset_path=dataset.get("path", ""),
            config=data.get("config", {}),
        )

    def save(self, output_dir: Path | str) -> Path:
        """
        Save experiment.json to output directory.

        Args:
            output_dir: Directory to save experiment.json

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / "experiment.json"
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, output_dir: Path | str) -> ExperimentContext | None:
        """
        Load experiment.json from output directory.

        Args:
            output_dir: Directory containing experiment.json

        Returns:
            ExperimentContext or None if file doesn't exist
        """
        filepath = Path(output_dir) / "experiment.json"
        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def print_summary(self) -> None:
        """Print human-readable summary to stdout."""
        dirty_marker = " (dirty)" if self.git_dirty else ""
        print(f"Experiment: {self.experiment_id}")
        print(f"  Started:  {self.started_at:%Y-%m-%d %H:%M:%S}")
        print(f"  Git:      {self.git_commit or 'unknown'}{dirty_marker}")
        print(f"  Branch:   {self.git_branch or 'unknown'}")
        print(f"  Dataset:  {self.dataset_hash} ({self.dataset_path})")
