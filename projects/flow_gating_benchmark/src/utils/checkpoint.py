"""
Generic checkpoint manager for resumable workflows.

Provides save/load functionality for any dataclass with to_dict/from_dict methods.
Used by PredictionCollector, BatchScorer, LLMJudge, and experiment runners.

Usage:
    from utils.checkpoint import CheckpointManager

    # Initialize
    manager = CheckpointManager(checkpoint_dir=Path("results/checkpoints"))

    # Save results
    manager.save(results, "predictions.json")

    # Load and resume
    results, completed_keys = manager.load_with_keys(
        "predictions.json",
        Prediction,
        key_fn=lambda p: p.key
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar


class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict."""

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Serializable: ...


T = TypeVar("T", bound=Serializable)


class CheckpointManager:
    """
    Generic checkpoint manager for resumable workflows.

    Supports:
    - Save/load lists of serializable objects
    - Track completed keys for resume logic
    - Atomic writes (write to temp, then rename)
    - Metadata (timestamps, counts)
    """

    def __init__(self, checkpoint_dir: Path | str | None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files. If None, checkpointing is disabled.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    @property
    def enabled(self) -> bool:
        """Whether checkpointing is enabled."""
        return self.checkpoint_dir is not None

    def save(
        self,
        results: list[T],
        filename: str,
        *,
        include_metadata: bool = True,
    ) -> Path | None:
        """
        Save results to checkpoint file.

        Args:
            results: List of serializable objects
            filename: Checkpoint filename (e.g., "predictions.json")
            include_metadata: Whether to include timestamp and count

        Returns:
            Path to saved file, or None if checkpointing disabled
        """
        if not self.enabled:
            return None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.checkpoint_dir / filename

        data = [r.to_dict() for r in results]

        if include_metadata:
            output = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "count": len(results),
                },
                "results": data,
            }
        else:
            output = data

        # Atomic write: write to temp file, then rename
        temp_path = filepath.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(output, f, indent=2)
        temp_path.rename(filepath)

        return filepath

    def load(
        self,
        filename: str,
        result_class: type[T],
    ) -> list[T]:
        """
        Load results from checkpoint file.

        Args:
            filename: Checkpoint filename
            result_class: Class to deserialize into (must have from_dict)

        Returns:
            List of deserialized objects (empty if file doesn't exist)
        """
        if not self.enabled:
            return []

        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            return []

        with open(filepath) as f:
            data = json.load(f)

        # Handle both formats: with metadata wrapper or raw list
        if isinstance(data, dict) and "results" in data:
            items = data["results"]
        else:
            items = data

        return [result_class.from_dict(item) for item in items]

    def load_with_keys(
        self,
        filename: str,
        result_class: type[T],
        key_fn: Callable[[T], tuple],
    ) -> tuple[list[T], set[tuple]]:
        """
        Load results and build set of completed keys for resume logic.

        Args:
            filename: Checkpoint filename
            result_class: Class to deserialize into
            key_fn: Function to extract unique key from result

        Returns:
            Tuple of (results list, set of completed keys)
        """
        results = self.load(filename, result_class)
        completed_keys = {key_fn(r) for r in results}
        return results, completed_keys

    def exists(self, filename: str) -> bool:
        """Check if checkpoint file exists."""
        if not self.enabled:
            return False
        return (self.checkpoint_dir / filename).exists()

    def get_metadata(self, filename: str) -> dict[str, Any] | None:
        """Get metadata from checkpoint file without loading all results."""
        if not self.enabled:
            return None

        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        if isinstance(data, dict) and "metadata" in data:
            return data["metadata"]

        # No metadata, return basic info
        items = data if isinstance(data, list) else data.get("results", [])
        return {"count": len(items)}

    def merge_checkpoints(
        self,
        filenames: list[str],
        result_class: type[T],
        key_fn: Callable[[T], tuple],
        output_filename: str,
    ) -> list[T]:
        """
        Merge multiple checkpoint files, deduplicating by key.

        Args:
            filenames: List of checkpoint filenames to merge
            result_class: Class to deserialize into
            key_fn: Function to extract unique key
            output_filename: Where to save merged results

        Returns:
            Merged and deduplicated results
        """
        all_results = []
        seen_keys = set()

        for filename in filenames:
            results = self.load(filename, result_class)
            for r in results:
                key = key_fn(r)
                if key not in seen_keys:
                    all_results.append(r)
                    seen_keys.add(key)

        self.save(all_results, output_filename)
        return all_results


# Convenience function for quick checkpoint operations
def create_checkpoint_manager(checkpoint_dir: Path | str | None) -> CheckpointManager:
    """Create a checkpoint manager (convenience wrapper)."""
    return CheckpointManager(checkpoint_dir)
