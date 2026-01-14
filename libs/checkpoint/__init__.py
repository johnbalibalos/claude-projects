"""
Checkpoint library for resumable workflows.

Provides two checkpoint patterns:
- CheckpointedRunner: Iterator-based, saves after each item (for long-running loops)
- CheckpointManager: List-based, saves/loads entire result sets (for batch operations)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict."""

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable": ...


S = TypeVar("S", bound=Serializable)


class CheckpointedRunner:
    """
    Iterator wrapper that enables resumable iteration over work items.

    Saves progress after each item, allowing long-running jobs to resume
    from where they left off after interruption.

    Usage:
        runner = CheckpointedRunner("my_experiment", checkpoint_dir="./checkpoints")

        for item in runner.iterate(work_items, key_fn=lambda x: x.id):
            result = process(item)
            runner.save_result(item.id, result)

        # On restart, already-completed items are skipped automatically
    """

    def __init__(self, name: str, checkpoint_dir: Path | str | None = None):
        """
        Initialize checkpointed runner.

        Args:
            name: Unique name for this checkpoint (used in filename)
            checkpoint_dir: Directory to store checkpoint files
        """
        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self._results: dict[str, Any] = {}
        self._completed: set[str] = set()
        self._total: int = 0
        self._current: int = 0

        # Load existing checkpoint if present
        self._load()

    @property
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint file."""
        return self.checkpoint_dir / f"{self.name}_checkpoint.json"

    def _load(self) -> None:
        """Load existing checkpoint data."""
        if not self.checkpoint_path.exists():
            return

        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            self._results = data.get("results", {})
            self._completed = set(data.get("completed", []))
        except (json.JSONDecodeError, KeyError):
            # Corrupted checkpoint, start fresh
            self._results = {}
            self._completed = set()

    def _save(self) -> None:
        """Save checkpoint data to disk."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "results": self._results,
            "completed": list(self._completed),
        }

        # Atomic write
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.rename(self.checkpoint_path)

    def iterate(
        self,
        items: list[T],
        key_fn: Callable[[T], str] | None = None,
    ) -> Iterator[T]:
        """
        Iterate over items, skipping already-completed ones.

        Args:
            items: List of work items to iterate over
            key_fn: Function to extract unique key from item (default: str(item))

        Yields:
            Items that haven't been completed yet
        """
        self._total = len(items)
        self._current = 0

        for item in items:
            key = key_fn(item) if key_fn else str(item)

            if key in self._completed:
                self._current += 1
                continue

            yield item
            self._current += 1

    def progress(self) -> tuple[int, int]:
        """
        Get current progress.

        Returns:
            Tuple of (completed_count, total_count)
        """
        return len(self._completed), self._total

    def save_result(self, key: str, result: Any) -> None:
        """
        Save a result and mark the item as completed.

        Args:
            key: Unique key for this result
            result: Result data (must be JSON-serializable)
        """
        self._results[key] = result
        self._completed.add(key)
        self._save()

    def get_all_results(self) -> dict[str, Any]:
        """Get all saved results."""
        return self._results.copy()

    def get_result(self, key: str) -> Any | None:
        """Get a specific result by key."""
        return self._results.get(key)

    def is_completed(self, key: str) -> bool:
        """Check if an item has been completed."""
        return key in self._completed

    def reset(self) -> None:
        """Reset checkpoint, clearing all progress."""
        self._results = {}
        self._completed = set()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


class CheckpointManager:
    """
    List-based checkpoint manager for batch save/load of serializable objects.

    Unlike CheckpointedRunner which saves incrementally during iteration,
    this class saves/loads entire result lists at once. Useful for batch
    operations like scoring or judging where you want to checkpoint after
    processing a full batch.

    Usage:
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
        results: list[S],
        filename: str,
        *,
        include_metadata: bool = True,
    ) -> Path | None:
        """
        Save results to checkpoint file.

        Args:
            results: List of serializable objects (must have to_dict method)
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
        result_class: type[S],
    ) -> list[S]:
        """
        Load results from checkpoint file.

        Args:
            filename: Checkpoint filename
            result_class: Class to deserialize into (must have from_dict classmethod)

        Returns:
            List of deserialized objects (empty if file doesn't exist or disabled)
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
        result_class: type[S],
        key_fn: Callable[[S], tuple],
    ) -> tuple[list[S], set[tuple]]:
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


__all__ = ["CheckpointedRunner", "CheckpointManager", "Serializable"]
