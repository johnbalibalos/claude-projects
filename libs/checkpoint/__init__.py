"""
Checkpoint library for resumable workflows.

Provides CheckpointedRunner for iterating over work items with automatic
save/resume capability.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


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


__all__ = ["CheckpointedRunner"]
