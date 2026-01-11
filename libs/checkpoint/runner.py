"""
CheckpointedRunner - Resumable iterator for long-running workflows.

Saves progress incrementally to disk, allowing workflows to resume
after interruption without losing completed work.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class CheckpointMetadata:
    """Metadata about the checkpoint state."""
    experiment_name: str
    started_at: str
    last_updated: str
    total_items: int
    completed_items: int
    failed_items: int
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        # Handle old checkpoints without schema_version
        if "schema_version" not in data:
            data = {**data, "schema_version": "0.0.0"}
        return cls(**data)


class CheckpointedRunner(Generic[T, R]):
    """
    Resumable workflow runner with automatic checkpointing.

    Example:
        runner = CheckpointedRunner("my_experiment")

        items = [{"id": 1, "data": "..."}, {"id": 2, "data": "..."}]

        for item in runner.iterate(items, key_fn=lambda x: x['id']):
            result = expensive_api_call(item)
            runner.save_result(item['id'], result)

        # If interrupted and restarted, will skip already-completed items
        all_results = runner.get_all_results()
    """

    def __init__(
        self,
        experiment_name: str,
        checkpoint_dir: Path | None = None,
        auto_save: bool = True
    ):
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        # File paths
        self._meta_file = self.checkpoint_dir / f"{experiment_name}_meta.json"
        self._results_file = self.checkpoint_dir / f"{experiment_name}_results.jsonl"
        self._errors_file = self.checkpoint_dir / f"{experiment_name}_errors.jsonl"

        # In-memory state
        self._completed_keys: set[str] = set()
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._total_items = 0
        self._started_at: str | None = None

        # Load existing checkpoint if present
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load existing checkpoint from disk."""
        # Load metadata
        if self._meta_file.exists():
            with open(self._meta_file) as f:
                meta = CheckpointMetadata.from_dict(json.load(f))
                self._started_at = meta.started_at
                self._total_items = meta.total_items

        # Load completed results
        if self._results_file.exists():
            with open(self._results_file) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        key = record['key']
                        self._completed_keys.add(key)
                        self._results[key] = record['result']

        # Load errors
        if self._errors_file.exists():
            with open(self._errors_file) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self._errors[record['key']] = record['error']

        if self._completed_keys:
            print(f"[Checkpoint] Resuming '{self.experiment_name}': "
                  f"{len(self._completed_keys)} completed, "
                  f"{len(self._errors)} errors")

    def _save_metadata(self) -> None:
        """Save current metadata to disk."""
        meta = CheckpointMetadata(
            experiment_name=self.experiment_name,
            started_at=self._started_at or datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_items=self._total_items,
            completed_items=len(self._completed_keys),
            failed_items=len(self._errors)
        )
        with open(self._meta_file, 'w') as f:
            json.dump(meta.to_dict(), f, indent=2)

    def iterate(
        self,
        items: list[T],
        key_fn: Callable[[T], str]
    ) -> Iterator[T]:
        """
        Iterate over items, skipping already-completed ones.

        Args:
            items: List of items to process
            key_fn: Function to extract unique key from each item

        Yields:
            Items that haven't been completed yet
        """
        self._total_items = len(items)
        if not self._started_at:
            self._started_at = datetime.now().isoformat()

        self._save_metadata()

        skipped = 0
        for item in items:
            key = str(key_fn(item))
            if key in self._completed_keys:
                skipped += 1
                continue
            yield item

        if skipped > 0:
            print(f"[Checkpoint] Skipped {skipped} already-completed items")

    def save_result(self, key: str, result: Any) -> None:
        """Save a completed result to checkpoint."""
        key = str(key)
        self._completed_keys.add(key)
        self._results[key] = result

        if self.auto_save:
            # Append to results file
            with open(self._results_file, 'a') as f:
                record = {
                    'key': key,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                f.write(json.dumps(record) + '\n')

            self._save_metadata()

    def save_error(self, key: str, error: str) -> None:
        """Save an error for an item."""
        key = str(key)
        self._errors[key] = error

        if self.auto_save:
            with open(self._errors_file, 'a') as f:
                record = {
                    'key': key,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                f.write(json.dumps(record) + '\n')

            self._save_metadata()

    def is_completed(self, key: str) -> bool:
        """Check if an item has been completed."""
        return str(key) in self._completed_keys

    def get_result(self, key: str) -> Any | None:
        """Get result for a specific key."""
        return self._results.get(str(key))

    def get_all_results(self) -> dict[str, Any]:
        """Get all completed results."""
        return self._results.copy()

    def get_all_errors(self) -> dict[str, str]:
        """Get all errors."""
        return self._errors.copy()

    def progress(self) -> tuple[int, int]:
        """Return (completed, total) counts."""
        return len(self._completed_keys), self._total_items

    def clear(self) -> None:
        """Clear all checkpoint data."""
        for f in [self._meta_file, self._results_file, self._errors_file]:
            if f.exists():
                f.unlink()

        self._completed_keys.clear()
        self._results.clear()
        self._errors.clear()
        self._started_at = None
        print(f"[Checkpoint] Cleared '{self.experiment_name}'")


class BatchCheckpointedRunner(CheckpointedRunner[T, R]):
    """
    Extension with batch processing and progress reporting.
    """

    def iterate_with_progress(
        self,
        items: list[T],
        key_fn: Callable[[T], str],
        desc: str = "Processing"
    ) -> Iterator[tuple[int, int, T]]:
        """
        Iterate with progress info: (current_idx, total, item).
        """
        total = len(items)
        idx = 0

        for item in self.iterate(items, key_fn):
            idx += 1
            completed, _ = self.progress()
            print(f"[{completed + 1}/{total}] {desc}...", end=" ", flush=True)
            yield idx, total, item
