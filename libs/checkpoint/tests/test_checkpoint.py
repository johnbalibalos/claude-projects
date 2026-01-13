"""
Tests for CheckpointedRunner.

Covers:
- Basic iteration
- Resume from checkpoint (skipping completed items)
- Save/load cycle
- Corrupted checkpoint recovery
- Custom key functions
- Progress tracking
- Reset functionality
- Edge cases
"""

import json
from dataclasses import dataclass
from pathlib import Path

from checkpoint import CheckpointedRunner


@dataclass
class WorkItem:
    """Sample work item for testing."""
    id: str
    value: int


class TestBasicIteration:
    """Tests for basic iteration without checkpointing."""

    def test_iterates_all_items(self, tmp_path: Path):
        """Should yield all items when no checkpoint exists."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = ["a", "b", "c"]

        result = list(runner.iterate(items))

        assert result == ["a", "b", "c"]

    def test_empty_items_yields_nothing(self, tmp_path: Path):
        """Should handle empty item list."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        result = list(runner.iterate([]))

        assert result == []

    def test_single_item(self, tmp_path: Path):
        """Should handle single item."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        result = list(runner.iterate(["only"]))

        assert result == ["only"]


class TestResumeFromCheckpoint:
    """Tests for resuming from existing checkpoint."""

    def test_skips_completed_items(self, tmp_path: Path):
        """Should skip items that were already completed."""
        # First run: complete some items
        runner1 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = ["a", "b", "c", "d"]

        for item in runner1.iterate(items):
            if item in ["a", "b"]:
                runner1.save_result(item, {"done": True})

        # Second run: should skip a and b
        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        result = list(runner2.iterate(items))

        assert result == ["c", "d"]

    def test_resumes_from_middle(self, tmp_path: Path):
        """Should resume from where we left off."""
        runner1 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = list(range(10))

        # Complete first 5
        for item in runner1.iterate(items):
            runner1.save_result(str(item), item * 2)
            if item == 4:
                break

        # Resume - should get 5-9
        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        result = list(runner2.iterate(items))

        assert result == [5, 6, 7, 8, 9]

    def test_all_completed_yields_nothing(self, tmp_path: Path):
        """Should yield nothing if all items already completed."""
        runner1 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = ["a", "b"]

        for item in runner1.iterate(items):
            runner1.save_result(item, True)

        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        result = list(runner2.iterate(items))

        assert result == []


class TestSaveAndLoadResults:
    """Tests for saving and loading results."""

    def test_save_result_persists(self, tmp_path: Path):
        """Results should persist across runner instances."""
        runner1 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        runner1.save_result("key1", {"value": 42})
        runner1.save_result("key2", {"value": 99})

        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert runner2.get_result("key1") == {"value": 42}
        assert runner2.get_result("key2") == {"value": 99}

    def test_get_all_results(self, tmp_path: Path):
        """Should return all saved results."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        runner.save_result("a", 1)
        runner.save_result("b", 2)
        runner.save_result("c", 3)

        results = runner.get_all_results()

        assert results == {"a": 1, "b": 2, "c": 3}

    def test_get_all_results_returns_copy(self, tmp_path: Path):
        """Should return a copy, not the internal dict."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        runner.save_result("a", 1)

        results = runner.get_all_results()
        results["b"] = 2  # Modify the returned dict

        assert runner.get_result("b") is None  # Internal state unchanged

    def test_get_nonexistent_result(self, tmp_path: Path):
        """Should return None for missing keys."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert runner.get_result("missing") is None

    def test_is_completed(self, tmp_path: Path):
        """Should track completion status correctly."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert not runner.is_completed("key1")

        runner.save_result("key1", "done")

        assert runner.is_completed("key1")
        assert not runner.is_completed("key2")


class TestCorruptedCheckpointRecovery:
    """Tests for handling corrupted checkpoint files."""

    def test_recovers_from_invalid_json(self, tmp_path: Path):
        """Should start fresh if checkpoint contains invalid JSON."""
        checkpoint_path = tmp_path / "test_checkpoint.json"
        checkpoint_path.write_text("not valid json {{{")

        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert runner.get_all_results() == {}
        assert list(runner.iterate(["a", "b"])) == ["a", "b"]

    def test_recovers_from_missing_keys(self, tmp_path: Path):
        """Should handle checkpoint with unexpected structure."""
        checkpoint_path = tmp_path / "test_checkpoint.json"
        checkpoint_path.write_text('{"unexpected": "structure"}')

        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        # Should have empty results but not crash
        assert runner.get_all_results() == {}

    def test_recovers_from_empty_file(self, tmp_path: Path):
        """Should handle empty checkpoint file."""
        checkpoint_path = tmp_path / "test_checkpoint.json"
        checkpoint_path.write_text("")

        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert runner.get_all_results() == {}


class TestCustomKeyFunction:
    """Tests for custom key extraction functions."""

    def test_custom_key_fn_with_dataclass(self, tmp_path: Path):
        """Should use custom key function for complex objects."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = [
            WorkItem(id="item1", value=10),
            WorkItem(id="item2", value=20),
            WorkItem(id="item3", value=30),
        ]

        # Complete first item
        for item in runner.iterate(items, key_fn=lambda x: x.id):
            runner.save_result(item.id, item.value * 2)
            break

        # Resume - should skip item1
        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        result = list(runner2.iterate(items, key_fn=lambda x: x.id))

        assert len(result) == 2
        assert result[0].id == "item2"
        assert result[1].id == "item3"

    def test_default_key_uses_str(self, tmp_path: Path):
        """Should use str() as default key function."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = [1, 2, 3]

        for item in runner.iterate(items):
            runner.save_result(str(item), item)
            if item == 1:
                break

        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        result = list(runner2.iterate(items))

        assert result == [2, 3]


class TestProgressTracking:
    """Tests for progress reporting."""

    def test_progress_during_iteration(self, tmp_path: Path):
        """Should track progress during iteration."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = ["a", "b", "c", "d"]

        progress_snapshots = []
        for item in runner.iterate(items):
            runner.save_result(item, True)
            progress_snapshots.append(runner.progress())

        # Progress shows (completed, total)
        assert progress_snapshots == [
            (1, 4),  # After completing 'a'
            (2, 4),  # After completing 'b'
            (3, 4),  # After completing 'c'
            (4, 4),  # After completing 'd'
        ]

    def test_progress_with_resume(self, tmp_path: Path):
        """Should show correct progress after resume."""
        # First run
        runner1 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        for item in runner1.iterate(["a", "b", "c"]):
            runner1.save_result(item, True)
            if item == "a":
                break

        # Resume
        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        _ = list(runner2.iterate(["a", "b", "c"]))  # Consume iterator

        completed, total = runner2.progress()
        assert completed == 1  # Only 'a' was saved
        assert total == 3

    def test_progress_before_iteration(self, tmp_path: Path):
        """Progress should be (0, 0) before iteration starts."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert runner.progress() == (0, 0)


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_results(self, tmp_path: Path):
        """Reset should clear all saved results."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        runner.save_result("a", 1)
        runner.save_result("b", 2)

        runner.reset()

        assert runner.get_all_results() == {}
        assert not runner.is_completed("a")

    def test_reset_deletes_checkpoint_file(self, tmp_path: Path):
        """Reset should delete the checkpoint file."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        runner.save_result("a", 1)

        assert runner.checkpoint_path.exists()

        runner.reset()

        assert not runner.checkpoint_path.exists()

    def test_reset_allows_full_reiteration(self, tmp_path: Path):
        """After reset, all items should be yielded again."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        items = ["a", "b", "c"]

        # Complete all
        for item in runner.iterate(items):
            runner.save_result(item, True)

        # Verify all completed
        assert list(runner.iterate(items)) == []

        # Reset and try again
        runner.reset()
        result = list(runner.iterate(items))

        assert result == ["a", "b", "c"]


class TestAtomicWrite:
    """Tests for atomic write behavior."""

    def test_no_temp_file_left_after_save(self, tmp_path: Path):
        """Temp file should be renamed, not left behind."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        runner.save_result("key", "value")

        temp_path = tmp_path / "test_checkpoint.tmp"
        assert not temp_path.exists()
        assert runner.checkpoint_path.exists()

    def test_checkpoint_file_is_valid_json(self, tmp_path: Path):
        """Checkpoint file should contain valid JSON after save."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)
        runner.save_result("key", {"nested": [1, 2, 3]})

        # Should be parseable JSON
        with open(runner.checkpoint_path) as f:
            data = json.load(f)

        assert "results" in data
        assert "completed" in data
        assert data["results"]["key"] == {"nested": [1, 2, 3]}


class TestCheckpointPath:
    """Tests for checkpoint path configuration."""

    def test_default_checkpoint_dir(self):
        """Should use ./checkpoints by default."""
        runner = CheckpointedRunner("mytest")

        assert runner.checkpoint_dir == Path("./checkpoints")
        assert runner.checkpoint_path == Path("./checkpoints/mytest_checkpoint.json")

    def test_custom_checkpoint_dir_string(self, tmp_path: Path):
        """Should accept string path."""
        runner = CheckpointedRunner("test", checkpoint_dir=str(tmp_path))

        assert runner.checkpoint_dir == tmp_path

    def test_custom_checkpoint_dir_path(self, tmp_path: Path):
        """Should accept Path object."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert runner.checkpoint_dir == tmp_path

    def test_creates_checkpoint_dir_on_save(self, tmp_path: Path):
        """Should create checkpoint directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "dirs"
        runner = CheckpointedRunner("test", checkpoint_dir=nested_dir)

        runner.save_result("key", "value")

        assert nested_dir.exists()
        assert runner.checkpoint_path.exists()


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_special_characters_in_name(self, tmp_path: Path):
        """Should handle special characters in checkpoint name."""
        runner = CheckpointedRunner("test-with_special.chars", checkpoint_dir=tmp_path)
        runner.save_result("key", "value")

        assert runner.checkpoint_path.exists()

        runner2 = CheckpointedRunner("test-with_special.chars", checkpoint_dir=tmp_path)
        assert runner2.get_result("key") == "value"

    def test_non_string_result_values(self, tmp_path: Path):
        """Should handle various JSON-serializable types."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        runner.save_result("int", 42)
        runner.save_result("float", 3.14)
        runner.save_result("bool", True)
        runner.save_result("none", None)
        runner.save_result("list", [1, 2, 3])
        runner.save_result("dict", {"a": 1})

        runner2 = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        assert runner2.get_result("int") == 42
        assert runner2.get_result("float") == 3.14
        assert runner2.get_result("bool") is True
        assert runner2.get_result("none") is None
        assert runner2.get_result("list") == [1, 2, 3]
        assert runner2.get_result("dict") == {"a": 1}

    def test_overwrite_existing_result(self, tmp_path: Path):
        """Should allow overwriting a result with same key."""
        runner = CheckpointedRunner("test", checkpoint_dir=tmp_path)

        runner.save_result("key", "first")
        runner.save_result("key", "second")

        assert runner.get_result("key") == "second"

    def test_multiple_runners_same_name(self, tmp_path: Path):
        """Multiple runners with same name share checkpoint."""
        runner1 = CheckpointedRunner("shared", checkpoint_dir=tmp_path)
        runner1.save_result("from_runner1", "value1")

        runner2 = CheckpointedRunner("shared", checkpoint_dir=tmp_path)
        runner2.save_result("from_runner2", "value2")

        # Both should see both results
        assert runner1.get_result("from_runner2") is None  # runner1 didn't reload

        runner3 = CheckpointedRunner("shared", checkpoint_dir=tmp_path)
        assert runner3.get_result("from_runner1") == "value1"
        assert runner3.get_result("from_runner2") == "value2"

    def test_different_runners_isolated(self, tmp_path: Path):
        """Runners with different names should be isolated."""
        runner1 = CheckpointedRunner("runner1", checkpoint_dir=tmp_path)
        runner1.save_result("key", "from_runner1")

        runner2 = CheckpointedRunner("runner2", checkpoint_dir=tmp_path)
        runner2.save_result("key", "from_runner2")

        assert runner1.get_result("key") == "from_runner1"
        assert runner2.get_result("key") == "from_runner2"
