"""
Merge and validate prediction JSONL files.

Combines predictions from multiple runs with validation:
- Checks git commit compatibility
- Deduplicates by prediction key
- Outputs to SQLite or consolidated JSONL
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""
    experiment_id: str
    git_commit: str
    git_dirty: bool
    started_at: str
    models: list[str]
    test_cases_dir: str
    n_bootstrap: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "started_at": self.started_at,
            "models": self.models,
            "test_cases_dir": self.test_cases_dir,
            "n_bootstrap": self.n_bootstrap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentMetadata:
        return cls(
            experiment_id=data["experiment_id"],
            git_commit=data["git_commit"],
            git_dirty=data.get("git_dirty", False),
            started_at=data["started_at"],
            models=data["models"],
            test_cases_dir=data["test_cases_dir"],
            n_bootstrap=data["n_bootstrap"],
        )

    @classmethod
    def create(
        cls,
        models: list[str],
        test_cases_dir: str,
        n_bootstrap: int,
    ) -> ExperimentMetadata:
        """Create metadata with current git info."""
        # Get git commit
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except subprocess.CalledProcessError:
            commit = "unknown"

        # Check if dirty
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            dirty = bool(status)
        except subprocess.CalledProcessError:
            dirty = False

        return cls(
            experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            git_commit=commit,
            git_dirty=dirty,
            started_at=datetime.now().isoformat(),
            models=models,
            test_cases_dir=test_cases_dir,
            n_bootstrap=n_bootstrap,
        )


def merge_jsonl_files(
    input_files: list[Path],
    output_path: Path,
    validate_commit: bool = True,
) -> dict[str, Any]:
    """
    Merge multiple JSONL prediction files.

    Args:
        input_files: List of JSONL files to merge
        output_path: Output path (.jsonl, .sqlite, or .json)
        validate_commit: If True, warn if commits differ

    Returns:
        Summary dict with counts and any warnings
    """
    predictions = []
    seen_keys = set()
    commits_seen = set()
    warnings = []

    for filepath in input_files:
        if not filepath.exists():
            warnings.append(f"File not found: {filepath}")
            continue

        file_count = 0
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    # Track git commit if present
                    if "git_commit" in data:
                        commits_seen.add(data["git_commit"])

                    # Deduplicate by key
                    key = (
                        data.get("bootstrap_run"),
                        data.get("test_case_id"),
                        data.get("model"),
                        data.get("condition"),
                    )
                    if key not in seen_keys:
                        predictions.append(data)
                        seen_keys.add(key)
                        file_count += 1

                except (json.JSONDecodeError, KeyError) as e:
                    warnings.append(f"Skipped malformed line in {filepath}: {e}")

        print(f"  {filepath.name}: {file_count} predictions")

    # Validate commits
    if validate_commit and len(commits_seen) > 1:
        warnings.append(f"Multiple git commits found: {commits_seen}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".sqlite":
        _write_sqlite(predictions, output_path)
    elif output_path.suffix == ".jsonl":
        _write_jsonl(predictions, output_path)
    else:
        _write_json(predictions, output_path)

    return {
        "total_predictions": len(predictions),
        "files_merged": len(input_files),
        "commits": list(commits_seen),
        "warnings": warnings,
    }


def _write_jsonl(predictions: list[dict], output_path: Path) -> None:
    """Write predictions to JSONL file."""
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def _write_json(predictions: list[dict], output_path: Path) -> None:
    """Write predictions to JSON file."""
    with open(output_path, "w") as f:
        json.dump({"results": predictions}, f, indent=2)


def _write_sqlite(predictions: list[dict], output_path: Path) -> None:
    """Write predictions to SQLite database."""
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_case_id TEXT,
            model TEXT,
            condition TEXT,
            bootstrap_run INTEGER,
            raw_response TEXT,
            tokens_used INTEGER,
            timestamp TEXT,
            error TEXT,
            git_commit TEXT,
            experiment_id TEXT,
            UNIQUE(test_case_id, model, condition, bootstrap_run)
        )
    """)

    # Insert predictions
    for pred in predictions:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO predictions
                (test_case_id, model, condition, bootstrap_run, raw_response,
                 tokens_used, timestamp, error, git_commit, experiment_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred.get("test_case_id"),
                pred.get("model"),
                pred.get("condition"),
                pred.get("bootstrap_run"),
                pred.get("raw_response"),
                pred.get("tokens_used"),
                pred.get("timestamp"),
                pred.get("error"),
                pred.get("git_commit"),
                pred.get("experiment_id"),
            ))
        except sqlite3.Error:
            continue

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON predictions(model)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_case ON predictions(test_case_id)")

    conn.commit()
    conn.close()


def main():
    """CLI for merging prediction files."""
    import argparse

    parser = argparse.ArgumentParser(description="Merge prediction JSONL files")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    parser.add_argument("--no-validate", action="store_true", help="Skip commit validation")

    args = parser.parse_args()

    print(f"Merging {len(args.inputs)} files...")
    result = merge_jsonl_files(args.inputs, args.output, validate_commit=not args.no_validate)

    print(f"\nMerged {result['total_predictions']} predictions")
    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
