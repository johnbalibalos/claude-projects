"""
Experiment tracking for hypothesis pipelines.

Tracks:
- Experiment metadata (config, git commit, timestamp)
- Results and metrics
- Conclusions and notes
- Enables comparison across experiments
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .models import ExperimentResults, TrialInput


@dataclass
class ExperimentMetadata:
    """Metadata about an experiment run."""

    # Identity
    experiment_id: str
    name: str
    timestamp: str

    # Config
    config: dict[str, Any]
    config_hash: str

    # Git info (if available)
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False

    # Environment
    python_version: str = ""
    hostname: str = ""
    user: str = ""

    # Hypothesis and description
    hypothesis: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, config: dict[str, Any]) -> "ExperimentMetadata":
        """Create metadata from config dict."""
        import platform
        import sys

        # Generate experiment ID from config hash + timestamp
        timestamp = datetime.now().isoformat()
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        experiment_id = f"{config.get('name', 'exp')}_{timestamp[:10]}_{config_hash}"

        return cls(
            experiment_id=experiment_id,
            name=config.get("name", "unnamed"),
            timestamp=timestamp,
            config=config,
            config_hash=config_hash,
            git_commit=_get_git_commit(),
            git_branch=_get_git_branch(),
            git_dirty=_is_git_dirty(),
            python_version=sys.version.split()[0],
            hostname=platform.node(),
            user=os.environ.get("USER", "unknown"),
            hypothesis=config.get("hypothesis", ""),
            description=config.get("description", ""),
            tags=config.get("tags", []),
        )


@dataclass
class ExperimentConclusion:
    """Conclusions and notes about an experiment."""

    # Summary
    summary: str = ""
    outcome: str = ""  # success, partial, failed, inconclusive

    # Key findings
    findings: list[str] = field(default_factory=list)

    # Statistical significance (optional)
    significant_differences: list[dict[str, Any]] = field(default_factory=list)

    # Next steps
    next_steps: list[str] = field(default_factory=list)

    # Notes
    notes: str = ""

    # Timestamp
    concluded_at: str = ""


@dataclass
class ExperimentRecord:
    """Complete record of an experiment."""

    metadata: ExperimentMetadata
    results_summary: dict[str, Any]  # Aggregated metrics
    conclusion: ExperimentConclusion | None = None

    # File paths
    results_file: str = ""
    config_file: str = ""
    inputs_file: str = ""  # Trial inputs

    # Input summary
    n_inputs: int = 0
    input_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": asdict(self.metadata),
            "results_summary": self.results_summary,
            "conclusion": asdict(self.conclusion) if self.conclusion else None,
            "results_file": self.results_file,
            "config_file": self.config_file,
            "inputs_file": self.inputs_file,
            "n_inputs": self.n_inputs,
            "input_ids": self.input_ids,
        }


class ExperimentTracker:
    """
    Tracks and manages experiment runs.

    Features:
    - Saves experiment metadata and configs
    - Records results with full reproducibility info
    - Allows adding conclusions after reviewing results
    - Maintains an index for easy querying
    - Supports experiment comparison
    """

    def __init__(self, experiments_dir: Path | str = "./experiments"):
        """
        Args:
            experiments_dir: Directory to store experiment records
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.experiments_dir / "index.yaml"
        self._index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict[str, dict]:
        """Load experiment index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_index(self) -> None:
        """Save experiment index."""
        with open(self.index_file, "w") as f:
            yaml.dump(self._index, f, default_flow_style=False, sort_keys=False)

    def start_experiment(self, config: dict[str, Any]) -> ExperimentMetadata:
        """
        Start tracking a new experiment.

        Args:
            config: Pipeline configuration dict

        Returns:
            ExperimentMetadata for the new experiment
        """
        metadata = ExperimentMetadata.create(config)

        # Create experiment directory
        exp_dir = self.experiments_dir / metadata.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_file = exp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save metadata
        meta_file = exp_dir / "metadata.yaml"
        with open(meta_file, "w") as f:
            yaml.dump(asdict(metadata), f, default_flow_style=False)

        # Update index
        self._index[metadata.experiment_id] = {
            "name": metadata.name,
            "timestamp": metadata.timestamp,
            "status": "running",
            "tags": metadata.tags,
            "hypothesis": metadata.hypothesis,
        }
        self._save_index()

        return metadata

    def save_inputs(
        self,
        metadata: ExperimentMetadata,
        trial_inputs: list[TrialInput],
        include_raw: bool = True,
    ) -> str:
        """
        Save trial inputs for an experiment.

        Args:
            metadata: Experiment metadata
            trial_inputs: List of trial inputs
            include_raw: Whether to include raw_input data (can be large)

        Returns:
            Path to saved inputs file
        """
        exp_dir = self.experiments_dir / metadata.experiment_id

        # Convert inputs to serializable format
        inputs_data = {
            "n_inputs": len(trial_inputs),
            "input_ids": [t.id for t in trial_inputs],
            "inputs": [t.to_dict(include_raw=include_raw) for t in trial_inputs],
        }

        # Save inputs
        inputs_file = exp_dir / "inputs.json"
        with open(inputs_file, "w") as f:
            json.dump(inputs_data, f, indent=2, default=str)

        # Update index
        self._index[metadata.experiment_id]["n_inputs"] = len(trial_inputs)
        self._save_index()

        return str(inputs_file)

    def save_results(
        self,
        metadata: ExperimentMetadata,
        results: ExperimentResults,
        trial_inputs: list[TrialInput] | None = None,
        include_raw_inputs: bool = True,
    ) -> ExperimentRecord:
        """
        Save experiment results.

        Args:
            metadata: Experiment metadata
            results: Full experiment results
            trial_inputs: Optional trial inputs to save alongside results
            include_raw_inputs: Whether to include raw_input data in saved inputs

        Returns:
            ExperimentRecord with paths to saved files
        """
        exp_dir = self.experiments_dir / metadata.experiment_id

        # Save inputs if provided
        inputs_file = ""
        n_inputs = 0
        input_ids = []
        if trial_inputs:
            inputs_file = self.save_inputs(metadata, trial_inputs, include_raw_inputs)
            n_inputs = len(trial_inputs)
            input_ids = [t.id for t in trial_inputs]

        # Save full results
        results_file = exp_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        # Save summary
        summary = self._compute_summary(results)
        summary_file = exp_dir / "summary.yaml"
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        # Create record
        record = ExperimentRecord(
            metadata=metadata,
            results_summary=summary,
            results_file=str(results_file),
            config_file=str(exp_dir / "config.yaml"),
            inputs_file=inputs_file,
            n_inputs=n_inputs,
            input_ids=input_ids,
        )

        # Save record
        record_file = exp_dir / "record.yaml"
        with open(record_file, "w") as f:
            yaml.dump(record.to_dict(), f, default_flow_style=False)

        # Update index
        self._index[metadata.experiment_id]["status"] = "completed"
        self._index[metadata.experiment_id]["summary"] = {
            "n_trials": summary.get("n_trials", 0),
            "success_rate": summary.get("overall_success_rate", 0),
            "best_condition": summary.get("best_condition", ""),
        }
        self._save_index()

        return record

    def add_conclusion(
        self,
        experiment_id: str,
        summary: str,
        outcome: str = "inconclusive",
        findings: list[str] | None = None,
        next_steps: list[str] | None = None,
        notes: str = "",
    ) -> ExperimentConclusion:
        """
        Add conclusions to an experiment.

        Args:
            experiment_id: Experiment ID
            summary: Brief summary of conclusions
            outcome: success, partial, failed, inconclusive
            findings: List of key findings
            next_steps: List of suggested next steps
            notes: Additional notes

        Returns:
            ExperimentConclusion
        """
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")

        conclusion = ExperimentConclusion(
            summary=summary,
            outcome=outcome,
            findings=findings or [],
            next_steps=next_steps or [],
            notes=notes,
            concluded_at=datetime.now().isoformat(),
        )

        # Save conclusion
        conclusion_file = exp_dir / "conclusion.yaml"
        with open(conclusion_file, "w") as f:
            yaml.dump(asdict(conclusion), f, default_flow_style=False)

        # Update record
        record_file = exp_dir / "record.yaml"
        if record_file.exists():
            with open(record_file) as f:
                record_data = yaml.safe_load(f)
            record_data["conclusion"] = asdict(conclusion)
            with open(record_file, "w") as f:
                yaml.dump(record_data, f, default_flow_style=False)

        # Update index
        self._index[experiment_id]["outcome"] = outcome
        self._index[experiment_id]["concluded"] = True
        self._save_index()

        return conclusion

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Get an experiment record by ID."""
        exp_dir = self.experiments_dir / experiment_id
        record_file = exp_dir / "record.yaml"

        if not record_file.exists():
            return None

        with open(record_file) as f:
            data = yaml.safe_load(f)

        return ExperimentRecord(
            metadata=ExperimentMetadata(**data["metadata"]),
            results_summary=data["results_summary"],
            conclusion=ExperimentConclusion(**data["conclusion"]) if data.get("conclusion") else None,
            results_file=data.get("results_file", ""),
            config_file=data.get("config_file", ""),
            inputs_file=data.get("inputs_file", ""),
            n_inputs=data.get("n_inputs", 0),
            input_ids=data.get("input_ids", []),
        )

    def get_inputs(self, experiment_id: str) -> list[dict[str, Any]] | None:
        """
        Load trial inputs for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of input dictionaries, or None if not found
        """
        exp_dir = self.experiments_dir / experiment_id
        inputs_file = exp_dir / "inputs.json"

        if not inputs_file.exists():
            return None

        with open(inputs_file) as f:
            data = json.load(f)

        return data.get("inputs", [])

    def list_experiments(
        self,
        tags: list[str] | None = None,
        status: str | None = None,
        name_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List experiments with optional filtering.

        Args:
            tags: Filter by tags (any match)
            status: Filter by status (running, completed)
            name_contains: Filter by name substring

        Returns:
            List of experiment index entries
        """
        results = []

        for exp_id, info in self._index.items():
            # Apply filters
            if tags and not any(t in info.get("tags", []) for t in tags):
                continue
            if status and info.get("status") != status:
                continue
            if name_contains and name_contains.lower() not in info.get("name", "").lower():
                continue

            results.append({"experiment_id": exp_id, **info})

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return results

    def compare_experiments(
        self,
        experiment_ids: list[str],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            Comparison data structure
        """
        comparison = {
            "experiments": [],
            "metrics_comparison": {},
            "best_by_metric": {},
        }

        all_metrics = set()

        for exp_id in experiment_ids:
            record = self.get_experiment(exp_id)
            if not record:
                continue

            exp_data = {
                "experiment_id": exp_id,
                "name": record.metadata.name,
                "timestamp": record.metadata.timestamp,
                "hypothesis": record.metadata.hypothesis,
                "metrics": {},
            }

            # Extract metrics
            summary = record.results_summary
            for key, value in summary.items():
                if isinstance(value, (int, float)) and (not metrics or key in metrics):
                    exp_data["metrics"][key] = value
                    all_metrics.add(key)

            # Also get per-condition metrics
            if "by_condition" in summary:
                exp_data["by_condition"] = summary["by_condition"]

            comparison["experiments"].append(exp_data)

        # Compare across experiments
        for metric in all_metrics:
            values = []
            for exp in comparison["experiments"]:
                if metric in exp["metrics"]:
                    values.append({
                        "experiment_id": exp["experiment_id"],
                        "value": exp["metrics"][metric],
                    })

            if values:
                comparison["metrics_comparison"][metric] = values
                # Find best
                best = max(values, key=lambda x: x["value"])
                comparison["best_by_metric"][metric] = best

        return comparison

    def _compute_summary(self, results: ExperimentResults) -> dict[str, Any]:
        """Compute summary statistics from results."""
        summary = {
            "n_trials": len(results.trials),
            "n_conditions": len(results.conditions),
            "started_at": results.started_at.isoformat() if results.started_at else None,
            "completed_at": results.completed_at.isoformat() if results.completed_at else None,
        }

        # Overall success rate
        successful = [t for t in results.trials if t.success]
        summary["n_successful"] = len(successful)
        summary["overall_success_rate"] = len(successful) / len(results.trials) if results.trials else 0

        # Aggregate metrics
        all_metrics = set()
        for trial in successful:
            all_metrics.update(trial.scores.keys())

        for metric in all_metrics:
            scores = [t.scores[metric] for t in successful if metric in t.scores]
            if scores:
                summary[f"{metric}_mean"] = sum(scores) / len(scores)
                summary[f"{metric}_min"] = min(scores)
                summary[f"{metric}_max"] = max(scores)
                summary[f"{metric}_std"] = _std(scores)

        # Per-condition summary
        summary["by_condition"] = {}
        best_score = -1
        best_condition = ""

        for cond in results.conditions:
            cond_trials = [t for t in results.trials if t.condition_name == cond.name]
            cond_successful = [t for t in cond_trials if t.success]

            cond_summary = {
                "n_trials": len(cond_trials),
                "success_rate": len(cond_successful) / len(cond_trials) if cond_trials else 0,
            }

            # Metrics
            for metric in all_metrics:
                scores = [t.scores[metric] for t in cond_successful if metric in t.scores]
                if scores:
                    mean = sum(scores) / len(scores)
                    cond_summary[f"{metric}_mean"] = mean

                    # Track best condition (by first metric)
                    if metric == list(all_metrics)[0] and mean > best_score:
                        best_score = mean
                        best_condition = cond.name

            summary["by_condition"][cond.name] = cond_summary

        summary["best_condition"] = best_condition

        return summary

    def generate_report(self, experiment_id: str) -> str:
        """Generate a markdown report for an experiment."""
        record = self.get_experiment(experiment_id)
        if not record:
            return f"Experiment not found: {experiment_id}"

        lines = [
            f"# Experiment Report: {record.metadata.name}",
            "",
            f"**ID:** {record.metadata.experiment_id}",
            f"**Timestamp:** {record.metadata.timestamp}",
            f"**Status:** {self._index.get(experiment_id, {}).get('status', 'unknown')}",
            "",
        ]

        if record.metadata.hypothesis:
            lines.extend([
                "## Hypothesis",
                "",
                record.metadata.hypothesis,
                "",
            ])

        if record.metadata.description:
            lines.extend([
                "## Description",
                "",
                record.metadata.description,
                "",
            ])

        # Git info
        if record.metadata.git_commit:
            lines.extend([
                "## Reproducibility",
                "",
                f"- **Git Commit:** {record.metadata.git_commit}",
                f"- **Git Branch:** {record.metadata.git_branch}",
                f"- **Dirty:** {record.metadata.git_dirty}",
                f"- **Config Hash:** {record.metadata.config_hash}",
                "",
            ])

        # Results summary
        lines.extend([
            "## Results Summary",
            "",
            f"- **Total Trials:** {record.results_summary.get('n_trials', 0)}",
            f"- **Successful:** {record.results_summary.get('n_successful', 0)}",
            f"- **Success Rate:** {record.results_summary.get('overall_success_rate', 0):.1%}",
            f"- **Best Condition:** {record.results_summary.get('best_condition', 'N/A')}",
            "",
        ])

        # Metrics
        lines.append("### Metrics")
        lines.append("")
        for key, value in record.results_summary.items():
            if key.endswith("_mean") and not key.startswith("by_"):
                metric_name = key[:-5]
                mean = value
                std = record.results_summary.get(f"{metric_name}_std", 0)
                lines.append(f"- **{metric_name}:** {mean:.3f} ± {std:.3f}")
        lines.append("")

        # Per-condition
        if "by_condition" in record.results_summary:
            lines.append("### By Condition")
            lines.append("")
            lines.append("| Condition | Success Rate | Key Metrics |")
            lines.append("|-----------|--------------|-------------|")

            for cond_name, cond_data in record.results_summary["by_condition"].items():
                success_rate = cond_data.get("success_rate", 0)
                metrics_str = ", ".join(
                    f"{k[:-5]}={v:.2f}"
                    for k, v in cond_data.items()
                    if k.endswith("_mean")
                )[:50]
                lines.append(f"| {cond_name} | {success_rate:.1%} | {metrics_str} |")

            lines.append("")

        # Conclusion
        if record.conclusion:
            lines.extend([
                "## Conclusion",
                "",
                f"**Outcome:** {record.conclusion.outcome}",
                "",
                record.conclusion.summary,
                "",
            ])

            if record.conclusion.findings:
                lines.append("### Key Findings")
                lines.append("")
                for finding in record.conclusion.findings:
                    lines.append(f"- {finding}")
                lines.append("")

            if record.conclusion.next_steps:
                lines.append("### Next Steps")
                lines.append("")
                for step in record.conclusion.next_steps:
                    lines.append(f"- {step}")
                lines.append("")

            if record.conclusion.notes:
                lines.extend([
                    "### Notes",
                    "",
                    record.conclusion.notes,
                    "",
                ])

        return "\n".join(lines)


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def _get_git_branch() -> str | None:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _is_git_dirty() -> bool:
    """Check if git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        pass
    return False
