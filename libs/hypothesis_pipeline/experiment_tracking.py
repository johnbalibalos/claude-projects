"""
Experiment tracking integration for LLM evaluation.

Provides integration with experiment tracking tools:
- MLflow integration
- Weights & Biases integration
- Custom lightweight tracking
- Result versioning and comparison
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Protocol


# =============================================================================
# PROTOCOLS AND BASE CLASSES
# =============================================================================


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        """Start a new experiment run. Returns run ID."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters/configuration."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        ...

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        """Log an artifact file."""
        ...

    @abstractmethod
    def log_text(self, text: str, artifact_name: str) -> None:
        """Log text as an artifact."""
        ...

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        ...

    @abstractmethod
    def get_run_url(self) -> str | None:
        """Get URL to view run in tracking UI."""
        ...


# =============================================================================
# MLFLOW INTEGRATION
# =============================================================================


class MLflowTracker(ExperimentTracker):
    """
    MLflow experiment tracking integration.

    Requires: pip install mlflow
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local)
            artifact_location: Where to store artifacts
        """
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            raise ImportError("mlflow not installed. Run: pip install mlflow")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self._run = None

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        """Start a new MLflow run."""
        self._run = self.mlflow.start_run(run_name=run_name, tags=tags)
        return self._run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        # MLflow requires string values, so convert
        flat_params = self._flatten_dict(params)
        for key, value in flat_params.items():
            try:
                self.mlflow.log_param(key, str(value)[:500])  # MLflow has length limits
            except Exception:
                pass  # Skip params that can't be logged

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (value != value):  # Check for NaN
                self.mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        """Log artifact to MLflow."""
        self.mlflow.log_artifact(local_path)

    def log_text(self, text: str, artifact_name: str) -> None:
        """Log text as artifact."""
        self.mlflow.log_text(text, artifact_name)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the MLflow run."""
        if self._run:
            self.mlflow.end_run(status=status)
            self._run = None

    def get_run_url(self) -> str | None:
        """Get MLflow run URL."""
        if self._run:
            tracking_uri = self.mlflow.get_tracking_uri()
            return f"{tracking_uri}/#/experiments/{self._run.info.experiment_id}/runs/{self._run.info.run_id}"
        return None

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


# =============================================================================
# WEIGHTS & BIASES INTEGRATION
# =============================================================================


class WandBTracker(ExperimentTracker):
    """
    Weights & Biases experiment tracking integration.

    Requires: pip install wandb
    """

    def __init__(
        self,
        project_name: str,
        entity: str | None = None,
    ):
        """
        Initialize W&B tracker.

        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Run: pip install wandb")

        self.project_name = project_name
        self.entity = entity
        self._run = None

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        """Start a new W&B run."""
        tag_list = list(tags.values()) if tags else None
        self._run = self.wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            tags=tag_list,
        )
        return self._run.id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to W&B."""
        if self._run:
            self.wandb.config.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to W&B."""
        if self._run:
            self.wandb.log(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        """Log artifact to W&B."""
        if self._run:
            artifact = self.wandb.Artifact(
                name=artifact_name or Path(local_path).name,
                type="output",
            )
            artifact.add_file(local_path)
            self._run.log_artifact(artifact)

    def log_text(self, text: str, artifact_name: str) -> None:
        """Log text as artifact."""
        if self._run:
            # Save to temp file and log
            temp_path = Path(f"/tmp/{artifact_name}")
            temp_path.write_text(text)
            self.log_artifact(str(temp_path), artifact_name)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the W&B run."""
        if self._run:
            self.wandb.finish()
            self._run = None

    def get_run_url(self) -> str | None:
        """Get W&B run URL."""
        if self._run:
            return self._run.get_url()
        return None


# =============================================================================
# LIGHTWEIGHT LOCAL TRACKER
# =============================================================================


@dataclass
class RunMetadata:
    """Metadata for a tracked run."""

    run_id: str
    run_name: str
    experiment_name: str
    start_time: str
    end_time: str | None = None
    status: str = "RUNNING"
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    git_commit: str | None = None
    git_branch: str | None = None


class LocalTracker(ExperimentTracker):
    """
    Lightweight local experiment tracker.

    Stores experiment data as JSON files without external dependencies.
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: str | Path = "./experiments",
    ):
        """
        Initialize local tracker.

        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for storing experiments
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._run: RunMetadata | None = None
        self._run_dir: Path | None = None

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        """Start a new run."""
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_name}_{timestamp}"

        # Create run directory
        self._run_dir = self.experiment_dir / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Get git info
        git_commit = self._get_git_commit()
        git_branch = self._get_git_branch()

        # Create run metadata
        self._run = RunMetadata(
            run_id=run_id,
            run_name=run_name,
            experiment_name=self.experiment_name,
            start_time=datetime.now().isoformat(),
            tags=tags or {},
            git_commit=git_commit,
            git_branch=git_branch,
        )

        self._save_metadata()
        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""
        if self._run:
            self._run.params.update(params)
            self._save_metadata()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        if self._run:
            if step is not None:
                # Store with step info
                for key, value in metrics.items():
                    metric_key = f"{key}_step_{step}"
                    self._run.metrics[metric_key] = value
            else:
                self._run.metrics.update(metrics)
            self._save_metadata()

            # Also append to metrics history file
            metrics_file = self._run_dir / "metrics_history.jsonl"
            with open(metrics_file, "a") as f:
                entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
                f.write(json.dumps(entry) + "\n")

    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        """Log artifact by copying to run directory."""
        if self._run and self._run_dir:
            import shutil
            src = Path(local_path)
            dst = self._run_dir / "artifacts" / (artifact_name or src.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            self._run.artifacts.append(str(dst.relative_to(self._run_dir)))
            self._save_metadata()

    def log_text(self, text: str, artifact_name: str) -> None:
        """Log text as artifact."""
        if self._run and self._run_dir:
            artifact_path = self._run_dir / "artifacts" / artifact_name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(text)
            self._run.artifacts.append(f"artifacts/{artifact_name}")
            self._save_metadata()

    def end_run(self, status: str = "FINISHED") -> None:
        """End the run."""
        if self._run:
            self._run.end_time = datetime.now().isoformat()
            self._run.status = status
            self._save_metadata()
            self._run = None
            self._run_dir = None

    def get_run_url(self) -> str | None:
        """Get path to run directory."""
        if self._run_dir:
            return str(self._run_dir.absolute())
        return None

    def _save_metadata(self) -> None:
        """Save run metadata to disk."""
        if self._run and self._run_dir:
            metadata_path = self._run_dir / "run_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(asdict(self._run), f, indent=2, default=str)

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _get_git_branch(self) -> str | None:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    # Additional methods for querying runs

    def list_runs(self) -> list[RunMetadata]:
        """List all runs in the experiment."""
        runs = []
        for run_dir in self.experiment_dir.iterdir():
            if run_dir.is_dir():
                metadata_path = run_dir / "run_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        data = json.load(f)
                        runs.append(RunMetadata(**data))
        return sorted(runs, key=lambda r: r.start_time, reverse=True)

    def get_run(self, run_id: str) -> RunMetadata | None:
        """Get metadata for a specific run."""
        run_dir = self.experiment_dir / run_id
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
                return RunMetadata(**data)
        return None

    def compare_runs(self, run_ids: list[str], metric: str) -> dict[str, float]:
        """Compare a metric across multiple runs."""
        results = {}
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run and metric in run.metrics:
                results[run_id] = run.metrics[metric]
        return results


# =============================================================================
# TRACKER FACTORY
# =============================================================================


def create_tracker(
    tracker_type: Literal["mlflow", "wandb", "local"] = "local",
    experiment_name: str = "default",
    **kwargs: Any,
) -> ExperimentTracker:
    """
    Factory function to create experiment trackers.

    Args:
        tracker_type: Type of tracker to create
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for the tracker

    Returns:
        ExperimentTracker instance
    """
    if tracker_type == "mlflow":
        return MLflowTracker(experiment_name, **kwargs)
    elif tracker_type == "wandb":
        return WandBTracker(experiment_name, **kwargs)
    else:
        return LocalTracker(experiment_name, **kwargs)


# =============================================================================
# EXPERIMENT COMPARISON
# =============================================================================


@dataclass
class RunComparison:
    """Comparison of multiple experiment runs."""

    run_ids: list[str]
    metrics_comparison: dict[str, dict[str, float]]  # metric -> run_id -> value
    params_diff: dict[str, dict[str, Any]]  # param -> run_id -> value
    best_run_by_metric: dict[str, str]  # metric -> best run_id


def compare_experiments(
    tracker: LocalTracker,
    run_ids: list[str],
    metrics_to_compare: list[str] | None = None,
) -> RunComparison:
    """
    Compare multiple experiment runs.

    Args:
        tracker: Local tracker with run data
        run_ids: List of run IDs to compare
        metrics_to_compare: Specific metrics to compare (default: all)

    Returns:
        RunComparison with comparison data
    """
    runs = [tracker.get_run(rid) for rid in run_ids]
    runs = [r for r in runs if r is not None]

    if not runs:
        raise ValueError("No valid runs found")

    # Collect all metrics
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.metrics.keys())

    if metrics_to_compare:
        all_metrics = all_metrics & set(metrics_to_compare)

    # Build metrics comparison
    metrics_comparison = {}
    for metric in all_metrics:
        metrics_comparison[metric] = {}
        for run in runs:
            if metric in run.metrics:
                metrics_comparison[metric][run.run_id] = run.metrics[metric]

    # Find best run for each metric
    best_run_by_metric = {}
    for metric, values in metrics_comparison.items():
        if values:
            # Assume higher is better (could be parameterized)
            best_run = max(values, key=values.get)
            best_run_by_metric[metric] = best_run

    # Build params diff
    params_diff = {}
    all_params = set()
    for run in runs:
        all_params.update(run.params.keys())

    for param in all_params:
        values = {}
        for run in runs:
            if param in run.params:
                values[run.run_id] = run.params[param]

        # Only include if values differ
        unique_values = set(str(v) for v in values.values())
        if len(unique_values) > 1:
            params_diff[param] = values

    return RunComparison(
        run_ids=run_ids,
        metrics_comparison=metrics_comparison,
        params_diff=params_diff,
        best_run_by_metric=best_run_by_metric,
    )


def generate_comparison_report(comparison: RunComparison) -> str:
    """Generate markdown report comparing runs."""
    lines = [
        "# Experiment Comparison Report",
        "",
        f"Comparing {len(comparison.run_ids)} runs",
        "",
        "## Metrics Comparison",
        "",
    ]

    # Build metrics table
    metrics = list(comparison.metrics_comparison.keys())
    if metrics:
        header = "| Metric | " + " | ".join(comparison.run_ids) + " | Best |"
        separator = "|--------|" + "|------|" * len(comparison.run_ids) + "|------|"
        lines.extend([header, separator])

        for metric in sorted(metrics):
            values = comparison.metrics_comparison[metric]
            row_values = []
            for run_id in comparison.run_ids:
                val = values.get(run_id, "N/A")
                if isinstance(val, float):
                    row_values.append(f"{val:.4f}")
                else:
                    row_values.append(str(val))

            best = comparison.best_run_by_metric.get(metric, "N/A")
            lines.append(f"| {metric} | " + " | ".join(row_values) + f" | {best[:8]} |")

    lines.append("")
    lines.append("## Parameter Differences")
    lines.append("")

    if comparison.params_diff:
        for param, values in comparison.params_diff.items():
            lines.append(f"**{param}:**")
            for run_id, val in values.items():
                lines.append(f"  - {run_id}: {val}")
            lines.append("")
    else:
        lines.append("No parameter differences found.")

    return "\n".join(lines)


# =============================================================================
# CONTEXT MANAGER FOR TRACKING
# =============================================================================


class TrackedExperiment:
    """Context manager for experiment tracking."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        """
        Initialize tracked experiment.

        Args:
            tracker: Experiment tracker to use
            run_name: Name for the run
            params: Parameters to log
            tags: Tags to add to run
        """
        self.tracker = tracker
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self._run_id: str | None = None

    def __enter__(self) -> "TrackedExperiment":
        """Start the experiment run."""
        self._run_id = self.tracker.start_run(self.run_name, self.tags)
        if self.params:
            self.tracker.log_params(self.params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the experiment run."""
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.tracker.end_run(status)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics during the experiment."""
        self.tracker.log_metrics(metrics, step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log an artifact."""
        self.tracker.log_artifact(path, name)

    def log_text(self, text: str, name: str) -> None:
        """Log text as artifact."""
        self.tracker.log_text(text, name)

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        return self._run_id

    @property
    def run_url(self) -> str | None:
        """Get URL to view run."""
        return self.tracker.get_run_url()


# =============================================================================
# INTEGRATION WITH HYPOTHESIS PIPELINE
# =============================================================================


def track_pipeline_run(
    tracker: ExperimentTracker,
    config: Any,  # PipelineConfig
    results: Any,  # ExperimentResults
) -> str:
    """
    Track a hypothesis pipeline run.

    Args:
        tracker: Experiment tracker
        config: Pipeline configuration
        results: Experiment results

    Returns:
        Run ID
    """
    run_id = tracker.start_run(
        run_name=config.name,
        tags={
            "hypothesis": config.hypothesis,
            "data_source": str(config.data_source),
        },
    )

    # Log config as params
    tracker.log_params({
        "models": config.models,
        "reasoning_types": [r.value for r in config.reasoning_types],
        "context_levels": [c.value for c in config.context_levels],
        "rag_modes": [r.value for r in config.rag_modes],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    })

    # Log aggregate metrics
    for condition_name, metrics in results.metrics_by_condition.items():
        prefixed_metrics = {f"{condition_name}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        tracker.log_metrics(prefixed_metrics)

    # Log summary metrics
    all_success_rates = [
        m.get("success_rate", 0) for m in results.metrics_by_condition.values()
    ]
    if all_success_rates:
        tracker.log_metrics({
            "avg_success_rate": sum(all_success_rates) / len(all_success_rates),
            "n_conditions": len(results.conditions),
            "n_trials": len(results.trials),
        })

    # Log results as artifact
    tracker.log_text(
        json.dumps(results.to_dict(), indent=2, default=str),
        "results.json",
    )

    tracker.end_run()
    return run_id
