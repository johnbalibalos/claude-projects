"""
Visualization utilities for experiment results.

Creates summary plots and comparison charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from ..evaluation.scorer import ScoringResult, compute_metrics_by_model, compute_metrics_by_condition


def check_plotting_available() -> bool:
    """Check if plotting libraries are available."""
    return HAS_PLOTTING


def create_summary_plots(
    results: list[ScoringResult],
    output_dir: str | Path,
    prefix: str = "benchmark",
) -> list[Path]:
    """
    Create summary visualization plots.

    Args:
        results: List of ScoringResults
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames

    Returns:
        List of paths to created plots
    """
    if not HAS_PLOTTING:
        print("Warning: matplotlib/seaborn not available. Skipping plots.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_plots = []

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # Plot 1: Metrics by model
    plot_path = output_dir / f"{prefix}_metrics_by_model.png"
    _plot_metrics_by_model(results, plot_path)
    created_plots.append(plot_path)

    # Plot 2: Metrics by context level
    plot_path = output_dir / f"{prefix}_metrics_by_context.png"
    _plot_metrics_by_context(results, plot_path)
    created_plots.append(plot_path)

    # Plot 3: F1 distribution
    plot_path = output_dir / f"{prefix}_f1_distribution.png"
    _plot_f1_distribution(results, plot_path)
    created_plots.append(plot_path)

    # Plot 4: Metric correlations
    plot_path = output_dir / f"{prefix}_metric_correlations.png"
    _plot_metric_correlations(results, plot_path)
    created_plots.append(plot_path)

    return created_plots


def _plot_metrics_by_model(results: list[ScoringResult], output_path: Path) -> None:
    """Plot key metrics grouped by model."""
    by_model = compute_metrics_by_model(results)

    models = list(by_model.keys())
    metrics = ["hierarchy_f1_mean", "structure_accuracy_mean", "critical_gate_recall_mean"]
    metric_labels = ["Hierarchy F1", "Structure Accuracy", "Critical Gate Recall"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(models))
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [by_model[m].get(metric, 0) for m in models]
        ax.bar([xi + i * width for xi in x], values, width, label=label)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Metrics by Model")
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_metrics_by_context(results: list[ScoringResult], output_path: Path) -> None:
    """Plot metrics grouped by context level."""
    # Group by context level extracted from condition name
    context_groups: dict[str, list[ScoringResult]] = {
        "minimal": [],
        "standard": [],
        "rich": [],
    }

    for result in results:
        for context in context_groups:
            if context in result.condition.lower():
                context_groups[context].append(result)
                break

    # Compute metrics for each context level
    context_metrics = {}
    for context, group_results in context_groups.items():
        if group_results:
            valid = [r for r in group_results if r.parse_success and r.evaluation]
            if valid:
                context_metrics[context] = {
                    "f1": sum(r.hierarchy_f1 for r in valid) / len(valid),
                    "structure": sum(r.structure_accuracy for r in valid) / len(valid),
                    "critical": sum(r.critical_gate_recall for r in valid) / len(valid),
                }

    if not context_metrics:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    contexts = list(context_metrics.keys())
    x = range(len(contexts))
    width = 0.25

    for i, (metric, label) in enumerate([
        ("f1", "Hierarchy F1"),
        ("structure", "Structure Accuracy"),
        ("critical", "Critical Gate Recall"),
    ]):
        values = [context_metrics[c].get(metric, 0) for c in contexts]
        ax.bar([xi + i * width for xi in x], values, width, label=label)

    ax.set_xlabel("Context Level")
    ax.set_ylabel("Score")
    ax.set_title("Metrics by Context Level")
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(contexts)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_f1_distribution(results: list[ScoringResult], output_path: Path) -> None:
    """Plot distribution of F1 scores."""
    valid = [r for r in results if r.parse_success and r.evaluation]

    if not valid:
        return

    f1_scores = [r.hierarchy_f1 for r in valid]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(f1_scores, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(sum(f1_scores) / len(f1_scores), color="red", linestyle="--",
               label=f"Mean: {sum(f1_scores) / len(f1_scores):.2f}")

    ax.set_xlabel("Hierarchy F1 Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of F1 Scores")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_metric_correlations(results: list[ScoringResult], output_path: Path) -> None:
    """Plot correlations between metrics."""
    valid = [r for r in results if r.parse_success and r.evaluation]

    if len(valid) < 5:
        return

    # Prepare data
    data = {
        "F1": [r.hierarchy_f1 for r in valid],
        "Structure": [r.structure_accuracy for r in valid],
        "Critical": [r.critical_gate_recall for r in valid],
        "Hallucination": [r.evaluation.hallucination_rate for r in valid if r.evaluation],
    }

    try:
        import pandas as pd
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            df.corr(),
            annot=True,
            cmap="coolwarm",
            center=0,
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        ax.set_title("Metric Correlations")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except ImportError:
        pass


def create_comparison_table(
    results: list[ScoringResult],
    output_path: str | Path,
) -> None:
    """
    Create a markdown comparison table.

    Args:
        results: List of ScoringResults
        output_path: Where to save the table
    """
    by_model = compute_metrics_by_model(results)
    by_condition = compute_metrics_by_condition(results)

    lines = [
        "# Benchmark Results Comparison",
        "",
        "## Results by Model",
        "",
        "| Model | F1 | Structure | Critical Gate | Parse Rate |",
        "|-------|-----|-----------|---------------|------------|",
    ]

    for model, metrics in by_model.items():
        model_short = model.split("/")[-1]
        lines.append(
            f"| {model_short} | "
            f"{metrics.get('hierarchy_f1_mean', 0):.3f} | "
            f"{metrics.get('structure_accuracy_mean', 0):.3f} | "
            f"{metrics.get('critical_gate_recall_mean', 0):.3f} | "
            f"{metrics.get('parse_success_rate', 0):.1%} |"
        )

    lines.extend([
        "",
        "## Results by Condition",
        "",
        "| Condition | F1 | Structure | Parse Rate |",
        "|-----------|-----|-----------|------------|",
    ])

    for condition, metrics in by_condition.items():
        lines.append(
            f"| {condition} | "
            f"{metrics.get('hierarchy_f1_mean', 0):.3f} | "
            f"{metrics.get('structure_accuracy_mean', 0):.3f} | "
            f"{metrics.get('parse_success_rate', 0):.1%} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
