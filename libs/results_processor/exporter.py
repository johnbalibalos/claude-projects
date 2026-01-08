"""
Results exporter for experiment data.

Provides JSON to CSV conversion and summary generation.
"""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


# Default column configurations for different experiment types
GATING_BENCHMARK_COLUMNS = [
    "test_case_id",
    "model",
    "condition",
    "context_level",
    "prompt_strategy",
    "parse_success",
    "hierarchy_f1",
    "hierarchy_precision",
    "hierarchy_recall",
    "structure_accuracy",
    "critical_gate_recall",
    "hallucination_rate",
    "depth_accuracy",
    "n_predicted_gates",
    "n_ground_truth_gates",
    "n_matching_gates",
    "n_missing_gates",
    "n_extra_gates",
    "n_hallucinated_gates",
    "predicted_gates",
    "ground_truth_gates",
    "missing_gates",
    "extra_gates",
]

PANEL_OPTIMIZER_COLUMNS = [
    "test_case_id",
    "condition",
    "case_type",
    "accuracy",
    "complexity_index",
    "ci_improvement",
    "latency",
    "tool_calls",
]


def parse_condition_parts(condition: str) -> tuple[str, str]:
    """
    Parse condition string into context_level and prompt_strategy.

    Args:
        condition: Condition string like "sonnet_standard_cot"

    Returns:
        Tuple of (context_level, prompt_strategy)
    """
    parts = condition.split("_")
    if len(parts) >= 3:
        return parts[-2], parts[-1]  # context_level, prompt_strategy
    return "", ""


class ResultsExporter:
    """
    Export experiment results to various formats.

    Supports:
    - JSON to CSV conversion with configurable columns
    - Summary report generation
    - Aggregation by condition
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        list_separator: str = "|",
    ):
        """
        Initialize exporter.

        Args:
            columns: Column names to export (auto-detected if None)
            list_separator: Separator for list values in CSV
        """
        self.columns = columns
        self.list_separator = list_separator

    def export_to_csv(
        self,
        json_path: Path | str,
        output_path: Path | str | None = None,
        row_transformer: Callable[[dict], dict] | None = None,
    ) -> Path:
        """
        Export experiment results JSON to CSV.

        Args:
            json_path: Path to experiment results JSON
            output_path: Output CSV path (default: same name with .csv)
            row_transformer: Optional function to transform each row

        Returns:
            Path to output CSV file
        """
        json_path = Path(json_path)
        output_path = Path(output_path) if output_path else json_path.with_suffix(".csv")

        with open(json_path) as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            raise ValueError(f"No results found in {json_path}")

        # Auto-detect columns if not specified
        columns = self.columns
        if columns is None:
            columns = self._detect_columns(results)

        rows = []
        for result in results:
            row = self._extract_row(result, columns)
            if row_transformer:
                row = row_transformer(row)
            rows.append(row)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def _detect_columns(self, results: list[dict]) -> list[str]:
        """Auto-detect columns from result structure."""
        if not results:
            return []

        sample = results[0]

        # Check for gating benchmark structure
        if "evaluation" in sample and "hierarchy_f1" in sample.get("evaluation", {}):
            return GATING_BENCHMARK_COLUMNS

        # Check for panel optimizer structure
        if "accuracy" in sample and "complexity_index" in sample:
            return PANEL_OPTIMIZER_COLUMNS

        # Fallback: use all top-level keys
        return list(sample.keys())

    def _extract_row(self, result: dict, columns: list[str]) -> dict:
        """Extract row data from a result dict."""
        row = {}
        evaluation = result.get("evaluation", {})
        condition = result.get("condition", "")

        # Parse condition parts
        context_level, prompt_strategy = parse_condition_parts(condition)

        for col in columns:
            if col == "context_level":
                row[col] = context_level
            elif col == "prompt_strategy":
                row[col] = prompt_strategy
            elif col.startswith("n_") and col.endswith("_gates"):
                # Count gate lists
                gate_key = col[2:]  # Remove "n_" prefix
                gates = evaluation.get(gate_key, [])
                row[col] = len(gates) if isinstance(gates, list) else 0
            elif col in ("predicted_gates", "ground_truth_gates", "missing_gates", "extra_gates"):
                # Join gate lists
                gates = evaluation.get(col, [])
                row[col] = self.list_separator.join(gates) if isinstance(gates, list) else ""
            elif col in evaluation:
                row[col] = evaluation[col]
            elif col in result:
                row[col] = result[col]
            else:
                row[col] = None

        return row

    def generate_summary(
        self,
        json_path: Path | str,
        output_path: Path | str | None = None,
        title: str | None = None,
    ) -> Path:
        """
        Generate text summary of experiment results.

        Args:
            json_path: Path to experiment results JSON
            output_path: Output text path (default: same name with _summary.txt)
            title: Title for the summary

        Returns:
            Path to output text file
        """
        json_path = Path(json_path)
        if output_path is None:
            output_path = json_path.with_name(
                json_path.stem + "_summary.txt"
            )
        output_path = Path(output_path)

        with open(json_path) as f:
            data = json.load(f)

        results = data.get("results", [])
        metadata = data.get("metadata", {})

        lines = []

        # Header
        if title:
            lines.append(title)
        else:
            lines.append(f"Experiment Summary: {json_path.stem}")
        lines.append("=" * 60)
        lines.append("")

        # Metadata
        lines.append(f"Date: {metadata.get('date', datetime.now().isoformat())}")
        lines.append(f"Total results: {len(results)}")
        lines.append("")

        # Aggregate by condition
        conditions = {}
        for result in results:
            cond = result.get("condition", "unknown")
            if cond not in conditions:
                conditions[cond] = []
            conditions[cond].append(result)

        lines.append("Results by Condition")
        lines.append("-" * 60)

        for cond, cond_results in sorted(conditions.items()):
            lines.append(f"\n{cond} (n={len(cond_results)})")

            # Calculate metrics based on result structure
            if cond_results and "evaluation" in cond_results[0]:
                # Gating benchmark
                successful = [r for r in cond_results if r.get("parse_success")]
                if successful:
                    avg_f1 = sum(
                        r["evaluation"].get("hierarchy_f1", 0) for r in successful
                    ) / len(successful)
                    avg_structure = sum(
                        r["evaluation"].get("structure_accuracy", 0) for r in successful
                    ) / len(successful)
                    parse_rate = len(successful) / len(cond_results)

                    lines.append(f"  Hierarchy F1: {avg_f1:.3f}")
                    lines.append(f"  Structure Accuracy: {avg_structure:.3f}")
                    lines.append(f"  Parse Success: {parse_rate:.1%}")

            elif cond_results and "accuracy" in cond_results[0]:
                # Panel optimizer
                avg_accuracy = sum(r.get("accuracy", 0) for r in cond_results) / len(cond_results)
                avg_ci = sum(r.get("complexity_index", 0) for r in cond_results) / len(cond_results)

                lines.append(f"  Accuracy: {avg_accuracy:.3f}")
                lines.append(f"  Complexity Index: {avg_ci:.3f}")

        lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path


def export_to_csv(
    json_path: Path | str,
    output_path: Path | str | None = None,
    columns: list[str] | None = None,
) -> Path:
    """
    Convenience function to export JSON to CSV.

    Args:
        json_path: Path to experiment results JSON
        output_path: Output CSV path
        columns: Column names to export

    Returns:
        Path to output CSV file
    """
    exporter = ResultsExporter(columns=columns)
    return exporter.export_to_csv(json_path, output_path)


def generate_summary(
    json_path: Path | str,
    output_path: Path | str | None = None,
    title: str | None = None,
) -> Path:
    """
    Convenience function to generate summary.

    Args:
        json_path: Path to experiment results JSON
        output_path: Output text path
        title: Title for the summary

    Returns:
        Path to output text file
    """
    exporter = ResultsExporter()
    return exporter.generate_summary(json_path, output_path, title)
