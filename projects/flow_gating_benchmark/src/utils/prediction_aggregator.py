"""
Aggregate n bootstrap predictions into single prompts for multi-judge evaluation.

Instead of judging each of 10 bootstrap runs separately, this aggregates all runs
for the same (model, test_case, condition) into a single prompt showing the
distribution of responses across runs.

Usage:
    from utils.prediction_aggregator import aggregate_predictions, build_aggregated_judge_prompt

    aggregated = aggregate_predictions(predictions_list)
    # Returns list of AggregatedPrediction objects, one per (model, test_case, condition)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.response_parser import parse_llm_response


@dataclass
class AggregatedPrediction:
    """Aggregated predictions across bootstrap runs."""

    test_case_id: str
    model: str
    condition: str
    n_bootstraps: int

    # All raw responses from bootstrap runs
    raw_responses: list[str] = field(default_factory=list)
    bootstrap_runs: list[int] = field(default_factory=list)

    # Parsed hierarchies (successful parses only)
    parsed_hierarchies: list[dict] = field(default_factory=list)
    parse_success_rate: float = 0.0

    # Consistency metrics
    unique_gate_sets: int = 0  # How many distinct gate sets across runs
    consistency_score: float = 0.0  # 1.0 = all runs identical, 0.0 = all different

    # Metadata
    timestamps: list[str] = field(default_factory=list)
    run_id: str = ""

    @property
    def key(self) -> tuple:
        """Unique key for this aggregation."""
        return (self.test_case_id, self.model, self.condition)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_case_id": self.test_case_id,
            "model": self.model,
            "condition": self.condition,
            "n_bootstraps": self.n_bootstraps,
            "raw_responses": self.raw_responses,
            "bootstrap_runs": self.bootstrap_runs,
            "parsed_hierarchies": self.parsed_hierarchies,
            "parse_success_rate": self.parse_success_rate,
            "unique_gate_sets": self.unique_gate_sets,
            "consistency_score": self.consistency_score,
            "timestamps": self.timestamps,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregatedPrediction:
        """Create from dictionary."""
        return cls(
            test_case_id=data["test_case_id"],
            model=data["model"],
            condition=data["condition"],
            n_bootstraps=data["n_bootstraps"],
            raw_responses=data.get("raw_responses", []),
            bootstrap_runs=data.get("bootstrap_runs", []),
            parsed_hierarchies=data.get("parsed_hierarchies", []),
            parse_success_rate=data.get("parse_success_rate", 0.0),
            unique_gate_sets=data.get("unique_gate_sets", 0),
            consistency_score=data.get("consistency_score", 0.0),
            timestamps=data.get("timestamps", []),
            run_id=data.get("run_id", ""),
        )


def extract_gate_names(hierarchy: dict) -> frozenset[str]:
    """Extract all gate names from a hierarchy as a frozen set."""
    names = set()

    def traverse(node: dict):
        if "name" in node:
            names.add(node["name"].lower().strip())
        for child in node.get("children", []):
            traverse(child)

    if "root" in hierarchy:
        traverse(hierarchy["root"])
    else:
        traverse(hierarchy)

    return frozenset(names)


def compute_consistency(parsed_hierarchies: list[dict]) -> tuple[int, float]:
    """Compute consistency metrics across parsed hierarchies.

    Returns:
        (unique_gate_sets, consistency_score)
        - unique_gate_sets: number of distinct gate name sets
        - consistency_score: 1.0 if all identical, 0.0 if all different
    """
    if not parsed_hierarchies:
        return 0, 0.0

    gate_sets = [extract_gate_names(h) for h in parsed_hierarchies]
    unique_sets = set(gate_sets)
    n_unique = len(unique_sets)

    # Consistency: 1 - (unique - 1) / (total - 1)
    # If all same: 1.0, if all different: 0.0
    n_total = len(gate_sets)
    if n_total <= 1:
        return n_unique, 1.0

    consistency = 1.0 - (n_unique - 1) / (n_total - 1)
    return n_unique, max(0.0, consistency)


def aggregate_predictions(predictions: list[dict]) -> list[AggregatedPrediction]:
    """Aggregate predictions by (model, test_case_id, condition).

    Args:
        predictions: List of prediction dicts with keys:
            - test_case_id, model, condition, bootstrap_run
            - raw_response, timestamp, run_id

    Returns:
        List of AggregatedPrediction objects, one per unique combination.
    """
    # Group by key
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for pred in predictions:
        key = (pred["test_case_id"], pred["model"], pred["condition"])
        groups[key].append(pred)

    # Build aggregated predictions
    aggregated = []

    for (test_case_id, model, condition), preds in groups.items():
        # Sort by bootstrap_run
        preds = sorted(preds, key=lambda p: p.get("bootstrap_run", 0))

        agg = AggregatedPrediction(
            test_case_id=test_case_id,
            model=model,
            condition=condition,
            n_bootstraps=len(preds),
            raw_responses=[p.get("raw_response", "") for p in preds],
            bootstrap_runs=[p.get("bootstrap_run", i) for i, p in enumerate(preds)],
            timestamps=[p.get("timestamp", "") for p in preds],
            run_id=preds[0].get("run_id", "") if preds else "",
        )

        # Parse hierarchies
        for response in agg.raw_responses:
            if response:
                result = parse_llm_response(response)
                if result.success and result.hierarchy:
                    agg.parsed_hierarchies.append(result.hierarchy)

        agg.parse_success_rate = len(agg.parsed_hierarchies) / len(agg.raw_responses) if agg.raw_responses else 0.0

        # Compute consistency
        agg.unique_gate_sets, agg.consistency_score = compute_consistency(agg.parsed_hierarchies)

        aggregated.append(agg)

    return aggregated


def flatten_hierarchy(hierarchy: dict, path: str = "") -> str:
    """Convert hierarchy dict to flat arrow notation."""
    if "root" in hierarchy:
        hierarchy = hierarchy["root"]

    name = hierarchy.get("name", "Unknown")
    current = f"{path} > {name}" if path else name

    children = hierarchy.get("children", [])
    if not children:
        return current

    paths = []
    for child in children:
        paths.append(flatten_hierarchy(child, current))

    return "\n".join(paths)


def summarize_response(response: str, max_len: int = 300) -> str:
    """Summarize a prediction response for display."""
    if not response or not response.strip():
        return "[EMPTY]"

    # Try to parse
    result = parse_llm_response(response)
    if result.success and result.hierarchy:
        flat = flatten_hierarchy(result.hierarchy)
        lines = flat.split("\n")
        summary = "\n".join(lines[:5])
        if len(lines) > 5:
            summary += f"\n... (+{len(lines) - 5} more)"
        return summary
    else:
        # Show truncated raw
        raw = response.strip()[:max_len]
        if len(response.strip()) > max_len:
            raw += "..."
        return f"[PARSE FAILED] {raw}"


def build_aggregated_judge_prompt(
    aggregated: AggregatedPrediction,
    ground_truth: dict,
    metrics_by_run: list[dict] | None = None,
) -> str:
    """Build a judge prompt that shows all bootstrap runs.

    Args:
        aggregated: AggregatedPrediction with all bootstrap responses
        ground_truth: Ground truth test case dict
        metrics_by_run: Optional list of metrics dicts, one per bootstrap run

    Returns:
        Prompt string for LLM judge
    """
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})

    # Build response summaries
    response_sections = []
    for i, (boot_run, response) in enumerate(zip(aggregated.bootstrap_runs, aggregated.raw_responses)):
        summary = summarize_response(response)
        metrics_str = ""
        if metrics_by_run and i < len(metrics_by_run):
            m = metrics_by_run[i]
            metrics_str = f" [F1={m.get('hierarchy_f1', 0):.2f}]"
        response_sections.append(f"--- Run {boot_run}{metrics_str} ---\n{summary}")

    responses_block = "\n\n".join(response_sections)

    prompt = f"""Evaluate this model's gating hierarchy predictions across {aggregated.n_bootstraps} independent runs.

TEST CASE: {aggregated.test_case_id}
MODEL: {aggregated.model}
CONDITION: {aggregated.condition}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

PREDICTIONS ACROSS {aggregated.n_bootstraps} RUNS:
{responses_block}

CONSISTENCY METRICS:
- Parse success rate: {aggregated.parse_success_rate:.0%}
- Unique gate sets: {aggregated.unique_gate_sets} / {aggregated.n_bootstraps}
- Consistency score: {aggregated.consistency_score:.2f}

Evaluate the model's OVERALL performance across all runs:

Reply in this EXACT format:
MEDIAN_QUALITY: [0-10] (typical prediction quality across runs)
CONSISTENCY: [0-10] (how consistent are predictions across runs?)
WORST_CASE: [0-10] (quality of worst prediction)
BEST_CASE: [0-10] (quality of best prediction)
FAILURE_MODES: [comma-separated list of recurring errors, or "none"]
RELIABILITY: [high/medium/low] [one sentence on whether this model is reliable for this task]
SUMMARY: [one sentence overall assessment]
"""
    return prompt


# Default config for aggregated judge
AGGREGATED_JUDGE_CONFIG = {
    "model": "gemini-2.5-pro",
    "max_tokens": 20000,  # Higher limit for aggregated responses
    "temperature": 0.0,
}


def load_and_aggregate(predictions_path: str | Path) -> list[AggregatedPrediction]:
    """Load predictions from JSON and aggregate by (model, test_case, condition).

    Args:
        predictions_path: Path to predictions.json file

    Returns:
        List of AggregatedPrediction objects
    """
    with open(predictions_path) as f:
        data = json.load(f)

    # Handle both list format and dict with 'results' key
    if isinstance(data, list):
        predictions = data
    else:
        predictions = data.get("results", data.get("predictions", []))

    return aggregate_predictions(predictions)


def save_aggregated(
    aggregated: list[AggregatedPrediction],
    output_path: str | Path,
) -> None:
    """Save aggregated predictions to JSON.

    Args:
        aggregated: List of AggregatedPrediction objects
        output_path: Path to write JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "aggregated_predictions": [a.to_dict() for a in aggregated],
        "n_aggregations": len(aggregated),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate bootstrap predictions")
    parser.add_argument("predictions", help="Path to predictions.json")
    parser.add_argument("-o", "--output", help="Output path for aggregated JSON")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")

    args = parser.parse_args()

    aggregated = load_and_aggregate(args.predictions)

    if args.stats:
        print(f"Aggregated {len(aggregated)} unique (model, test_case, condition) combinations\n")

        # Stats by model
        from collections import Counter
        by_model = Counter(a.model for a in aggregated)
        print("By model:")
        for model, count in sorted(by_model.items()):
            avg_consistency = sum(a.consistency_score for a in aggregated if a.model == model) / count
            print(f"  {model}: {count} aggregations, avg consistency: {avg_consistency:.2f}")
    else:
        output = args.output or args.predictions.replace(".json", "_aggregated.json")
        save_aggregated(aggregated, output)
        print(f"Saved {len(aggregated)} aggregated predictions to {output}")
