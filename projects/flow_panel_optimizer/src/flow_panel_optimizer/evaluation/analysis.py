"""
Analyze ablation study results.

Key questions:
1. Does MCP outperform best retrieval condition?
2. How does performance vary by test case type (in-dist vs out-of-dist)?
3. What's the retrieval weight that maximizes performance?
4. When does MCP help vs hurt?
"""

import json
from typing import Optional
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .runner import ExperimentResults


def compute_mcp_lift(results: ExperimentResults) -> dict:
    """
    Calculate performance lift from MCP vs best retrieval-only condition.

    Returns dict with:
    - best_retrieval_condition: name of best retrieval-only condition
    - best_retrieval_ci: CI of that condition (lower is better)
    - mcp_only_ci: CI of mcp_only condition
    - mcp_lift: improvement in CI (positive = MCP is better)
    - is_significant: whether lift exceeds noise threshold
    """
    if not HAS_PANDAS:
        return _compute_mcp_lift_simple(results)

    df = results.to_dataframe()

    # Get retrieval-only conditions (no MCP)
    retrieval_conditions = df[
        ~df["condition"].str.contains("mcp")
    ].groupby("condition")["complexity_index"].mean()

    if len(retrieval_conditions) == 0:
        return {"error": "No retrieval-only conditions found"}

    # Best = lowest CI
    best_retrieval = retrieval_conditions.idxmin()
    best_retrieval_ci = retrieval_conditions.min()

    # Get MCP-only CI
    mcp_data = df[df["condition"] == "mcp_only"]
    if len(mcp_data) == 0:
        return {"error": "No mcp_only condition found"}

    mcp_only_ci = mcp_data["complexity_index"].mean()

    # Lift = improvement (lower CI is better, so positive lift = MCP better)
    lift = (best_retrieval_ci - mcp_only_ci) / best_retrieval_ci if best_retrieval_ci > 0 else 0

    return {
        "best_retrieval_condition": best_retrieval,
        "best_retrieval_ci": round(best_retrieval_ci, 4),
        "mcp_only_ci": round(mcp_only_ci, 4),
        "mcp_lift": round(lift, 4),
        "mcp_lift_pct": f"{lift*100:+.1f}%",
        "is_significant": abs(lift) > 0.10  # 10% threshold
    }


def _compute_mcp_lift_simple(results: ExperimentResults) -> dict:
    """Simple implementation without pandas."""
    retrieval_cis = {}
    mcp_cis = []

    for trial in results.trials:
        if "mcp" not in trial.condition_name:
            if trial.condition_name not in retrieval_cis:
                retrieval_cis[trial.condition_name] = []
            retrieval_cis[trial.condition_name].append(trial.complexity_index)
        elif trial.condition_name == "mcp_only":
            mcp_cis.append(trial.complexity_index)

    if not retrieval_cis:
        return {"error": "No retrieval-only conditions found"}
    if not mcp_cis:
        return {"error": "No mcp_only condition found"}

    # Calculate means
    retrieval_means = {k: sum(v)/len(v) for k, v in retrieval_cis.items()}
    best_retrieval = min(retrieval_means, key=retrieval_means.get)
    best_retrieval_ci = retrieval_means[best_retrieval]
    mcp_only_ci = sum(mcp_cis) / len(mcp_cis)

    lift = (best_retrieval_ci - mcp_only_ci) / best_retrieval_ci if best_retrieval_ci > 0 else 0

    return {
        "best_retrieval_condition": best_retrieval,
        "best_retrieval_ci": round(best_retrieval_ci, 4),
        "mcp_only_ci": round(mcp_only_ci, 4),
        "mcp_lift": round(lift, 4),
        "mcp_lift_pct": f"{lift*100:+.1f}%",
        "is_significant": abs(lift) > 0.10
    }


def analyze_by_case_type(results: ExperimentResults) -> dict:
    """
    Break down performance by test case type.

    Critical insight: MCP should help MORE on out-of-distribution cases
    where retrieval has no precedent to find.
    """
    if not HAS_PANDAS:
        return _analyze_by_case_type_simple(results)

    df = results.to_dataframe()

    pivot = df.pivot_table(
        values="complexity_index",
        index="condition",
        columns="case_type",
        aggfunc="mean"
    ).round(2)

    return pivot.to_dict()


def _analyze_by_case_type_simple(results: ExperimentResults) -> dict:
    """Simple implementation without pandas."""
    data = {}

    for trial in results.trials:
        key = (trial.condition_name, trial.test_case_type)
        if key not in data:
            data[key] = []
        data[key].append(trial.complexity_index)

    # Aggregate
    result = {}
    for (cond, case_type), values in data.items():
        if cond not in result:
            result[cond] = {}
        result[cond][case_type] = round(sum(values) / len(values), 2)

    return result


def find_optimal_retrieval_weight(results: ExperimentResults) -> dict:
    """
    Find the retrieval weight that maximizes performance.

    Returns the weight and whether increasing weight helps.
    """
    weight_map = {
        "baseline": 0,
        "retrieval_standard": 1,
        "retrieval_2x": 2,
        "retrieval_5x": 5,
        "retrieval_10x": 10,
        "retrieval_exclusive": 100
    }

    weight_cis = {}
    for trial in results.trials:
        if trial.condition_name in weight_map:
            weight = weight_map[trial.condition_name]
            if weight not in weight_cis:
                weight_cis[weight] = []
            weight_cis[weight].append(trial.complexity_index)

    if not weight_cis:
        return {"error": "No retrieval conditions found"}

    # Calculate means
    weight_means = {k: sum(v)/len(v) for k, v in weight_cis.items()}

    # Optimal = lowest CI
    optimal_weight = min(weight_means, key=weight_means.get)

    # Check if increasing weight helps (negative correlation = helps)
    weights = sorted(weight_means.keys())
    if len(weights) > 1:
        cis = [weight_means[w] for w in weights]
        # Simple correlation check
        increasing_helps = cis[0] > cis[-1]
    else:
        increasing_helps = None

    return {
        "optimal_weight": optimal_weight,
        "optimal_ci": round(weight_means[optimal_weight], 4),
        "weight_performance": {k: round(v, 4) for k, v in weight_means.items()},
        "increasing_weight_helps": increasing_helps
    }


def generate_report(
    results: ExperimentResults,
    output_path: Optional[Path] = None
) -> str:
    """Generate human-readable analysis report."""

    mcp_lift = compute_mcp_lift(results)
    case_type_analysis = analyze_by_case_type(results)
    weight_analysis = find_optimal_retrieval_weight(results)

    # Calculate summary stats
    by_condition = {}
    for trial in results.trials:
        if trial.condition_name not in by_condition:
            by_condition[trial.condition_name] = {
                "ci": [], "accuracy": [], "latency": [], "tool_calls": []
            }
        by_condition[trial.condition_name]["ci"].append(trial.complexity_index)
        by_condition[trial.condition_name]["accuracy"].append(trial.assignment_accuracy)
        by_condition[trial.condition_name]["latency"].append(trial.latency_seconds)
        by_condition[trial.condition_name]["tool_calls"].append(len(trial.tool_calls_made))

    report = f"""
# MCP Ablation Study Results
## Experiment: {results.experiment_name}

---

## Executive Summary

**Key Finding:** {"MCP provides significant improvement over retrieval" if mcp_lift.get("is_significant") and mcp_lift.get("mcp_lift", 0) > 0 else "Results inconclusive or MCP does not significantly outperform retrieval"}

- Best retrieval-only condition: {mcp_lift.get("best_retrieval_condition", "N/A")} (CI={mcp_lift.get("best_retrieval_ci", "N/A")})
- MCP-only complexity index: {mcp_lift.get("mcp_only_ci", "N/A")}
- MCP lift: {mcp_lift.get("mcp_lift_pct", "N/A")} (lower CI is better)

---

## Detailed Results by Condition

| Condition | Avg CI | Avg Accuracy | Avg Latency | Tool Calls |
|-----------|--------|--------------|-------------|------------|
"""

    for cond, stats in sorted(by_condition.items()):
        avg_ci = sum(stats["ci"]) / len(stats["ci"]) if stats["ci"] else 0
        avg_acc = sum(stats["accuracy"]) / len(stats["accuracy"]) if stats["accuracy"] else 0
        avg_lat = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else 0
        avg_tools = sum(stats["tool_calls"]) / len(stats["tool_calls"]) if stats["tool_calls"] else 0

        report += f"| {cond} | {avg_ci:.2f} | {avg_acc:.1%} | {avg_lat:.1f}s | {avg_tools:.0f} |\n"

    report += f"""
---

## Retrieval Weight Analysis

Optimal retrieval weight: {weight_analysis.get("optimal_weight", "N/A")}x
{"Increasing weight improves performance" if weight_analysis.get("increasing_weight_helps") else "Diminishing returns from increased weight" if weight_analysis.get("increasing_weight_helps") is False else "Insufficient data"}

Weight performance curve:
"""
    if "weight_performance" in weight_analysis:
        for weight, ci in sorted(weight_analysis["weight_performance"].items()):
            report += f"  - {weight}x: CI={ci}\n"

    report += f"""
---

## Performance by Test Case Type

"""
    if isinstance(case_type_analysis, dict):
        for cond, types in case_type_analysis.items():
            report += f"**{cond}:**\n"
            if isinstance(types, dict):
                for case_type, ci in types.items():
                    report += f"  - {case_type}: CI={ci}\n"
            report += "\n"

    report += f"""
---

## Interpretation

### In-Distribution Cases
Both retrieval and MCP should perform well on in-distribution cases since the answers exist in the OMIP corpus.

### Out-of-Distribution Cases
MCP should outperform retrieval on OOD cases since there's no precedent to retrieve.
This is the critical test of MCP value.

### Adversarial Cases
Tests robustness when retrieval precedent conflicts with spectral physics.
MCP should recognize spectral conflicts even when OMIPs use problematic pairs.

---

## Recommendations

"""
    if mcp_lift.get("mcp_lift", 0) > 0.15:
        report += "1. **MCP provides substantial value** - consider deploying as primary tool\n"
    elif mcp_lift.get("mcp_lift", 0) > 0.05:
        report += "1. **MCP provides moderate value** - useful for complex panels\n"
    else:
        report += "1. **MCP provides limited value** - retrieval may be sufficient\n"

    if weight_analysis.get("optimal_weight", 0) > 1:
        report += f"2. **Optimal retrieval weight is {weight_analysis['optimal_weight']}x** - heavier weighting helps\n"
    else:
        report += "2. **Standard retrieval weight is optimal** - no need for upweighting\n"

    report += f"""
---

*Generated: {results.trials[0].timestamp if results.trials else "N/A"}*
*Total trials: {len(results.trials)}*
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def quick_summary(results: ExperimentResults) -> str:
    """Generate a quick one-line summary."""
    mcp_lift = compute_mcp_lift(results)

    if "error" in mcp_lift:
        return f"Error: {mcp_lift['error']}"

    return (
        f"MCP lift: {mcp_lift['mcp_lift_pct']} | "
        f"Retrieval CI: {mcp_lift['best_retrieval_ci']:.2f} | "
        f"MCP CI: {mcp_lift['mcp_only_ci']:.2f}"
    )


if __name__ == "__main__":
    # Test with sample data
    from .runner import TrialResult, ExperimentResults

    sample_trials = [
        TrialResult(
            condition_name="baseline",
            test_case_id="test1",
            test_case_type="in_distribution",
            raw_response="test",
            extracted_assignments={"CD3": "FITC"},
            tool_calls_made=[],
            assignment_accuracy=0.5,
            complexity_index=5.0,
            ground_truth_ci=4.0,
            ci_improvement=-0.25,
            latency_seconds=2.0,
            input_tokens=100,
            output_tokens=50
        ),
        TrialResult(
            condition_name="mcp_only",
            test_case_id="test1",
            test_case_type="in_distribution",
            raw_response="test",
            extracted_assignments={"CD3": "BV421"},
            tool_calls_made=[{"tool": "analyze_panel"}],
            assignment_accuracy=0.8,
            complexity_index=2.5,
            ground_truth_ci=4.0,
            ci_improvement=0.375,
            latency_seconds=10.0,
            input_tokens=200,
            output_tokens=100
        ),
    ]

    results = ExperimentResults(
        experiment_name="test_experiment",
        trials=sample_trials
    )

    print(quick_summary(results))
    print("\n" + "="*60 + "\n")
    print(generate_report(results))
