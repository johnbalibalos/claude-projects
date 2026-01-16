#!/usr/bin/env python3
"""
Analyze structure errors in gating hierarchy predictions.

Categorizes errors into:
1. WRONG_PARENT - Gate exists but has incorrect parent
2. MISSING_GATE - Gate from ground truth not found in prediction
3. SWAPPED_RELATIONSHIP - Parent-child relationship inverted
4. WRONG_DEPTH - Gate at incorrect depth in hierarchy
5. ORPHANED_GATE - Predicted gate with broken ancestor chain
6. MISSING_ANCESTOR - Intermediate gate in lineage missing
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def normalize_gate_semantic(name: str) -> str:
    """Normalize gate name for semantic comparison."""
    if not name:
        return ""
    # Lowercase
    n = name.lower().strip()
    # Normalize + and - markers
    n = re.sub(r'\s*\+\s*', '+', n)
    n = re.sub(r'\s*-\s*', '-', n)
    # Remove extra whitespace
    n = re.sub(r'\s+', ' ', n)
    return n


def extract_all_parent_relationships(hierarchy: dict) -> list[tuple[str, str | None, int]]:
    """Extract ALL parent-child relationships from hierarchy."""
    relationships: list[tuple[str, str | None, int]] = []

    def traverse_dict(node: dict, parent: str | None = None, depth: int = 0) -> None:
        if "name" in node:
            relationships.append((node["name"], parent, depth))
            for child in node.get("children", []):
                traverse_dict(child, node["name"], depth + 1)

    if "root" in hierarchy:
        traverse_dict(hierarchy["root"])
    elif "name" in hierarchy:
        traverse_dict(hierarchy)

    return relationships


def extract_gate_names(hierarchy: dict) -> set[str]:
    """Extract all gate names from a hierarchy."""
    gates: set[str] = set()

    def traverse_dict(node: dict) -> None:
        if "name" in node:
            gates.add(node["name"])
        for child in node.get("children", []):
            traverse_dict(child)

    if "root" in hierarchy:
        traverse_dict(hierarchy["root"])
    elif "name" in hierarchy:
        traverse_dict(hierarchy)

    return gates


@dataclass
class StructureErrorCategory:
    """Categorized structure error."""

    category: str
    gate: str
    expected_parent: str | None
    predicted_parent: str | None
    expected_depth: int | None
    predicted_depth: int | None
    details: str


@dataclass
class StructureAnalysis:
    """Complete structure error analysis for a prediction."""

    test_case_id: str
    model: str
    condition: str

    # Error counts by category
    wrong_parent: int = 0
    missing_gate: int = 0
    swapped_relationship: int = 0
    wrong_depth: int = 0
    orphaned_gate: int = 0
    missing_ancestor: int = 0

    # Detailed errors
    errors: list[StructureErrorCategory] = field(default_factory=list)

    # Metrics
    total_gt_relationships: int = 0
    correct_relationships: int = 0


def normalize_hierarchy_for_analysis(hierarchy: dict) -> dict:
    """Extract normalized relationships from hierarchy."""
    rels = extract_all_parent_relationships(hierarchy)

    # Build lookup structures
    gate_to_parent: dict[str, tuple[str | None, int]] = {}  # gate -> (parent, depth)
    parent_to_children: dict[str, list[str]] = defaultdict(list)

    for gate, parent, depth in rels:
        norm_gate = normalize_gate_semantic(gate)
        norm_parent = normalize_gate_semantic(parent) if parent else None
        gate_to_parent[norm_gate] = (norm_parent, depth)
        if norm_parent:
            parent_to_children[norm_parent].append(norm_gate)

    return {
        "gate_to_parent": gate_to_parent,
        "parent_to_children": parent_to_children,
        "all_gates": set(gate_to_parent.keys()),
    }


def analyze_structure_errors(
    predicted: dict,
    ground_truth: dict,
) -> StructureAnalysis:
    """Analyze and categorize structure errors."""

    pred_info = normalize_hierarchy_for_analysis(predicted)
    gt_info = normalize_hierarchy_for_analysis(ground_truth)

    analysis = StructureAnalysis(
        test_case_id="",
        model="",
        condition="",
    )

    pred_gate_to_parent = pred_info["gate_to_parent"]
    gt_gate_to_parent = gt_info["gate_to_parent"]
    pred_parent_to_children = pred_info["parent_to_children"]
    gt_parent_to_children = gt_info["parent_to_children"]
    pred_gates = pred_info["all_gates"]
    gt_gates = gt_info["all_gates"]

    analysis.total_gt_relationships = len(gt_gate_to_parent)

    # Check each ground truth relationship
    for gt_gate, (gt_parent, gt_depth) in gt_gate_to_parent.items():
        if gt_gate not in pred_gates:
            # Gate completely missing
            analysis.missing_gate += 1
            analysis.errors.append(StructureErrorCategory(
                category="MISSING_GATE",
                gate=gt_gate,
                expected_parent=gt_parent,
                predicted_parent=None,
                expected_depth=gt_depth,
                predicted_depth=None,
                details=f"Gate '{gt_gate}' not found in prediction",
            ))
            continue

        pred_parent, pred_depth = pred_gate_to_parent[gt_gate]

        # Check if relationship is correct
        if pred_parent == gt_parent and pred_depth == gt_depth:
            analysis.correct_relationships += 1
            continue

        # Categorize the error

        # Check for swapped relationship (parent-child inverted)
        if gt_parent and pred_parent:
            # Is the expected parent now a child of this gate?
            if gt_parent in pred_parent_to_children.get(gt_gate, []):
                analysis.swapped_relationship += 1
                analysis.errors.append(StructureErrorCategory(
                    category="SWAPPED_RELATIONSHIP",
                    gate=gt_gate,
                    expected_parent=gt_parent,
                    predicted_parent=pred_parent,
                    expected_depth=gt_depth,
                    predicted_depth=pred_depth,
                    details=f"'{gt_gate}' should be child of '{gt_parent}', but prediction has '{gt_parent}' as child of '{gt_gate}'",
                ))
                continue

        # Check for wrong depth (same parent but different depth)
        if pred_parent == gt_parent and pred_depth != gt_depth:
            analysis.wrong_depth += 1
            analysis.errors.append(StructureErrorCategory(
                category="WRONG_DEPTH",
                gate=gt_gate,
                expected_parent=gt_parent,
                predicted_parent=pred_parent,
                expected_depth=gt_depth,
                predicted_depth=pred_depth,
                details=f"'{gt_gate}' at depth {pred_depth}, expected depth {gt_depth}",
            ))
            continue

        # Check for missing ancestor (parent exists in GT but not connected in pred)
        if gt_parent and gt_parent not in pred_gates:
            analysis.missing_ancestor += 1
            analysis.errors.append(StructureErrorCategory(
                category="MISSING_ANCESTOR",
                gate=gt_gate,
                expected_parent=gt_parent,
                predicted_parent=pred_parent,
                expected_depth=gt_depth,
                predicted_depth=pred_depth,
                details=f"'{gt_gate}' expected parent '{gt_parent}' not in prediction",
            ))
            continue

        # Default: wrong parent
        analysis.wrong_parent += 1
        analysis.errors.append(StructureErrorCategory(
            category="WRONG_PARENT",
            gate=gt_gate,
            expected_parent=gt_parent,
            predicted_parent=pred_parent,
            expected_depth=gt_depth,
            predicted_depth=pred_depth,
            details=f"'{gt_gate}' has parent '{pred_parent}', expected '{gt_parent}'",
        ))

    # Check for orphaned gates (predicted gates whose expected parent chain is broken)
    for pred_gate, (pred_parent, pred_depth) in pred_gate_to_parent.items():
        if pred_gate in gt_gates:
            continue  # Already analyzed
        if pred_gate not in gt_gates and pred_parent and pred_parent not in gt_gates:
            # This is an extra gate with an extra parent - might be orphaned
            # Check if it connects to any valid chain
            current = pred_parent
            depth = 0
            while current and depth < 20:
                if current in gt_gates:
                    break
                parent_info = pred_gate_to_parent.get(current)
                if parent_info:
                    current = parent_info[0]
                else:
                    current = None
                depth += 1

            if current is None or current not in gt_gates:
                analysis.orphaned_gate += 1
                analysis.errors.append(StructureErrorCategory(
                    category="ORPHANED_GATE",
                    gate=pred_gate,
                    expected_parent=None,
                    predicted_parent=pred_parent,
                    expected_depth=None,
                    predicted_depth=pred_depth,
                    details=f"Predicted gate '{pred_gate}' with no valid ancestor chain",
                ))

    return analysis


def parse_hierarchy_from_response(raw_response: str) -> dict | None:
    """Extract JSON hierarchy from LLM response."""
    if not raw_response:
        return None

    # Try to find JSON in the response
    # Look for ```json blocks first
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object - find first { and matching }
    first_brace = raw_response.find('{')
    if first_brace != -1:
        depth = 0
        for i, char in enumerate(raw_response[first_brace:], first_brace):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw_response[first_brace:i+1])
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def load_test_cases(data_dir: Path) -> dict[str, dict]:
    """Load ground truth hierarchies."""
    test_cases = {}
    for json_file in data_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            test_case_id = data.get("test_case_id", json_file.stem)
            test_cases[test_case_id] = data
    return test_cases


def main():
    """Run structure error analysis."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "full_benchmark_20260114"
    data_dir = project_root / "data" / "verified"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Load predictions
    predictions_file = results_dir / "predictions.json"
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        sys.exit(1)

    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file) as f:
        predictions = json.load(f)

    # Load test cases
    print(f"Loading test cases from {data_dir}...")
    test_cases = load_test_cases(data_dir)
    print(f"Loaded {len(test_cases)} test cases")

    # Analyze each prediction
    all_analyses: list[StructureAnalysis] = []
    error_counts = Counter()
    errors_by_category: dict[str, list[StructureErrorCategory]] = defaultdict(list)
    errors_by_model: dict[str, Counter] = defaultdict(Counter)
    errors_by_test_case: dict[str, Counter] = defaultdict(Counter)

    print(f"\nAnalyzing {len(predictions)} predictions...")

    for i, pred in enumerate(predictions):
        test_case_id = pred.get("test_case_id")
        model = pred.get("model", "unknown")
        condition = pred.get("condition", "unknown")

        if test_case_id not in test_cases:
            continue

        gt = test_cases[test_case_id]
        gt_hierarchy = gt.get("gating_hierarchy", gt.get("hierarchy", {}))

        # Parse predicted hierarchy from raw_response
        raw_response = pred.get("raw_response", "")
        parsed = parse_hierarchy_from_response(raw_response)
        if not parsed:
            continue

        analysis = analyze_structure_errors(parsed, gt_hierarchy)
        analysis.test_case_id = test_case_id
        analysis.model = model
        analysis.condition = condition

        all_analyses.append(analysis)

        # Aggregate counts
        error_counts["WRONG_PARENT"] += analysis.wrong_parent
        error_counts["MISSING_GATE"] += analysis.missing_gate
        error_counts["SWAPPED_RELATIONSHIP"] += analysis.swapped_relationship
        error_counts["WRONG_DEPTH"] += analysis.wrong_depth
        error_counts["ORPHANED_GATE"] += analysis.orphaned_gate
        error_counts["MISSING_ANCESTOR"] += analysis.missing_ancestor

        for err in analysis.errors:
            errors_by_category[err.category].append(err)
            errors_by_model[model][err.category] += 1
            errors_by_test_case[test_case_id][err.category] += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(predictions)}")

    # Print results
    print("\n" + "=" * 80)
    print("STRUCTURE ERROR ANALYSIS RESULTS")
    print("=" * 80)

    total_errors = sum(error_counts.values())
    print(f"\nTotal predictions analyzed: {len(all_analyses)}")
    print(f"Total structure errors: {total_errors}")

    print("\n--- Error Distribution by Category ---")
    print(f"{'Category':<25} {'Count':>8} {'Percentage':>12}")
    print("-" * 50)
    for category, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_errors if total_errors > 0 else 0
        print(f"{category:<25} {count:>8} {pct:>11.1f}%")

    print("\n--- Error Distribution by Model ---")
    print(f"{'Model':<25} {'WRONG_PARENT':>12} {'MISSING_GATE':>13} {'SWAPPED':>8} {'WRONG_DEPTH':>12} {'ORPHANED':>9} {'MISS_ANCESTOR':>14}")
    print("-" * 100)
    for model in sorted(errors_by_model.keys()):
        counts = errors_by_model[model]
        print(f"{model:<25} {counts['WRONG_PARENT']:>12} {counts['MISSING_GATE']:>13} {counts['SWAPPED_RELATIONSHIP']:>8} {counts['WRONG_DEPTH']:>12} {counts['ORPHANED_GATE']:>9} {counts['MISSING_ANCESTOR']:>14}")

    print("\n--- Error Distribution by Test Case ---")
    print(f"{'Test Case':<25} {'WRONG_PARENT':>12} {'MISSING_GATE':>13} {'SWAPPED':>8} {'WRONG_DEPTH':>12}")
    print("-" * 75)
    for tc in sorted(errors_by_test_case.keys()):
        counts = errors_by_test_case[tc]
        print(f"{tc:<25} {counts['WRONG_PARENT']:>12} {counts['MISSING_GATE']:>13} {counts['SWAPPED_RELATIONSHIP']:>8} {counts['WRONG_DEPTH']:>12}")

    # Sample errors for each category
    print("\n--- Sample Errors by Category ---")
    for category in ["WRONG_PARENT", "SWAPPED_RELATIONSHIP", "MISSING_ANCESTOR", "WRONG_DEPTH"]:
        errors = errors_by_category[category][:5]
        if errors:
            print(f"\n{category} (showing {len(errors)} of {len(errors_by_category[category])}):")
            for err in errors:
                print(f"  • {err.details}")

    # Save detailed results
    output_file = results_dir / "structure_error_analysis.json"
    output_data = {
        "summary": {
            "total_predictions": len(all_analyses),
            "total_errors": total_errors,
            "error_counts": dict(error_counts),
        },
        "by_model": {m: dict(c) for m, c in errors_by_model.items()},
        "by_test_case": {tc: dict(c) for tc, c in errors_by_test_case.items()},
        "sample_errors": {
            cat: [{"gate": e.gate, "expected_parent": e.expected_parent,
                   "predicted_parent": e.predicted_parent, "details": e.details}
                  for e in errs[:20]]
            for cat, errs in errors_by_category.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
