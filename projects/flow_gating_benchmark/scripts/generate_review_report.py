#!/usr/bin/env python3
"""
Generate manual review reports from existing experiment results.

This script generates detailed comparison reports without re-running experiments.
Use this to regenerate reports from saved result JSON files.

Usage:
    python scripts/generate_review_report.py results/experiment_results_*.json
    python scripts/generate_review_report.py results/experiment_results_*.json --level full
    python scripts/generate_review_report.py results/experiment_results_*.json --level outliers --outlier-f1 0.4
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.manual_review_report import (
    OutlierThresholds,
    generate_manual_review_report,
    is_outlier,
)
from curation.schemas import TestCase
from evaluation.metrics import EvaluationResult
from evaluation.scorer import ScoringResult


def load_test_cases(test_cases_dir: Path) -> dict[str, TestCase]:
    """Load all test cases into a dictionary."""
    test_cases = {}
    for path in test_cases_dir.glob("omip_*.json"):
        with open(path) as f:
            data = json.load(f)

        # Fix null fluorophores
        for entry in data.get("panel", {}).get("entries", []):
            if entry.get("fluorophore") is None:
                entry["fluorophore"] = "Unknown"

        try:
            tc = TestCase(**data)
            test_cases[tc.omip_id] = tc
        except Exception as e:
            print(f"Warning: Could not load {path.name}: {e}")

    return test_cases


def reconstruct_scoring_result(result_dict: dict[str, Any]) -> ScoringResult:
    """Reconstruct ScoringResult from serialized dict."""
    eval_dict = result_dict.get("evaluation")
    evaluation = None

    if eval_dict:
        evaluation = EvaluationResult(
            hierarchy_f1=eval_dict.get("hierarchy_f1", 0),
            hierarchy_precision=eval_dict.get("hierarchy_precision", 0),
            hierarchy_recall=eval_dict.get("hierarchy_recall", 0),
            structure_accuracy=eval_dict.get("structure_accuracy", 0),
            critical_gate_recall=eval_dict.get("critical_gate_recall", 0),
            hallucination_rate=eval_dict.get("hallucination_rate", 0),
            depth_accuracy=eval_dict.get("depth_accuracy", 0),
            predicted_gates=eval_dict.get("predicted_gates", []),
            ground_truth_gates=eval_dict.get("ground_truth_gates", []),
            matching_gates=eval_dict.get("matching_gates", []),
            missing_gates=eval_dict.get("missing_gates", []),
            extra_gates=eval_dict.get("extra_gates", []),
            hallucinated_gates=eval_dict.get("hallucinated_gates", []),
            missing_critical=eval_dict.get("missing_critical", []),
            correct_relationships=eval_dict.get("correct_relationships", 0),
            total_relationships=eval_dict.get("total_relationships", 0),
            structure_errors=eval_dict.get("structure_errors", []),
        )

    return ScoringResult(
        test_case_id=result_dict.get("test_case_id", "unknown"),
        model=result_dict.get("model", "unknown"),
        condition=result_dict.get("condition", "unknown"),
        parse_success=result_dict.get("parse_success", False),
        parse_format=result_dict.get("parse_format"),
        parse_error=result_dict.get("parse_error"),
        evaluation=evaluation,
        parsed_hierarchy=result_dict.get("parsed_hierarchy"),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate manual review reports from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Report Levels:
    summary   - Overall metrics table + worst performers (default)
    outliers  - Summary + detailed reports for outliers
    full      - Detailed report for every test case

Examples:
    python scripts/generate_review_report.py results/experiment_results_*.json
    python scripts/generate_review_report.py results/experiment_results_*.json --level full
    python scripts/generate_review_report.py results/experiment_results_*.json --level outliers --outlier-f1 0.4
        """,
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to experiment results JSON file",
    )
    parser.add_argument(
        "--level",
        choices=["summary", "outliers", "full"],
        default="summary",
        help="Report detail level (default: summary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: auto-generated)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name",
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Filter by condition name",
    )
    parser.add_argument(
        "--outlier-f1",
        type=float,
        default=0.3,
        help="F1 threshold for outlier detection (default: 0.3)",
    )
    parser.add_argument(
        "--outlier-halluc",
        type=float,
        default=0.2,
        help="Hallucination threshold for outlier detection (default: 0.2)",
    )
    parser.add_argument(
        "--outlier-critical",
        type=float,
        default=0.5,
        help="Critical recall threshold for outlier detection (default: 0.5)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("data/verified"),
        help="Directory with ground truth JSON files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    with open(args.results_file) as f:
        results_data = json.load(f)

    # Reconstruct ScoringResults
    raw_results = results_data.get("results", [])
    results = [reconstruct_scoring_result(r) for r in raw_results]
    print(f"Loaded {len(results)} results")

    # Load test cases
    project_dir = Path(__file__).parent.parent
    test_cases_dir = project_dir / args.ground_truth_dir
    test_cases = load_test_cases(test_cases_dir)
    print(f"Loaded {len(test_cases)} test cases")

    # Set up thresholds
    thresholds = OutlierThresholds(
        min_f1=args.outlier_f1,
        max_hallucination=args.outlier_halluc,
        min_critical_recall=args.outlier_critical,
    )

    # Count outliers
    outliers = [r for r in results if is_outlier(r, thresholds)]
    print(f"Found {len(outliers)} outliers")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_dir / "results" / f"manual_review_{args.level}_{timestamp}.md"

    # Generate report
    print(f"\nGenerating {args.level} report...")
    generate_manual_review_report(
        results=results,
        test_cases=test_cases,
        level=args.level,
        output_path=output_path,
        model=args.model,
        condition=args.condition,
        thresholds=thresholds,
    )

    print(f"Report saved to: {output_path}")

    # Print summary stats
    valid = [r for r in results if r.parse_success and r.evaluation]
    if valid:
        avg_f1 = sum(r.hierarchy_f1 for r in valid) / len(valid)
        print(f"\nSummary: {len(valid)} valid results, avg F1 = {avg_f1:.3f}")


if __name__ == "__main__":
    main()
