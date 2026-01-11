#!/usr/bin/env python3
"""
Generate full review reports showing complete data flow.

This script generates comprehensive reports for manual inspection, showing:
- Prompt sent to benchmark model
- Raw LLM response
- Flattened version (what judge sees)
- Parsed hierarchy (what scorer uses)
- Scoring metrics
- Judge prompt, response, and scores

Usage:
    # From modular pipeline checkpoints
    python scripts/generate_full_review.py --checkpoint-dir results/modular_pipeline

    # From specific JSON files
    python scripts/generate_full_review.py \
        --predictions results/predictions.json \
        --scoring results/scoring_results.json \
        --judge results/judge_results.json

    # Filter by model or condition
    python scripts/generate_full_review.py \
        --checkpoint-dir results/modular_pipeline \
        --model claude-sonnet-cli \
        --condition standard-cot

    # Limit output size
    python scripts/generate_full_review.py \
        --checkpoint-dir results/modular_pipeline \
        --max-entries 10 \
        --max-response-length 1000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis.full_review_report import (  # noqa: E402
    generate_full_review_report,
)
from curation.schemas import TestCase  # noqa: E402
from experiments.batch_scorer import ScoringResult  # noqa: E402
from experiments.llm_judge import JudgeResult  # noqa: E402
from experiments.prediction_collector import Prediction  # noqa: E402


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
            test_cases[tc.test_case_id] = tc
        except Exception as e:
            print(f"Warning: Could not load {path.name}: {e}")

    return test_cases


def load_predictions(path: Path) -> list[Prediction]:
    """Load predictions from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Handle both list format and dict with 'predictions' key
    if isinstance(data, list):
        items = data
    else:
        items = data.get("predictions", data.get("items", []))

    return [Prediction.from_dict(d) for d in items]


def load_scoring_results(path: Path) -> list[ScoringResult]:
    """Load scoring results from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Handle both list format and dict with 'results' key
    if isinstance(data, list):
        items = data
    else:
        items = data.get("results", data.get("items", []))

    return [ScoringResult.from_dict(d) for d in items]


def load_judge_results(path: Path) -> list[JudgeResult]:
    """Load judge results from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Handle both list format and dict with 'results' key
    if isinstance(data, list):
        items = data
    else:
        items = data.get("results", data.get("items", []))

    return [JudgeResult.from_dict(d) for d in items]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate full review reports for manual inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input sources
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory with checkpoint files (predictions.json, scoring_results.json, judge_results.json)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--scoring",
        type=Path,
        help="Path to scoring results JSON file",
    )
    parser.add_argument(
        "--judge",
        type=Path,
        help="Path to judge results JSON file (optional)",
    )

    # Filters
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
        "--test-case",
        type=str,
        help="Filter by test case ID",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: auto-generated in results/)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Maximum number of entries to include (0 = all)",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=2000,
        help="Maximum characters for response sections (default: 2000)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Exclude judge sections from report",
    )

    # Data paths
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("data/ground_truth"),
        help="Directory with ground truth JSON files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input files
    if args.checkpoint_dir:
        predictions_path = args.checkpoint_dir / "predictions.json"
        scoring_path = args.checkpoint_dir / "scoring_results.json"
        judge_path = args.checkpoint_dir / "judge_results.json"
    else:
        if not args.predictions or not args.scoring:
            print("Error: Must specify either --checkpoint-dir or both --predictions and --scoring")
            sys.exit(1)
        predictions_path = args.predictions
        scoring_path = args.scoring
        judge_path = args.judge

    # Check files exist
    if not predictions_path.exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        sys.exit(1)
    if not scoring_path.exists():
        print(f"Error: Scoring file not found: {scoring_path}")
        sys.exit(1)

    # Load data
    print(f"Loading predictions from: {predictions_path}")
    predictions = load_predictions(predictions_path)
    print(f"  Loaded {len(predictions)} predictions")

    print(f"Loading scoring results from: {scoring_path}")
    scoring_results = load_scoring_results(scoring_path)
    print(f"  Loaded {len(scoring_results)} scoring results")

    judge_results = None
    if judge_path and judge_path.exists():
        print(f"Loading judge results from: {judge_path}")
        judge_results = load_judge_results(judge_path)
        print(f"  Loaded {len(judge_results)} judge results")
    elif not args.no_judge:
        print("  No judge results found (will exclude judge sections)")

    # Load test cases
    test_cases_dir = PROJECT_ROOT / args.ground_truth_dir
    test_cases = load_test_cases(test_cases_dir)
    print(f"Loaded {len(test_cases)} test cases from {test_cases_dir}")

    # Apply filters
    if args.model:
        predictions = [p for p in predictions if p.model == args.model]
        scoring_results = [s for s in scoring_results if s.model == args.model]
        if judge_results:
            judge_results = [j for j in judge_results if j.model == args.model]
        print(f"Filtered to model '{args.model}': {len(predictions)} predictions")

    if args.condition:
        predictions = [p for p in predictions if p.condition == args.condition]
        scoring_results = [s for s in scoring_results if s.condition == args.condition]
        if judge_results:
            judge_results = [j for j in judge_results if j.condition == args.condition]
        print(f"Filtered to condition '{args.condition}': {len(predictions)} predictions")

    if args.test_case:
        predictions = [p for p in predictions if p.test_case_id == args.test_case]
        scoring_results = [s for s in scoring_results if s.test_case_id == args.test_case]
        if judge_results:
            judge_results = [j for j in judge_results if j.test_case_id == args.test_case]
        print(f"Filtered to test case '{args.test_case}': {len(predictions)} predictions")

    # Limit entries
    if args.max_entries > 0 and len(predictions) > args.max_entries:
        predictions = predictions[: args.max_entries]
        print(f"Limited to {args.max_entries} entries")

    if not predictions:
        print("No predictions to report after filtering")
        sys.exit(0)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "results" / f"full_review_{timestamp}.md"

    # Generate report
    print(f"\nGenerating full review report...")
    report = generate_full_review_report(
        predictions=predictions,
        scoring_results=scoring_results,
        test_cases=test_cases,
        judge_results=judge_results if not args.no_judge else None,
        output_path=output_path,
        include_judge=not args.no_judge and judge_results is not None,
        max_response_length=args.max_response_length,
    )

    print(f"Report saved to: {output_path}")
    print(f"Report size: {len(report):,} characters")

    # Print quick summary
    print("\n--- Quick Summary ---")
    for line in report.split("\n")[: 30]:
        if line.startswith("|") or line.startswith("#"):
            print(line)


if __name__ == "__main__":
    main()
