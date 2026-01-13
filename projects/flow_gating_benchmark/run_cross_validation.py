#!/usr/bin/env python3
"""
Cross-validation test script for flow gating benchmark results.

Tests the cross-validation subagent on existing benchmark results,
using Opus to debate discrepancies between predictions and ground truth.

Usage:
    python run_cross_validation.py [--n-samples 5] [--results-file PATH]

Cost estimate: ~$0.05-0.10 per sample with Opus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add libs to path - import directly to avoid numpy dependency in __init__
libs_path = Path(__file__).parent.parent.parent / "libs"
sys.path.insert(0, str(libs_path))

# Direct import to avoid loading full hypothesis_pipeline package
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hypothesis_pipeline.cross_validator",
    libs_path / "hypothesis_pipeline" / "cross_validator.py"
)
cross_validator_module = importlib.util.module_from_spec(spec)
sys.modules["hypothesis_pipeline.cross_validator"] = cross_validator_module
spec.loader.exec_module(cross_validator_module)

CrossValidator = cross_validator_module.CrossValidator
BatchCrossValidator = cross_validator_module.BatchCrossValidator
create_anthropic_validator = cross_validator_module.create_anthropic_validator
CrossValidationResult = cross_validator_module.CrossValidationResult


def load_results(results_path: Path) -> dict:
    """Load experiment results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def load_ground_truths(data_dir: Path) -> dict[str, dict]:
    """Load all ground truth test cases."""
    ground_truths = {}

    # Load from verified directory
    gt_dir = data_dir / "verified"
    if gt_dir.exists():
        for gt_file in gt_dir.glob("*.json"):
            if gt_file.stem.startswith("omip_"):
                with open(gt_file) as f:
                    tc = json.load(f)
                    # Use OMIP-XXX format for test_case_id matching
                    test_case_id = tc.get("test_case_id", gt_file.stem.upper().replace("_", "-"))
                    ground_truths[test_case_id] = tc

    return ground_truths


def extract_hierarchy_for_comparison(tc: dict) -> dict:
    """Extract gating hierarchy in a format suitable for comparison."""
    return tc.get("gating_hierarchy", {})


def extract_panel_markers(tc: dict) -> list[str]:
    """Extract panel marker names from test case."""
    panel = tc.get("panel", {})
    entries = panel.get("entries", [])
    return [e.get("marker", "") for e in entries if e.get("marker")]


def format_result_for_display(
    result: dict,
    validation: CrossValidationResult,
) -> str:
    """Format a result with its validation for display."""
    lines = [
        "=" * 80,
        f"Test Case: {result.get('test_case_id', 'unknown')}",
        f"Model: {result.get('model', 'unknown')}",
        f"Condition: {result.get('condition', 'unknown')}",
        "-" * 40,
        f"Original Metrics:",
        f"  Hierarchy F1: {result.get('evaluation', {}).get('hierarchy_f1', 0):.3f}",
        f"  Structure Accuracy: {result.get('evaluation', {}).get('structure_accuracy', 0):.3f}",
        f"  Critical Gate Recall: {result.get('evaluation', {}).get('critical_gate_recall', 0):.3f}",
        "-" * 40,
        f"Cross-Validation Result:",
        f"  Confidence Score: {validation.confidence_score}/100",
        f"  Interpretation: {validation.interpretation}",
        f"  Trust Prediction: {validation.should_trust_prediction}",
        "-" * 40,
        f"Discrepancies:",
        f"  Critical: {validation.n_critical}",
        f"  Moderate: {validation.n_moderate}",
        f"  Minor: {validation.n_minor}",
    ]

    if validation.discrepancies:
        lines.append("-" * 40)
        lines.append("Top Discrepancies:")
        for i, d in enumerate(validation.discrepancies[:5]):
            lines.append(f"  {i+1}. [{d.severity.upper()}] {d.aspect}")
            lines.append(f"     Prediction: {d.prediction_value[:50]}...")
            lines.append(f"     Ground Truth: {d.ground_truth_value[:50]}...")
            lines.append(f"     Leaning: {d.leaning}/100 - {d.reasoning[:80]}...")

    lines.append("-" * 40)
    lines.append("Prediction Defense (excerpt):")
    lines.append(f"  {validation.prediction_defense[:200]}...")
    lines.append("")
    lines.append("Ground Truth Defense (excerpt):")
    lines.append(f"  {validation.ground_truth_defense[:200]}...")
    lines.append("")
    lines.append("Reconciliation:")
    lines.append(f"  {validation.reconciliation[:300]}...")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run cross-validation on benchmark results")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Path to results JSON file (default: most recent)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of samples to validate (default: 3)",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.0,
        help="Minimum F1 to include (default: 0.0)",
    )
    parser.add_argument(
        "--max-f1",
        type=float,
        default=0.9,
        help="Maximum F1 to include - focus on imperfect predictions (default: 0.9)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be validated without making API calls",
    )

    args = parser.parse_args()

    # Find results file
    results_dir = Path(__file__).parent / "results"
    if args.results_file:
        results_path = args.results_file
    else:
        # Find most recent experiment results
        result_files = sorted(results_dir.glob("experiment_results_*.json"))
        if not result_files:
            print("No experiment results found in results/")
            sys.exit(1)
        results_path = result_files[-1]

    print(f"Loading results from: {results_path}")
    results_data = load_results(results_path)

    # Load ground truths
    data_dir = Path(__file__).parent / "data"
    print(f"Loading ground truths from: {data_dir}")
    ground_truths = load_ground_truths(data_dir)
    print(f"Found {len(ground_truths)} ground truth test cases")

    # Filter results
    all_results = results_data.get("results", [])
    print(f"Total results in file: {len(all_results)}")

    # Filter by F1 range to focus on interesting cases
    filtered_results = []
    for r in all_results:
        eval_data = r.get("evaluation", {})
        f1 = eval_data.get("hierarchy_f1", 0)
        if args.min_f1 <= f1 <= args.max_f1:
            if r.get("test_case_id") in ground_truths:
                filtered_results.append(r)

    print(f"Results in F1 range [{args.min_f1:.2f}, {args.max_f1:.2f}]: {len(filtered_results)}")

    # Select samples
    samples = filtered_results[:args.n_samples]
    print(f"Selected {len(samples)} samples for cross-validation")

    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        for i, r in enumerate(samples):
            tc_id = r.get("test_case_id")
            f1 = r.get("evaluation", {}).get("hierarchy_f1", 0)
            print(f"{i+1}. {tc_id} - F1: {f1:.3f}")
        print("\nEstimated cost: ${:.2f} - ${:.2f}".format(
            len(samples) * 0.05,
            len(samples) * 0.10,
        ))
        sys.exit(0)

    # Create validator
    print("\nInitializing Opus cross-validator...")
    try:
        validator = create_anthropic_validator()
    except Exception as e:
        print(f"Failed to create validator: {e}")
        print("Make sure ANTHROPIC_API_KEY is set")
        sys.exit(1)

    # Run validation
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)

    validation_results = []

    for i, result in enumerate(samples):
        tc_id = result.get("test_case_id")
        print(f"\n[{i+1}/{len(samples)}] Validating {tc_id}...")

        gt = ground_truths[tc_id]
        gt_hierarchy = extract_hierarchy_for_comparison(gt)
        markers = extract_panel_markers(gt)

        try:
            validation = validator.validate_gating_result(
                result_dict=result.get("evaluation", result),
                ground_truth_hierarchy=gt_hierarchy,
                panel_markers=markers,
            )

            print(format_result_for_display(result, validation))

            validation_results.append({
                "test_case_id": tc_id,
                "model": result.get("model"),
                "condition": result.get("condition"),
                "original_f1": result.get("evaluation", {}).get("hierarchy_f1"),
                "confidence_score": validation.confidence_score,
                "interpretation": validation.interpretation,
                "should_trust": validation.should_trust_prediction,
                "n_discrepancies": len(validation.discrepancies),
                "n_critical": validation.n_critical,
                "n_moderate": validation.n_moderate,
                "n_minor": validation.n_minor,
                "prediction_defense": validation.prediction_defense,
                "ground_truth_defense": validation.ground_truth_defense,
                "reconciliation": validation.reconciliation,
                "discrepancies": [
                    {
                        "aspect": d.aspect,
                        "prediction_value": d.prediction_value,
                        "ground_truth_value": d.ground_truth_value,
                        "severity": d.severity,
                        "leaning": d.leaning,
                        "reasoning": d.reasoning,
                    }
                    for d in validation.discrepancies
                ],
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            validation_results.append({
                "test_case_id": tc_id,
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid_results = [r for r in validation_results if "error" not in r]
    if valid_results:
        scores = [r["confidence_score"] for r in valid_results]
        print(f"Samples validated: {len(valid_results)}/{len(samples)}")
        print(f"Mean confidence score: {sum(scores)/len(scores):.1f}/100")
        print(f"Score range: {min(scores)} - {max(scores)}")

        # Count interpretations
        interps = {}
        for r in valid_results:
            i = r["interpretation"]
            interps[i] = interps.get(i, 0) + 1
        print("\nInterpretation breakdown:")
        for interp, count in sorted(interps.items()):
            print(f"  {interp}: {count}")

        # Interesting findings
        trusted = [r for r in valid_results if r["should_trust"]]
        print(f"\nPredictions deemed acceptable: {len(trusted)}/{len(valid_results)}")

        # Cases where low F1 but high confidence (prediction might be right)
        interesting = [
            r for r in valid_results
            if r["original_f1"] < 0.5 and r["confidence_score"] > 60
        ]
        if interesting:
            print(f"\nInteresting cases (low F1 but prediction may be valid):")
            for r in interesting:
                print(f"  - {r['test_case_id']}: F1={r['original_f1']:.3f}, confidence={r['confidence_score']}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"cross_validation_{timestamp}.json"

    output_data = {
        "source_results": str(results_path),
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(samples),
        "f1_range": [args.min_f1, args.max_f1],
        "results": validation_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
