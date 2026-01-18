#!/usr/bin/env python3
"""
Score LLM predictions against ground truth gating hierarchies.

Takes predictions.json from predict.py and computes metrics:
- hierarchy_f1: Gate name precision/recall
- synonym_f1: Gate matching via synonym dictionary
- semantic_f1: Gate matching via sentence embeddings
- structure_accuracy: Parent-child relationships correct
- critical_gate_recall: Must-have gates present
- hallucination_rate: Gates not in panel

Usage:
    python scripts/score.py --input results/predictions.json --output results/scores.json
    python scripts/score.py --input results/predictions.json --test-cases data/staging

Examples:
    # Score predictions from predict.py
    python scripts/score.py --input results/predictions.json

    # Use different test cases directory
    python scripts/score.py --input results/predictions.json --test-cases data/staging
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    ensure_parent_dir,
    print_phase,
)

from curation.omip_extractor import load_all_test_cases
from experiments.batch_scorer import (
    BatchScorer,
    ScoringResult,
    compute_aggregate_stats,
)
from experiments.prediction_collector import Prediction


def main():
    parser = argparse.ArgumentParser(
        description="Score predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input predictions JSON file (from predict.py)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output scores JSON file (default: <input_dir>/scores.json)",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "verified",
        help="Directory with test case JSON files (default: data/verified)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints (default: <output_dir>/checkpoints)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        args.output = args.input.parent / "scores.json"

    print_phase("BATCH SCORING")

    # Load test cases
    test_cases = load_all_test_cases(args.test_cases)
    print(f"Loaded {len(test_cases)} test cases from {args.test_cases}")

    # Load predictions
    if not args.input.exists():
        print(f"ERROR: Predictions file not found: {args.input}")
        return 1

    with open(args.input) as f:
        data = json.load(f)
        predictions = [Prediction.from_dict(p) for p in data]
    print(f"Loaded {len(predictions)} predictions from {args.input}")

    # Determine checkpoint directory
    checkpoint_dir = args.checkpoint_dir or (args.output.parent / "checkpoints")

    # Score predictions
    scorer = BatchScorer(test_cases, checkpoint_dir=checkpoint_dir)
    results: list[ScoringResult] = scorer.score_all(predictions)

    print(f"\nScored {len(results)} predictions")

    # Compute aggregate statistics
    stats = compute_aggregate_stats(results)

    # Print summary
    print("\n" + "-" * 40)
    print("OVERALL STATISTICS")
    print("-" * 40)

    overall = stats.get("overall", {})
    metrics = [
        ("hierarchy_f1", "Hierarchy F1"),
        ("synonym_f1", "Synonym F1"),
        ("semantic_f1", "Semantic F1"),
        ("weighted_semantic_f1", "Weighted Semantic F1"),
        ("structure_accuracy", "Structure Accuracy"),
        ("critical_gate_recall", "Critical Gate Recall"),
        ("hallucination_rate", "Hallucination Rate"),
    ]

    for key, label in metrics:
        m = overall.get(key, {})
        if m:
            print(f"  {label}: {m.get('mean', 0):.3f} ± {m.get('std', 0):.3f}")

    print(f"\n  Parse success rate: {overall.get('parse_success_rate', 0):.1%}")
    print(f"  Errors: {overall.get('error_count', 0)}")

    # By model
    print("\n" + "-" * 40)
    print("BY MODEL")
    print("-" * 40)
    for model, model_stats in sorted(stats.get("by_model", {}).items()):
        f1 = model_stats.get("hierarchy_f1", {})
        sem_f1 = model_stats.get("semantic_f1", {})
        n = model_stats.get("n", 0)
        print(f"  {model}:")
        print(f"    Hierarchy F1: {f1.get('mean', 0):.3f} (n={n})")
        print(f"    Semantic F1:  {sem_f1.get('mean', 0):.3f}")

    # By condition
    if stats.get("by_condition"):
        print("\n" + "-" * 40)
        print("BY CONDITION")
        print("-" * 40)
        for cond, cond_stats in sorted(stats.get("by_condition", {}).items()):
            f1 = cond_stats.get("hierarchy_f1", {})
            n = cond_stats.get("n", 0)
            print(f"  {cond}: F1={f1.get('mean', 0):.3f} (n={n})")

    # Save results
    output_file = ensure_parent_dir(args.output)
    output_data = {
        "results": [r.to_dict() for r in results],
        "stats": stats,
        "input_file": str(args.input),
        "test_cases_dir": str(args.test_cases),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
