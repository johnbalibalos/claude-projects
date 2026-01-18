#!/usr/bin/env python3
"""
Collect LLM predictions for flow cytometry gating strategies.

Takes test cases and generates predictions across specified models and conditions.
Outputs a JSON file that can be passed to score.py.

Usage:
    python scripts/predict.py --test-cases data/verified --output results/predictions.json
    python scripts/predict.py --models gemini-2.0-flash --max-cases 1 --force
    python scripts/predict.py --resume --output results/predictions.json

Examples:
    # Quick test with 1 case
    python scripts/predict.py --models gemini-2.0-flash --max-cases 1 --force

    # Full run with multiple models
    python scripts/predict.py --models gemini-2.0-flash claude-sonnet-cli --force

    # Resume interrupted run
    python scripts/predict.py --resume --output results/predictions.json
"""

import argparse
import json
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    ensure_parent_dir,
    print_phase,
    progress_callback,
)

from curation.omip_extractor import load_all_test_cases
from experiments.conditions import get_all_conditions
from experiments.prediction_collector import (
    CollectorConfig,
    PredictionCollector,
)


def main():
    parser = argparse.ArgumentParser(
        description="Collect LLM predictions for gating strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "verified",
        help="Directory with test case JSON files (default: data/verified)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "results" / "predictions.json",
        help="Output JSON file for predictions (default: results/predictions.json)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints (default: <output_dir>/checkpoints)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-sonnet-cli"],
        help="Models to test (default: claude-sonnet-cli)",
    )
    parser.add_argument(
        "--context-levels",
        nargs="+",
        default=["minimal", "standard"],
        help="Context levels to test (default: minimal standard)",
    )
    parser.add_argument(
        "--prompt-strategies",
        nargs="+",
        default=["direct", "cot"],
        help="Prompt strategies (default: direct cot)",
    )
    parser.add_argument(
        "--references",
        nargs="+",
        default=["none"],
        help="Reference modes: none, hipc (default: none)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1,
        help="Number of bootstrap runs per condition (default: 1)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of test cases (for testing)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=6000,
        help="Max output tokens for predictions (default: 6000)",
    )
    parser.add_argument(
        "--cli-delay",
        type=float,
        default=0.5,
        help="Delay between CLI model calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock API calls (for testing pipeline)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run ID for provenance tracking",
    )

    args = parser.parse_args()

    print_phase("PREDICTION COLLECTION")

    # Load test cases
    test_cases = load_all_test_cases(args.test_cases)
    if args.max_cases is not None:
        test_cases = test_cases[: args.max_cases]
    print(f"Loaded {len(test_cases)} test cases from {args.test_cases}")

    # Generate conditions
    conditions = get_all_conditions(
        models=args.models,
        context_levels=args.context_levels,
        prompt_strategies=args.prompt_strategies,
        references=args.references,
    )
    print(f"Generated {len(conditions)} conditions")
    for c in conditions[:5]:
        print(f"  - {c.name}")
    if len(conditions) > 5:
        print(f"  ... and {len(conditions) - 5} more")

    # Determine checkpoint directory
    checkpoint_dir = args.checkpoint_dir or (args.output.parent / "checkpoints")

    # Configure collector
    config = CollectorConfig(
        n_bootstrap=args.n_bootstrap,
        cli_delay_seconds=args.cli_delay,
        checkpoint_dir=checkpoint_dir,
        dry_run=args.dry_run,
        run_id=args.run_id,
        max_tokens=args.max_tokens,
    )

    collector = PredictionCollector(test_cases, conditions, config)

    total_calls = collector.total_calls
    print(f"\nTotal API calls: {total_calls}")

    # Cost estimate
    cost_per_call = {
        "gemini-2.0-flash": 0.0025,
        "gemini-2.5-flash": 0.005,
        "gemini-2.5-pro": 0.025,
        "claude-sonnet-cli": 0.015,
        "claude-opus-cli": 0.125,
        "gpt-4o": 0.02,
    }
    estimated_cost = sum(
        cost_per_call.get(m, 0.01) * len(test_cases) * len(args.context_levels) * len(args.prompt_strategies) * args.n_bootstrap
        for m in args.models
    )
    print(f"Estimated cost: ${estimated_cost:.2f}")

    if not args.force and not args.dry_run:
        confirm = input("\nProceed? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    # Collect predictions
    predictions = collector.collect(resume=args.resume, progress_callback=progress_callback)

    print(f"\nCollected {len(predictions)} predictions")
    errors = [p for p in predictions if p.error]
    print(f"  Successful: {len(predictions) - len(errors)}")
    print(f"  Errors: {len(errors)}")

    # Save predictions
    output_file = ensure_parent_dir(args.output)
    with open(output_file, "w") as f:
        json.dump([p.to_dict() for p in predictions], f, indent=2)
    print(f"\nSaved to: {output_file}")

    # Print summary by model
    by_model = {}
    for p in predictions:
        model = p.condition_name.split("_")[0] if p.condition_name else "unknown"
        by_model.setdefault(model, {"total": 0, "errors": 0})
        by_model[model]["total"] += 1
        if p.error:
            by_model[model]["errors"] += 1

    print("\nBy model:")
    for model, stats in sorted(by_model.items()):
        success_rate = (stats["total"] - stats["errors"]) / stats["total"] * 100
        print(f"  {model}: {stats['total']} predictions ({success_rate:.0f}% success)")


if __name__ == "__main__":
    main()
