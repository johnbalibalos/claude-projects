#!/usr/bin/env python3
"""
Run full Gemini benchmark.

Tests all 3 Gemini models across all conditions on all test cases.
Requires GOOGLE_API_KEY environment variable.

Usage:
    python scripts/run_gemini_benchmark.py
    python scripts/run_gemini_benchmark.py --dry-run  # Mock API calls
    python scripts/run_gemini_benchmark.py --models gemini-2.0-flash  # Single model
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.conditions import (
    GEMINI_MODELS,
    get_all_conditions,
    print_conditions,
)
from experiments.runner import ExperimentConfig, ExperimentRunner


def estimate_cost(n_test_cases: int, conditions: list) -> dict:
    """Estimate API costs for the experiment."""
    # Approximate token usage per call
    input_tokens = 2000  # Panel + prompt
    output_tokens = 1000  # Gating hierarchy

    # Gemini pricing (per 1M tokens)
    pricing = {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.60},
        "gemini-2.5-pro-preview-05-06": {"input": 1.25, "output": 5.00},
    }

    costs = {}
    total = 0.0

    for model_id in {c.model for c in conditions}:
        n_calls = n_test_cases * len([c for c in conditions if c.model == model_id])
        if model_id in pricing:
            p = pricing[model_id]
            cost = n_calls * (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
            costs[model_id] = {"n_calls": n_calls, "cost": cost}
            total += cost

    return {"by_model": costs, "total": total, "n_api_calls": n_test_cases * len(conditions)}


def main():
    parser = argparse.ArgumentParser(description="Run Gemini benchmark")
    parser.add_argument("--test-cases", type=Path,
                        default=Path("data/ground_truth"),
                        help="Directory with test cases")
    parser.add_argument("--output", type=Path,
                        default=Path("results/gemini_benchmark"),
                        help="Output directory")
    parser.add_argument("--models", nargs="+", choices=GEMINI_MODELS,
                        default=GEMINI_MODELS,
                        help="Gemini models to test")
    parser.add_argument("--context-levels", nargs="+",
                        choices=["minimal", "standard", "rich"],
                        default=["minimal", "standard", "rich"],
                        help="Context levels to test")
    parser.add_argument("--prompt-strategies", nargs="+",
                        choices=["direct", "cot"],
                        default=["direct", "cot"],
                        help="Prompt strategies to test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock API calls (no actual requests)")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of runs for statistical significance")
    parser.add_argument("--force", action="store_true",
                        help="Skip cost confirmation")
    args = parser.parse_args()

    # Check API key
    if not args.dry_run and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set")
        print("Export your API key: export GOOGLE_API_KEY=your_key")
        sys.exit(1)

    # Generate conditions
    conditions = get_all_conditions(
        models=args.models,
        context_levels=args.context_levels,
        prompt_strategies=args.prompt_strategies,
    )

    # Count test cases
    test_cases_dir = Path(__file__).parent.parent / args.test_cases
    n_test_cases = len(list(test_cases_dir.glob("*.json")))

    print("=" * 60)
    print("GEMINI GATING BENCHMARK")
    print("=" * 60)
    print(f"\nTest cases: {n_test_cases}")
    print(f"Conditions: {len(conditions)}")
    print(f"Total API calls: {n_test_cases * len(conditions)}")
    print(f"Number of runs: {args.n_runs}")

    print("\nConditions:")
    print_conditions(conditions)

    # Estimate cost
    cost_estimate = estimate_cost(n_test_cases, conditions)
    print("\n" + "-" * 60)
    print("COST ESTIMATE")
    print("-" * 60)
    for model, data in cost_estimate["by_model"].items():
        print(f"  {model}: {data['n_calls']} calls, ~${data['cost']:.2f}")
    print(f"\n  TOTAL: ~${cost_estimate['total']:.2f}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No actual API calls]")
    elif not args.force:
        print("\nProceed with experiment? (y/n): ", end="")
        if input().strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Create output directory
    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    config = ExperimentConfig(
        name=f"gemini_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        test_cases_dir=str(test_cases_dir),
        output_dir=str(output_dir),
        conditions=conditions,
        dry_run=args.dry_run,
        n_runs=args.n_runs,
    )

    runner = ExperimentRunner(config)

    print("\n" + "=" * 60)
    print("STARTING EXPERIMENT")
    print("=" * 60 + "\n")

    if args.n_runs > 1:
        result = runner.run_multi(args.n_runs)
        print("\n" + result.format_summary())
    else:
        result = runner.run()
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Results: {len(result.results)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Duration: {result.end_time - result.start_time}")

        if result.results:
            avg_f1 = sum(r.hierarchy_f1 for r in result.results) / len(result.results)
            print(f"Average F1: {avg_f1:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
