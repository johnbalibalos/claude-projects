#!/usr/bin/env python3
"""
Run LLM judge evaluation on scored predictions.

Takes scores.json from score.py and uses an LLM to qualitatively evaluate
predictions. This addresses limitations of F1 scoring which penalizes
valid biological alternatives (e.g., "T cells" vs "CD3+ T cells").

Usage:
    python scripts/judge.py --input results/scores.json --output results/judge.json
    python scripts/judge.py --input results/scores.json --judge-model gemini-2.0-flash --force

Examples:
    # Run judge on scored results
    python scripts/judge.py --input results/scores.json --force

    # Use different judge model
    python scripts/judge.py --input results/scores.json --judge-model gemini-2.0-flash

    # Run with specific style
    python scripts/judge.py --input results/scores.json --judge-style qualitative
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    ensure_parent_dir,
    print_phase,
    progress_callback,
)

from experiments.batch_scorer import ScoringResult
from experiments.llm_judge import (
    JUDGE_STYLES,
    JudgeConfig,
    LLMJudge,
    compute_judge_stats,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM judge evaluation on scored predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input scores JSON file (from score.py)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output judge results JSON file (default: <input_dir>/judge.json)",
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
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-pro",
        help="Model to use for LLM judge (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--judge-style",
        type=str,
        choices=JUDGE_STYLES,
        default="default",
        help=f"Judge prompt style: {', '.join(JUDGE_STYLES)} (default: default)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=50,
        help="Number of parallel workers for API calls (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock API calls (for testing pipeline)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        if args.judge_style == "default":
            args.output = args.input.parent / "judge.json"
        else:
            args.output = args.input.parent / f"judge_{args.judge_style}.json"

    print_phase("LLM JUDGE EVALUATION")

    # Load scoring results
    if not args.input.exists():
        print(f"ERROR: Scores file not found: {args.input}")
        return 1

    with open(args.input) as f:
        data = json.load(f)
        scoring_results = [ScoringResult.from_dict(r) for r in data["results"]]
    print(f"Loaded {len(scoring_results)} scoring results from {args.input}")

    # Cost estimate
    cost_per_call = {
        "gemini-2.0-flash": 0.001,
        "gemini-2.5-flash": 0.002,
        "gemini-2.5-pro": 0.01,
    }
    estimated_cost = len(scoring_results) * cost_per_call.get(args.judge_model, 0.01)
    print(f"\nJudge model: {args.judge_model}")
    print(f"Judge style: {args.judge_style}")
    print(f"Total evaluations: {len(scoring_results)}")
    print(f"Estimated cost: ${estimated_cost:.2f}")

    if not args.force and not args.dry_run:
        confirm = input("\nProceed? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return 0

    # Determine checkpoint directory
    checkpoint_dir = args.checkpoint_dir or (args.output.parent / "checkpoints")

    # Configure judge
    config = JudgeConfig(
        model=args.judge_model,
        parallel_workers=args.parallel_workers,
        checkpoint_dir=checkpoint_dir,
        dry_run=args.dry_run,
        prompt_style=args.judge_style,
    )

    judge = LLMJudge(args.test_cases, config)

    # Run judge
    results = judge.judge_all(scoring_results, progress_callback=progress_callback)

    print(f"\nJudged {len(results)} results")

    # Compute statistics
    stats = compute_judge_stats(results)

    # Print summary
    print("\n" + "-" * 40)
    print("JUDGE STATISTICS")
    print("-" * 40)

    overall = stats.get("overall", {})
    for key in ["completeness", "accuracy", "scientific", "overall"]:
        s = overall.get(key, {})
        if s:
            print(f"  {key.capitalize()}: {s.get('mean', 0):.3f} ± {s.get('std', 0):.3f}")

    # By model
    print("\n" + "-" * 40)
    print("BY MODEL")
    print("-" * 40)
    for model, model_stats in sorted(stats.get("by_model", {}).items()):
        s = model_stats.get("overall", {})
        n = model_stats.get("n", 0)
        print(f"  {model}: {s.get('mean', 0):.3f} (n={n})")

    # Common issues
    if stats.get("common_issues"):
        print("\n" + "-" * 40)
        print("COMMON ISSUES")
        print("-" * 40)
        for issue in stats["common_issues"][:5]:
            print(f"  - {issue[:70]}...")

    # Save results
    output_file = ensure_parent_dir(args.output)
    output_data = {
        "results": [r.to_dict() for r in results],
        "stats": stats,
        "judge_model": args.judge_model,
        "judge_style": args.judge_style,
        "input_file": str(args.input),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
