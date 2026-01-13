#!/usr/bin/env python3
"""
Run multi-judge on opus-cli predictions only, then merge with existing results.

Usage:
    python scripts/run_opus_judge.py --dry-run  # Test without API calls
    python scripts/run_opus_judge.py --force    # Run with API calls
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.batch_scorer import ScoringResult
from experiments.llm_judge import JUDGE_STYLES, JudgeConfig, LLMJudge


def load_opus_scoring_results(multi_judge_dir: Path) -> list[ScoringResult]:
    """Load only opus-cli scoring results."""
    # Check opus_cli_run first (dedicated opus run)
    opus_scoring = multi_judge_dir.parent / "opus_cli_run" / "scoring_results.json"
    if opus_scoring.exists():
        with open(opus_scoring) as f:
            data = json.load(f)
        all_results = [ScoringResult.from_dict(r) for r in data["results"]]
        opus_results = [r for r in all_results if "opus" in r.model.lower()]
        if opus_results:
            print(f"Loaded {len(opus_results)} opus-cli scoring results from opus_cli_run/")
            return opus_results

    # Fall back to multi_judge_run scoring
    scoring_file = multi_judge_dir / "scoring_results.json"
    if not scoring_file.exists():
        raise FileNotFoundError("No scoring results found")

    with open(scoring_file) as f:
        data = json.load(f)

    all_results = [ScoringResult.from_dict(r) for r in data["results"]]
    opus_results = [r for r in all_results if "opus" in r.model.lower()]

    print(f"Loaded {len(opus_results)} opus-cli scoring results (from {len(all_results)} total)")
    return opus_results


def load_existing_judge_results(multi_judge_dir: Path, style: str) -> list[dict]:
    """Load existing judge results for a style."""
    if style == "default":
        judge_file = multi_judge_dir / "judge_results.json"
    else:
        judge_file = multi_judge_dir / f"judge_results_{style}.json"

    if not judge_file.exists():
        return []

    with open(judge_file) as f:
        data = json.load(f)

    return data.get("results", [])


def save_merged_results(
    multi_judge_dir: Path,
    style: str,
    existing_results: list[dict],
    new_results: list,
    stats: dict,
):
    """Save merged judge results."""
    # Convert new results to dicts
    new_dicts = [r.to_dict() for r in new_results]

    # Merge
    merged = existing_results + new_dicts

    # Save
    if style == "default":
        output_file = multi_judge_dir / "judge_results.json"
    else:
        output_file = multi_judge_dir / f"judge_results_{style}.json"

    # Backup existing
    if output_file.exists():
        backup = output_file.with_suffix(".json.bak")
        output_file.rename(backup)
        print(f"  Backed up existing to {backup.name}")

    with open(output_file, "w") as f:
        json.dump({
            "results": merged,
            "stats": stats,
            "judge_style": style,
            "timestamp": datetime.now().isoformat(),
            "note": f"Merged: {len(existing_results)} existing + {len(new_dicts)} opus",
        }, f, indent=2)

    print(f"  Saved {len(merged)} total results to {output_file.name}")


def progress_callback(completed: int, total: int, last_result=None):
    """Progress callback for judge."""
    pct = completed / total * 100 if total > 0 else 0
    status = ""
    if last_result and hasattr(last_result, 'overall'):
        status = f" | last: {last_result.overall:.2f}"
    print(f"\r  Progress: {completed}/{total} ({pct:.1f}%){status}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run multi-judge on opus-cli predictions")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation")
    parser.add_argument("--styles", nargs="+", default=JUDGE_STYLES,
                        help=f"Judge styles to run (default: all). Options: {JUDGE_STYLES}")
    parser.add_argument("--output", type=Path, default=Path("results/multi_judge_run"),
                        help="Output directory")
    parser.add_argument("--test-cases", type=Path, default=Path("data/staging"),
                        help="Test cases directory")

    args = parser.parse_args()

    if not args.dry_run and not args.force:
        print("⚠️  This will make API calls. Use --dry-run to test or --force to proceed.")
        print("   Estimated cost: ~$2-3 (520 predictions × 5 styles with gemini-2.0-flash)")
        sys.exit(1)

    # Load opus scoring results
    opus_results = load_opus_scoring_results(args.output)

    if not opus_results:
        print("ERROR: No opus-cli results found")
        sys.exit(1)

    print(f"\nWill judge {len(opus_results)} opus predictions with {len(args.styles)} styles")
    print(f"Styles: {args.styles}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Run judge for each style
    for style in args.styles:
        print(f"\n{'='*60}")
        print(f"JUDGE STYLE: {style}")
        print(f"{'='*60}")

        # Load existing results
        existing = load_existing_judge_results(args.output, style)
        print(f"Existing results: {len(existing)}")

        # Check if opus already judged
        opus_existing = [r for r in existing if "opus" in r.get("model", "").lower()]
        if opus_existing:
            print(f"  WARNING: {len(opus_existing)} opus results already exist, will be duplicated")

        # Configure judge - use gemini-2.0-flash to match existing results
        config = JudgeConfig(
            model="gemini-2.0-flash",
            parallel_workers=50,
            checkpoint_dir=args.output / "checkpoints",
            dry_run=args.dry_run,
            prompt_style=style,
            max_tokens=20000,
        )

        judge = LLMJudge(args.test_cases, config)

        print(f"Judging {len(opus_results)} opus predictions...")
        results = judge.judge_all(opus_results, progress_callback=progress_callback)
        print()  # newline after progress

        print(f"Judged {len(results)} results")

        # Compute stats for new results
        from experiments.llm_judge import compute_judge_stats
        stats = compute_judge_stats(results)

        # Show stats
        overall = stats.get("overall", {})
        print(f"  Overall: {overall.get('overall', {}).get('mean', 0):.3f}")

        # Save merged results
        save_merged_results(args.output, style, existing, results, stats)

    print(f"\n{'='*60}")
    print("✓ All judge styles complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
