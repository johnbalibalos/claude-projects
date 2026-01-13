#!/usr/bin/env python3
"""
Run aggregated bootstrap predictions through the multi-judge pipeline using Claude Opus CLI.

Uses aggregated n=10 bootstrap data with all 5 judge styles via claude-opus-cli.
Runs sequentially due to CLI rate limits. Supports checkpointing for resume.

Usage:
    python scripts/run_aggregated_multijudge_opus.py --dry-run
    python scripts/run_aggregated_multijudge_opus.py --force
    python scripts/run_aggregated_multijudge_opus.py --force --resume  # Resume from checkpoint
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.llm_client import ClaudeCLIClient
from utils.prediction_aggregator import (
    AggregatedPrediction,
    build_aggregated_judge_prompt,
    load_and_aggregate,
)

# Judge styles - same as multi-judge pipeline
JUDGE_STYLES = ["default", "validation", "qualitative", "orthogonal", "binary"]


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint if it exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed": {}, "results": {}}


def save_checkpoint(checkpoint_file: Path, checkpoint_data: dict):
    """Save checkpoint atomically."""
    tmp_file = checkpoint_file.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        json.dump(checkpoint_data, f)
    tmp_file.rename(checkpoint_file)


def make_checkpoint_key(agg: AggregatedPrediction, style: str) -> str:
    """Create unique key for checkpointing."""
    return f"{agg.model}|{agg.test_case_id}|{agg.condition}|{style}"


def load_test_cases(test_cases_dir: Path) -> dict[str, dict]:
    """Load all test cases into a dict keyed by test_case_id."""
    test_cases = {}
    for f in test_cases_dir.glob("*.json"):
        with open(f) as fp:
            tc = json.load(fp)
            tc_id = tc.get("test_case_id", f.stem)
            test_cases[tc_id] = tc
    return test_cases


def parse_aggregated_response(response: str) -> dict:
    """Parse the aggregated judge response."""
    result = {
        "median_quality": 0.0,
        "consistency": 0.0,
        "worst_case": 0.0,
        "best_case": 0.0,
        "failure_modes": "",
        "reliability": "",
        "summary": "",
        "raw_response": response,
    }

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("MEDIAN_QUALITY:"):
            try:
                val = line.split(":", 1)[1].strip().split()[0]
                result["median_quality"] = float(val) / 10.0
            except (ValueError, IndexError):
                pass
        elif line.startswith("CONSISTENCY:"):
            try:
                val = line.split(":", 1)[1].strip().split()[0]
                result["consistency"] = float(val) / 10.0
            except (ValueError, IndexError):
                pass
        elif line.startswith("WORST_CASE:"):
            try:
                val = line.split(":", 1)[1].strip().split()[0]
                result["worst_case"] = float(val) / 10.0
            except (ValueError, IndexError):
                pass
        elif line.startswith("BEST_CASE:"):
            try:
                val = line.split(":", 1)[1].strip().split()[0]
                result["best_case"] = float(val) / 10.0
            except (ValueError, IndexError):
                pass
        elif line.startswith("FAILURE_MODES:"):
            result["failure_modes"] = line.split(":", 1)[1].strip()
        elif line.startswith("RELIABILITY:"):
            result["reliability"] = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()

    return result


def judge_one(
    agg: AggregatedPrediction,
    test_cases: dict[str, dict],
    client: ClaudeCLIClient,
    dry_run: bool = False,
) -> dict:
    """Judge one aggregated prediction."""
    gt = test_cases.get(agg.test_case_id, {})
    prompt = build_aggregated_judge_prompt(agg, gt)

    try:
        if dry_run:
            response = "[DRY RUN] Mock response\nMEDIAN_QUALITY: 7\nCONSISTENCY: 8\nWORST_CASE: 5\nBEST_CASE: 9\nFAILURE_MODES: none\nRELIABILITY: high Good model\nSUMMARY: Mock summary"
            tokens = 100
        else:
            result = client.call(prompt)
            response = result.content
            tokens = result.tokens_used

        parsed = parse_aggregated_response(response)

        return {
            "test_case_id": agg.test_case_id,
            "model": agg.model,
            "condition": agg.condition,
            "n_bootstraps": agg.n_bootstraps,
            "parse_success_rate": agg.parse_success_rate,
            "consistency_score": agg.consistency_score,
            "unique_gate_sets": agg.unique_gate_sets,
            "judge_model": "claude-opus-4-cli",
            "tokens_used": tokens,
            "timestamp": datetime.now().isoformat(),
            "error": None,
            **parsed,
        }
    except Exception as e:
        return {
            "test_case_id": agg.test_case_id,
            "model": agg.model,
            "condition": agg.condition,
            "n_bootstraps": agg.n_bootstraps,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def run_judge_style(
    aggregated: list[AggregatedPrediction],
    test_cases: dict[str, dict],
    style: str,
    client: ClaudeCLIClient,
    dry_run: bool,
    checkpoint_file: Path,
    checkpoint_data: dict,
) -> list[dict]:
    """Run judge for one style (sequential for CLI rate limits) with checkpointing."""
    results = []
    skipped = 0

    for i, agg in enumerate(aggregated):
        key = make_checkpoint_key(agg, style)

        # Check if already completed
        if key in checkpoint_data["completed"]:
            results.append(checkpoint_data["results"][key])
            skipped += 1
            continue

        result = judge_one(agg, test_cases, client, dry_run)
        result["judge_style"] = style
        results.append(result)

        # Save to checkpoint
        checkpoint_data["completed"][key] = True
        checkpoint_data["results"][key] = result
        save_checkpoint(checkpoint_file, checkpoint_data)

        done = i + 1
        pct = done / len(aggregated) * 100
        quality = result.get("median_quality", 0)
        print(f"\r  Progress: {done}/{len(aggregated)} ({pct:.1f}%) | last: {quality:.2f} | skipped: {skipped}", end="", flush=True)

    print()
    if skipped > 0:
        print(f"  Resumed: skipped {skipped} already-completed items")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run aggregated multi-judge with claude-opus-cli")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/multi_judge_run/predictions.json"),
                        help="Predictions file")
    parser.add_argument("--test-cases", type=Path, default=Path("data/staging"),
                        help="Test cases directory")
    parser.add_argument("--output", type=Path,
                        default=Path("results/multi_judge_run"),
                        help="Output directory")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific models")
    parser.add_argument("--styles", nargs="+", default=JUDGE_STYLES,
                        help=f"Judge styles (default: all). Options: {JUDGE_STYLES}")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between CLI calls in seconds (default: 2.0)")

    args = parser.parse_args()

    # Load and aggregate predictions
    print("Loading and aggregating predictions...")
    aggregated = load_and_aggregate(args.predictions)

    # Filter by model if specified
    if args.models:
        aggregated = [a for a in aggregated if any(m in a.model.lower() for m in args.models)]

    print(f"Aggregated {len(aggregated)} unique (model, test_case, condition) combinations")

    # Estimate time: ~5-10 seconds per call with CLI
    total_calls = len(aggregated) * len(args.styles)
    estimated_time_min = total_calls * (args.delay + 5) / 60  # delay + response time
    estimated_time_max = total_calls * (args.delay + 15) / 60
    print(f"Total judge calls: {total_calls} ({len(aggregated)} × {len(args.styles)} styles)")
    print(f"Estimated time: {estimated_time_min:.0f}-{estimated_time_max:.0f} minutes")

    if not args.dry_run and not args.force:
        print("\n⚠️  Use --dry-run to test or --force to proceed.")
        print("   Note: opus-cli uses Claude Max subscription (no API cost)")
        sys.exit(1)

    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    print(f"Loaded {len(test_cases)} test cases")

    # Initialize client
    if not args.dry_run:
        client = ClaudeCLIClient(model="claude-opus-4-20250514", delay_seconds=args.delay)
    else:
        client = None

    # Initialize checkpoint
    checkpoint_file = args.output / "opus_judge_checkpoint.json"
    if args.resume and checkpoint_file.exists():
        checkpoint_data = load_checkpoint(checkpoint_file)
        print(f"Resuming from checkpoint: {len(checkpoint_data['completed'])} items already done")
    else:
        checkpoint_data = {"completed": {}, "results": {}}
        if checkpoint_file.exists() and not args.resume:
            print("Warning: checkpoint exists but --resume not specified, starting fresh")

    print(f"\nRunning sequentially with claude-opus-cli (delay: {args.delay}s)")
    print(f"Styles: {args.styles}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume: {args.resume}")

    # Run each style
    all_results = {}
    for style in args.styles:
        print(f"\n{'='*60}")
        print(f"JUDGE STYLE: {style}")
        print(f"{'='*60}")

        results = run_judge_style(
            aggregated, test_cases, style, client, args.dry_run,
            checkpoint_file, checkpoint_data
        )

        # Stats
        valid = [r for r in results if r.get("median_quality") is not None and not r.get("error")]
        if valid:
            avg_quality = sum(r["median_quality"] for r in valid) / len(valid)
            avg_consistency = sum(r["consistency"] for r in valid) / len(valid)
            print(f"  Avg median_quality: {avg_quality:.3f}")
            print(f"  Avg consistency: {avg_consistency:.3f}")

        errors = [r for r in results if r.get("error")]
        if errors:
            print(f"  Errors: {len(errors)}")

        all_results[style] = results

        # Save per-style
        output_file = args.output / f"aggregated_judge_opus_{style}.json"
        with open(output_file, "w") as f:
            json.dump({
                "results": results,
                "judge_style": style,
                "judge_model": "claude-opus-4-cli",
                "n_aggregations": len(aggregated),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Saved to: {output_file}")

    # Save combined
    combined_output = args.output / "aggregated_judge_opus_all_styles.json"
    with open(combined_output, "w") as f:
        json.dump({
            "results_by_style": all_results,
            "styles": args.styles,
            "judge_model": "claude-opus-4-cli",
            "n_aggregations": len(aggregated),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved combined results to: {combined_output}")

    print(f"\n{'='*60}")
    print("✓ All judge styles complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
