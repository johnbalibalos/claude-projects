#!/usr/bin/env python3
"""
Run aggregated bootstrap predictions through the multi-judge pipeline using Anthropic API.

Supports parallel execution for faster processing.

Usage:
    ANTHROPIC_API_KEY=... python scripts/run_aggregated_multijudge_api.py --dry-run
    ANTHROPIC_API_KEY=... python scripts/run_aggregated_multijudge_api.py --force --model claude-sonnet-4
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.prediction_aggregator import (
    AggregatedPrediction,
    build_aggregated_judge_prompt,
    load_and_aggregate,
)

# Judge styles
JUDGE_STYLES = ["default", "validation", "qualitative", "orthogonal", "binary"]


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint if it exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed": set(), "results": {}}


def save_checkpoint(checkpoint_file: Path, checkpoint_data: dict):
    """Save checkpoint atomically."""
    # Convert set to list for JSON
    data_to_save = {
        "completed": list(checkpoint_data["completed"]),
        "results": checkpoint_data["results"],
    }
    tmp_file = checkpoint_file.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        json.dump(data_to_save, f)
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


def call_anthropic(prompt: str, model: str, api_key: str, dry_run: bool = False) -> tuple[str, int]:
    """Call Anthropic API."""
    if dry_run:
        return "[DRY RUN] Mock response\nMEDIAN_QUALITY: 7\nCONSISTENCY: 8\nWORST_CASE: 5\nBEST_CASE: 9\nFAILURE_MODES: none\nRELIABILITY: high\nSUMMARY: Mock summary", 100

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    text = getattr(message.content[0], "text", "") if message.content else ""
    tokens = message.usage.input_tokens + message.usage.output_tokens

    return text, tokens


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
    style: str,
    model: str,
    api_key: str,
    dry_run: bool = False,
) -> dict:
    """Judge one aggregated prediction."""
    gt = test_cases.get(agg.test_case_id, {})
    prompt = build_aggregated_judge_prompt(agg, gt)

    try:
        response, tokens = call_anthropic(prompt, model, api_key, dry_run)
        parsed = parse_aggregated_response(response)

        return {
            "test_case_id": agg.test_case_id,
            "model": agg.model,
            "condition": agg.condition,
            "n_bootstraps": agg.n_bootstraps,
            "parse_success_rate": agg.parse_success_rate,
            "consistency_score": agg.consistency_score,
            "unique_gate_sets": agg.unique_gate_sets,
            "judge_model": model,
            "judge_style": style,
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
            "judge_style": style,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def run_all_styles_parallel(
    aggregated: list[AggregatedPrediction],
    test_cases: dict[str, dict],
    styles: list[str],
    model: str,
    api_key: str,
    workers: int,
    dry_run: bool,
    checkpoint_file: Path,
    checkpoint_data: dict,
) -> dict[str, list[dict]]:
    """Run all styles in parallel."""
    # Build list of all (agg, style) pairs to process
    tasks = []
    for style in styles:
        for agg in aggregated:
            key = make_checkpoint_key(agg, style)
            if key not in checkpoint_data["completed"]:
                tasks.append((agg, style, key))

    total = len(tasks)
    skipped = len(aggregated) * len(styles) - total
    print(f"Total tasks: {len(aggregated) * len(styles)}, Already done: {skipped}, Remaining: {total}")

    if total == 0:
        print("All tasks already completed!")
        # Return results from checkpoint
        results_by_style = {style: [] for style in styles}
        for _key, result in checkpoint_data["results"].items():
            style = result.get("judge_style")
            if style in results_by_style:
                results_by_style[style].append(result)
        return results_by_style

    results_by_style = {style: [] for style in styles}
    completed = 0
    lock = __import__("threading").Lock()

    def process_task(task):
        agg, style, key = task
        result = judge_one(agg, test_cases, style, model, api_key, dry_run)
        return key, style, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}

        for future in as_completed(futures):
            key, style, result = future.result()

            with lock:
                results_by_style[style].append(result)
                checkpoint_data["completed"].add(key)
                checkpoint_data["results"][key] = result
                completed += 1

                # Save checkpoint after every result for resumability
                save_checkpoint(checkpoint_file, checkpoint_data)

                quality = result.get("median_quality", 0)
                error = "ERR" if result.get("error") else ""
                print(f"\r  Progress: {completed}/{total} ({100*completed/total:.1f}%) | {style}: {quality:.2f} {error}    ", end="", flush=True)

    print()

    # Final checkpoint save
    save_checkpoint(checkpoint_file, checkpoint_data)

    # Add any previously completed results
    for _key, result in checkpoint_data["results"].items():
        style = result.get("judge_style")
        if style in styles and result not in results_by_style[style]:
            results_by_style[style].append(result)

    return results_by_style


def main():
    parser = argparse.ArgumentParser(description="Run aggregated multi-judge with Anthropic API")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Model to use (default: claude-sonnet-4-20250514)")
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/multi_judge_run/predictions.json"),
                        help="Predictions file")
    parser.add_argument("--test-cases", type=Path, default=Path("data/verified"),
                        help="Test cases directory")
    parser.add_argument("--output", type=Path,
                        default=Path("results/multi_judge_run"),
                        help="Output directory")
    parser.add_argument("--models-filter", nargs="+", default=None,
                        help="Filter to specific prediction models")
    parser.add_argument("--styles", nargs="+", default=JUDGE_STYLES,
                        help=f"Judge styles (default: all). Options: {JUDGE_STYLES}")
    parser.add_argument("--workers", type=int, default=20,
                        help="Parallel workers (default: 20)")

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Load and aggregate predictions
    print("Loading and aggregating predictions...")
    aggregated = load_and_aggregate(args.predictions)

    # Filter by model if specified
    if args.models_filter:
        aggregated = [a for a in aggregated if any(m in a.model.lower() for m in args.models_filter)]

    print(f"Aggregated {len(aggregated)} unique (model, test_case, condition) combinations")

    # Cost estimate
    total_calls = len(aggregated) * len(args.styles)
    if "sonnet" in args.model.lower():
        cost_per_call = (4000 * 3.0 + 800 * 15.0) / 1_000_000
    elif "opus" in args.model.lower():
        cost_per_call = (4000 * 15.0 + 800 * 75.0) / 1_000_000
    else:
        cost_per_call = 0.02  # default estimate
    estimated_cost = total_calls * cost_per_call

    print(f"Total judge calls: {total_calls} ({len(aggregated)} × {len(args.styles)} styles)")
    print(f"Model: {args.model}")
    print(f"Estimated cost: ~${estimated_cost:.2f}")

    if not args.dry_run and not args.force:
        print("\n⚠️  Use --dry-run to test or --force to proceed.")
        sys.exit(1)

    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    print(f"Loaded {len(test_cases)} test cases")

    # Initialize checkpoint
    model_short = args.model.split("-")[1] if "-" in args.model else args.model
    checkpoint_file = args.output / f"checkpoint_{model_short}_judge.json"
    if args.resume and checkpoint_file.exists():
        raw_data = load_checkpoint(checkpoint_file)
        checkpoint_data = {
            "completed": set(raw_data.get("completed", [])),
            "results": raw_data.get("results", {}),
        }
        print(f"Resuming from checkpoint: {len(checkpoint_data['completed'])} items already done")
    else:
        checkpoint_data = {"completed": set(), "results": {}}
        if checkpoint_file.exists() and not args.resume:
            print("Warning: checkpoint exists but --resume not specified, starting fresh")

    print(f"\nRunning with {args.workers} workers")
    print(f"Styles: {args.styles}")
    print(f"Dry run: {args.dry_run}")

    # Run all styles in parallel
    print(f"\n{'='*60}")
    print("RUNNING ALL STYLES IN PARALLEL")
    print(f"{'='*60}")

    results_by_style = run_all_styles_parallel(
        aggregated, test_cases, args.styles, args.model, api_key or "",
        args.workers, args.dry_run, checkpoint_file, checkpoint_data
    )

    # Save results per style
    for style, results in results_by_style.items():
        if not results:
            continue

        # Stats
        valid = [r for r in results if r.get("median_quality") is not None and not r.get("error")]
        if valid:
            avg_quality = sum(r["median_quality"] for r in valid) / len(valid)
            avg_consistency = sum(r.get("consistency", 0) for r in valid) / len(valid)
            print(f"\n{style}: avg_quality={avg_quality:.3f}, avg_consistency={avg_consistency:.3f}, n={len(valid)}")

        errors = [r for r in results if r.get("error")]
        if errors:
            print(f"  Errors: {len(errors)}")

        # Save per-style
        output_file = args.output / f"aggregated_judge_{model_short}_{style}.json"
        with open(output_file, "w") as f:
            json.dump({
                "results": results,
                "judge_style": style,
                "judge_model": args.model,
                "n_aggregations": len(aggregated),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Saved to: {output_file}")

    # Save combined
    combined_output = args.output / f"aggregated_judge_{model_short}_all_styles.json"
    with open(combined_output, "w") as f:
        json.dump({
            "results_by_style": dict(results_by_style),
            "styles": args.styles,
            "judge_model": args.model,
            "n_aggregations": len(aggregated),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved combined results to: {combined_output}")

    print(f"\n{'='*60}")
    print("✓ All judge styles complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
