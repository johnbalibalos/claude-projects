#!/usr/bin/env python3
"""
Run aggregated bootstrap predictions through the multi-judge pipeline.

Uses aggregated n=10 bootstrap data with all 5 judge styles via gemini-2.0-flash.

Usage:
    GOOGLE_API_KEY=... python scripts/run_aggregated_multijudge.py --dry-run
    GOOGLE_API_KEY=... python scripts/run_aggregated_multijudge.py --force
"""

import argparse
import json
import os
import random
import sys
import time
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

# Judge styles - same as multi-judge pipeline
JUDGE_STYLES = ["default", "validation", "qualitative", "orthogonal", "binary"]


def load_test_cases(test_cases_dir: Path) -> dict[str, dict]:
    """Load all test cases into a dict keyed by test_case_id."""
    test_cases = {}
    for f in test_cases_dir.glob("*.json"):
        with open(f) as fp:
            tc = json.load(fp)
            tc_id = tc.get("test_case_id", f.stem)
            test_cases[tc_id] = tc
    return test_cases


def call_gemini(prompt: str, api_key: str, dry_run: bool = False, max_retries: int = 5) -> tuple[str, int]:
    """Call Gemini 2.0 Flash API with exponential backoff retry."""
    if dry_run:
        return "[DRY RUN] Mock response\nMEDIAN_QUALITY: 7\nCONSISTENCY: 8\nWORST_CASE: 5\nBEST_CASE: 9\nFAILURE_MODES: none\nRELIABILITY: high Good model\nSUMMARY: Mock summary", 100

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 20000,
                    "temperature": 0.0,
                },
            )

            text = response.text if response.text else ""
            tokens = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0

            return text, tokens

        except Exception as e:
            last_error = e
            error_str = str(e)

            # Check if it's a rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                # Exponential backoff with jitter
                base_delay = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                jitter = random.uniform(0, 1)
                delay = base_delay + jitter

                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
            else:
                # Non-rate-limit error, don't retry
                raise

    # All retries exhausted
    raise last_error


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
    api_key: str,
    dry_run: bool = False,
) -> dict:
    """Judge one aggregated prediction."""
    gt = test_cases.get(agg.test_case_id, {})
    prompt = build_aggregated_judge_prompt(agg, gt)

    try:
        response, tokens = call_gemini(prompt, api_key, dry_run)
        parsed = parse_aggregated_response(response)

        return {
            "test_case_id": agg.test_case_id,
            "model": agg.model,
            "condition": agg.condition,
            "n_bootstraps": agg.n_bootstraps,
            "parse_success_rate": agg.parse_success_rate,
            "consistency_score": agg.consistency_score,
            "unique_gate_sets": agg.unique_gate_sets,
            "judge_model": "gemini-2.0-flash",
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
    api_key: str,
    workers: int,
    dry_run: bool,
) -> list[dict]:
    """Run judge for one style."""
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(judge_one, agg, test_cases, api_key, dry_run): agg
            for agg in aggregated
        }

        for future in as_completed(futures):
            result = future.result()
            result["judge_style"] = style
            results.append(result)
            completed += 1

            pct = completed / len(aggregated) * 100
            quality = result.get("median_quality", 0)
            print(f"\r  Progress: {completed}/{len(aggregated)} ({pct:.1f}%) | last: {quality:.2f}", end="", flush=True)

    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run aggregated multi-judge with gemini-2.0-flash")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation")
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
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers (default: 10)")

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    # Load and aggregate predictions
    print("Loading and aggregating predictions...")
    aggregated = load_and_aggregate(args.predictions)

    # Filter by model if specified
    if args.models:
        aggregated = [a for a in aggregated if any(m in a.model.lower() for m in args.models)]

    print(f"Aggregated {len(aggregated)} unique (model, test_case, condition) combinations")

    # Estimate cost: gemini-2.0-flash ~$0.02 per call
    total_calls = len(aggregated) * len(args.styles)
    estimated_cost = total_calls * 0.02
    print(f"Total judge calls: {total_calls} ({len(aggregated)} × {len(args.styles)} styles)")
    print(f"Estimated cost: ~${estimated_cost:.2f}")

    if not args.dry_run and not args.force:
        print("\n⚠️  Use --dry-run to test or --force to proceed.")
        sys.exit(1)

    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    print(f"Loaded {len(test_cases)} test cases")

    print(f"\nRunning with {args.workers} workers, gemini-2.0-flash")
    print(f"Styles: {args.styles}")
    print(f"Dry run: {args.dry_run}")

    # Run each style
    all_results = {}
    for style in args.styles:
        print(f"\n{'='*60}")
        print(f"JUDGE STYLE: {style}")
        print(f"{'='*60}")

        results = run_judge_style(
            aggregated, test_cases, style, api_key, args.workers, args.dry_run
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

        # Save per-style (flash version)
        output_file = args.output / f"aggregated_judge_flash_{style}.json"
        with open(output_file, "w") as f:
            json.dump({
                "results": results,
                "judge_style": style,
                "judge_model": "gemini-2.0-flash",
                "n_aggregations": len(aggregated),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Saved to: {output_file}")

    # Save combined (flash version)
    combined_output = args.output / "aggregated_judge_flash_all_styles.json"
    with open(combined_output, "w") as f:
        json.dump({
            "results_by_style": all_results,
            "styles": args.styles,
            "judge_model": "gemini-2.0-flash",
            "n_aggregations": len(aggregated),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved combined results to: {combined_output}")

    print(f"\n{'='*60}")
    print("✓ All judge styles complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
