#!/usr/bin/env python3
"""
Run aggregated bootstrap judge using gemini-2.5-pro.

Aggregates n=10 bootstrap predictions per (model, test_case, condition) into
single prompts, then evaluates with gemini-2.5-pro for cross-run consistency analysis.

Usage:
    python scripts/run_aggregated_judge.py --dry-run  # Test without API calls
    python scripts/run_aggregated_judge.py --force    # Run with API calls
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
    AGGREGATED_JUDGE_CONFIG,
    AggregatedPrediction,
    build_aggregated_judge_prompt,
    load_and_aggregate,
)


def load_test_cases(test_cases_dir: Path) -> dict[str, dict]:
    """Load all test cases into a dict keyed by test_case_id."""
    test_cases = {}
    for f in test_cases_dir.glob("*.json"):
        with open(f) as fp:
            tc = json.load(fp)
            tc_id = tc.get("test_case_id", f.stem)
            test_cases[tc_id] = tc
    return test_cases


def call_gemini(prompt: str, dry_run: bool = False) -> tuple[str, int]:
    """Call Gemini 2.5 Pro API."""
    if dry_run:
        return "[DRY RUN] Mock response for aggregated judge", 0

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)

    generation_config = types.GenerateContentConfig(
        max_output_tokens=AGGREGATED_JUDGE_CONFIG["max_tokens"],
        temperature=AGGREGATED_JUDGE_CONFIG["temperature"],
    )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=generation_config,
    )

    text = response.text if response.text else ""
    tokens = 0
    if response.usage_metadata:
        tokens = (response.usage_metadata.prompt_token_count or 0) + \
                 (response.usage_metadata.candidates_token_count or 0)

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
    dry_run: bool = False,
) -> dict:
    """Judge one aggregated prediction."""
    gt = test_cases.get(agg.test_case_id, {})
    prompt = build_aggregated_judge_prompt(agg, gt)

    try:
        response, tokens = call_gemini(prompt, dry_run)
        parsed = parse_aggregated_response(response)

        return {
            "test_case_id": agg.test_case_id,
            "model": agg.model,
            "condition": agg.condition,
            "n_bootstraps": agg.n_bootstraps,
            "parse_success_rate": agg.parse_success_rate,
            "consistency_score": agg.consistency_score,
            "unique_gate_sets": agg.unique_gate_sets,
            "judge_model": "gemini-2.5-pro",
            "tokens_used": tokens,
            "timestamp": datetime.now().isoformat(),
            **parsed,
        }
    except Exception as e:
        return {
            "test_case_id": agg.test_case_id,
            "model": agg.model,
            "condition": agg.condition,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="Run aggregated bootstrap judge")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation")
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/multi_judge_run/predictions.json"),
                        help="Predictions file")
    parser.add_argument("--test-cases", type=Path, default=Path("data/staging"),
                        help="Test cases directory")
    parser.add_argument("--output", type=Path,
                        default=Path("results/multi_judge_run/aggregated_judge_results.json"),
                        help="Output file")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific models (default: all)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers (default: 10)")

    args = parser.parse_args()

    # Load and aggregate predictions
    print("Loading and aggregating predictions...")
    aggregated = load_and_aggregate(args.predictions)

    # Filter by model if specified
    if args.models:
        aggregated = [a for a in aggregated if any(m in a.model.lower() for m in args.models)]

    print(f"Aggregated {len(aggregated)} unique (model, test_case, condition) combinations")

    # Estimate cost
    # gemini-2.5-pro: ~$0.01-0.02 per call for these prompts
    estimated_cost = len(aggregated) * 0.015
    print(f"Estimated cost: ~${estimated_cost:.2f}")

    if not args.dry_run and not args.force:
        print("\n⚠️  This will make API calls. Use --dry-run to test or --force to proceed.")
        sys.exit(1)

    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    print(f"Loaded {len(test_cases)} test cases")

    # Run judge
    print(f"\nRunning aggregated judge with {args.workers} workers...")
    print(f"Dry run: {args.dry_run}")

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_one, agg, test_cases, args.dry_run): agg
            for agg in aggregated
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress
            pct = completed / len(aggregated) * 100
            quality = result.get("median_quality", 0)
            print(f"\r  Progress: {completed}/{len(aggregated)} ({pct:.1f}%) | last: {quality:.2f}", end="", flush=True)

    print()  # newline

    # Stats
    print("\nResults summary:")
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    for model, model_results in sorted(by_model.items()):
        valid = [r for r in model_results if "median_quality" in r]
        if valid:
            avg_quality = sum(r["median_quality"] for r in valid) / len(valid)
            avg_consistency = sum(r["consistency"] for r in valid) / len(valid)
            print(f"  {model}:")
            print(f"    median_quality: {avg_quality:.3f}")
            print(f"    consistency: {avg_consistency:.3f}")
            print(f"    n: {len(valid)}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "results": results,
            "n_aggregations": len(aggregated),
            "judge_model": "gemini-2.5-pro",
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
