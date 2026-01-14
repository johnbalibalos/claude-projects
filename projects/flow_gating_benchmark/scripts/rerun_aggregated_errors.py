#!/usr/bin/env python3
"""
Rerun only failed/error aggregated judge results.

Loads existing results, identifies errors, and reruns just those with retry logic.

Usage:
    python scripts/rerun_aggregated_errors.py --style default --dry-run
    python scripts/rerun_aggregated_errors.py --style default --force
    python scripts/rerun_aggregated_errors.py --all-styles --force
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
    build_aggregated_judge_prompt,
    load_and_aggregate,
)

JUDGE_STYLES = ["default", "validation", "qualitative", "orthogonal", "binary"]


def call_gemini(prompt: str, api_key: str, model_name: str = "gemini-2.5-pro",
                max_retries: int = 5) -> tuple[str, int]:
    """Call Gemini API with exponential backoff retry."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    generation_config = types.GenerateContentConfig(
        max_output_tokens=20000,
        temperature=0.0,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )

            text = response.text if response.text else ""
            tokens = 0
            if response.usage_metadata:
                tokens = (response.usage_metadata.prompt_token_count or 0) + \
                         (response.usage_metadata.candidates_token_count or 0)

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
                    print(f"\n    Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
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


def load_test_cases(test_cases_dir: Path) -> dict[str, dict]:
    """Load all test cases into a dict keyed by test_case_id."""
    test_cases = {}
    for f in test_cases_dir.glob("*.json"):
        with open(f) as fp:
            tc = json.load(fp)
            tc_id = tc.get("test_case_id", f.stem)
            test_cases[tc_id] = tc
    return test_cases


def main():
    parser = argparse.ArgumentParser(description="Rerun failed aggregated judge results")
    parser.add_argument("--style", choices=JUDGE_STYLES, help="Single style to rerun")
    parser.add_argument("--all-styles", action="store_true", help="Rerun all styles with errors")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    parser.add_argument("--results-dir", type=Path, default=Path("results/multi_judge_run"),
                        help="Results directory")
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/multi_judge_run/predictions.json"),
                        help="Predictions file")
    parser.add_argument("--test-cases", type=Path, default=Path("data/verified"),
                        help="Test cases directory")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers (default: 10, lower to avoid rate limits)")
    parser.add_argument("--model", default="gemini-2.5-pro",
                        help="Judge model (default: gemini-2.5-pro)")

    args = parser.parse_args()

    if not args.style and not args.all_styles:
        print("ERROR: Must specify --style or --all-styles")
        sys.exit(1)

    # Get API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    # Determine which styles to process
    styles = JUDGE_STYLES if args.all_styles else [args.style]

    # Load aggregated predictions
    print("Loading aggregated predictions...")
    aggregated = load_and_aggregate(args.predictions)
    agg_by_key = {(a.model, a.test_case_id, a.condition): a for a in aggregated}

    # Load test cases
    test_cases = load_test_cases(args.test_cases)

    for style in styles:
        # Load existing results for this style
        results_file = args.results_dir / f"aggregated_judge_{style}.json"
        if not results_file.exists():
            print(f"\nNo results file for {style}, skipping")
            continue

        with open(results_file) as f:
            data = json.load(f)

        results = data["results"]
        errors = [r for r in results if r.get("error")]
        successful = [r for r in results if not r.get("error")]

        print(f"\n{'='*60}")
        print(f"STYLE: {style}")
        print(f"{'='*60}")
        print(f"Total: {len(results)}, Errors: {len(errors)}, Successful: {len(successful)}")

        if not errors:
            print("No errors to rerun")
            continue

        if not args.dry_run and not args.force:
            print(f"\n⚠️  Will rerun {len(errors)} failed results. Use --force to proceed.")
            continue

        # Rerun errors
        print(f"\nRerunning {len(errors)} errors with {args.workers} workers...")
        rerun_results = []
        completed = 0

        def make_rerun_fn(current_style):
            """Factory to create rerun function with bound style."""
            def _rerun_one(error_result):
                key = (error_result["model"], error_result["test_case_id"], error_result["condition"])
                agg = agg_by_key.get(key)
                if not agg:
                    return error_result  # Can't find aggregation, return original error

                gt = test_cases.get(agg.test_case_id, {})
                prompt = build_aggregated_judge_prompt(agg, gt)

                if args.dry_run:
                    return {
                        **error_result,
                        "error": None,
                        "median_quality": 0.5,
                        "consistency": 0.5,
                        "rerun": True,
                    }

                try:
                    response, tokens = call_gemini(prompt, api_key, args.model)
                    parsed = parse_aggregated_response(response)

                    return {
                        "test_case_id": agg.test_case_id,
                        "model": agg.model,
                        "condition": agg.condition,
                        "n_bootstraps": agg.n_bootstraps,
                        "parse_success_rate": agg.parse_success_rate,
                        "consistency_score": agg.consistency_score,
                        "unique_gate_sets": agg.unique_gate_sets,
                        "judge_model": args.model,
                        "judge_style": current_style,
                        "tokens_used": tokens,
                        "timestamp": datetime.now().isoformat(),
                        "error": None,
                        "rerun": True,
                        **parsed,
                    }
                except Exception as err:
                    return {
                        **error_result,
                        "error": str(err),
                        "rerun_attempted": True,
                    }
            return _rerun_one

        rerun_fn = make_rerun_fn(style)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(rerun_fn, e): e for e in errors}

            for future in as_completed(futures):
                result = future.result()
                rerun_results.append(result)
                completed += 1

                pct = completed / len(errors) * 100
                quality = result.get("median_quality", 0)
                status = "OK" if not result.get("error") else "ERR"
                print(f"\r  Progress: {completed}/{len(errors)} ({pct:.1f}%) | {status} | q={quality:.2f}", end="", flush=True)

        print()

        # Merge results
        new_successes = [r for r in rerun_results if not r.get("error")]
        still_failed = [r for r in rerun_results if r.get("error")]

        print(f"  Rerun complete: {len(new_successes)} fixed, {len(still_failed)} still failed")

        # Replace old error results with new results
        error_keys = {(r["model"], r["test_case_id"], r["condition"]) for r in errors}
        merged = [r for r in results if (r["model"], r["test_case_id"], r["condition"]) not in error_keys]
        merged.extend(rerun_results)

        # Backup and save
        backup = results_file.with_suffix(".json.pre_rerun")
        results_file.rename(backup)
        print(f"  Backed up to: {backup.name}")

        with open(results_file, "w") as f:
            json.dump({
                "results": merged,
                "judge_style": style,
                "judge_model": args.model,
                "n_aggregations": len(aggregated),
                "timestamp": datetime.now().isoformat(),
                "rerun_note": f"Reran {len(errors)} errors, {len(new_successes)} fixed",
            }, f, indent=2)
        print(f"  Saved to: {results_file.name}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
