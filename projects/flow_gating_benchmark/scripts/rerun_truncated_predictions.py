#!/usr/bin/env python3
"""
Rerun only truncated predictions with higher max_tokens.

Identifies predictions with unbalanced braces (truncated JSON) and reruns
them with increased token limit.

Usage:
    python scripts/rerun_truncated_predictions.py --dry-run
    python scripts/rerun_truncated_predictions.py --force --max-tokens 30000
    python scripts/rerun_truncated_predictions.py --models gemini-2.5-flash --force
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curation.omip_extractor import load_all_test_cases


def is_truncated(response: str) -> bool:
    """Check if response is truncated (unbalanced braces)."""
    open_brace = response.count('{')
    close_brace = response.count('}')
    return open_brace > close_brace


def call_gemini(prompt: str, api_key: str, model_name: str, max_tokens: int,
                max_retries: int = 5) -> tuple[str, int]:
    """Call Gemini API with exponential backoff retry."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )

            text = response.text if response.text else ""
            tokens = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0

            return text, tokens

        except Exception as e:
            last_error = e
            error_str = str(e)

            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                base_delay = 2 ** attempt
                jitter = random.uniform(0, 1)
                delay = base_delay + jitter

                if attempt < max_retries - 1:
                    print(f"\n  Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            else:
                raise

    raise last_error


def main():
    parser = argparse.ArgumentParser(description="Rerun truncated predictions with higher token limit")
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/multi_judge_run/predictions.json"),
                        help="Predictions file")
    parser.add_argument("--test-cases", type=Path, default=Path("data/staging"),
                        help="Test cases directory")
    parser.add_argument("--output", type=Path,
                        default=Path("results/multi_judge_run/predictions_rerun.json"),
                        help="Output file for rerun predictions")
    parser.add_argument("--max-tokens", type=int, default=30000,
                        help="Max output tokens (default: 30000)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific models (default: all truncated)")
    parser.add_argument("--workers", type=int, default=20,
                        help="Parallel workers (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with original predictions file")

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    # Load predictions
    print("Loading predictions...")
    with open(args.predictions) as f:
        predictions = json.load(f)

    print(f"Total predictions: {len(predictions)}")

    # Identify truncated predictions
    truncated = []
    complete = []
    for p in predictions:
        resp = p.get("raw_response", "")
        if is_truncated(resp):
            truncated.append(p)
        else:
            complete.append(p)

    print(f"Truncated: {len(truncated)}")
    print(f"Complete: {len(complete)}")

    # Filter by model if specified
    if args.models:
        truncated = [p for p in truncated if any(m in p["model"] for m in args.models)]
        print(f"After model filter: {len(truncated)}")

    if not truncated:
        print("No truncated predictions to rerun")
        return

    # Group by model
    by_model = defaultdict(list)
    for p in truncated:
        by_model[p["model"]].append(p)

    print("\n## Truncated by Model")
    for model, preds in sorted(by_model.items()):
        print(f"  {model}: {len(preds)}")

    # Estimate cost
    # Gemini 2.5 Pro: ~$0.00125/1K input + $0.005/1K output
    # With 30k output tokens, ~$0.15 per call
    estimated_cost = len(truncated) * 0.15
    print(f"\nEstimated cost: ~${estimated_cost:.2f}")

    if not args.dry_run and not args.force:
        print("\nUse --dry-run to test or --force to proceed")
        sys.exit(0)

    # Load test cases for prompts
    test_cases = load_all_test_cases(args.test_cases)
    tc_by_id = {tc.test_case_id: tc for tc in test_cases}

    # Build prompts module
    from experiments.prompts import build_prompt

    print(f"\nRerunning {len(truncated)} predictions with max_tokens={args.max_tokens}...")
    print(f"Workers: {args.workers}")

    rerun_results = []
    completed = 0

    def rerun_one(pred):
        """Rerun a single prediction."""
        tc = tc_by_id.get(pred["test_case_id"])
        if not tc:
            return {**pred, "error": "Test case not found", "rerun": True}

        # Parse condition
        cond = pred["condition"]
        parts = cond.split("_")
        # Format: model_context_strategy_rag e.g., gemini-2.5-flash_minimal_direct_none
        if len(parts) >= 4:
            context_level = parts[-3]
            prompt_strategy = parts[-2]
            rag_mode = parts[-1]
        else:
            context_level = "standard"
            prompt_strategy = "direct"
            rag_mode = "none"

        prompt = build_prompt(
            tc,
            template_name=prompt_strategy,
            context_level=context_level,
            rag_mode=rag_mode,
        )

        if args.dry_run:
            return {
                **pred,
                "raw_response": '{"name": "All Events", "children": []}',
                "tokens_used": 100,
                "rerun": True,
                "max_tokens": args.max_tokens,
            }

        try:
            # Use model name directly (Gemini SDK handles resolution)
            api_model = pred["model"]

            response, tokens = call_gemini(prompt, api_key, api_model, args.max_tokens)

            return {
                **pred,
                "raw_response": response,
                "tokens_used": tokens,
                "timestamp": datetime.now().isoformat(),
                "error": None,
                "rerun": True,
                "max_tokens": args.max_tokens,
            }
        except Exception as e:
            return {
                **pred,
                "error": str(e),
                "rerun": True,
                "rerun_attempted": True,
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(rerun_one, p): p for p in truncated}

        for future in as_completed(futures):
            result = future.result()
            rerun_results.append(result)
            completed += 1

            # Check if still truncated
            still_truncated = is_truncated(result.get("raw_response", ""))
            status = "TRUNC" if still_truncated else "OK" if not result.get("error") else "ERR"
            resp_len = len(result.get("raw_response", ""))

            pct = completed / len(truncated) * 100
            print(f"\r  Progress: {completed}/{len(truncated)} ({pct:.1f}%) | {status} | len={resp_len}", end="", flush=True)

    print()

    # Stats
    new_complete = [r for r in rerun_results if not is_truncated(r.get("raw_response", "")) and not r.get("error")]
    still_trunc = [r for r in rerun_results if is_truncated(r.get("raw_response", ""))]
    errors = [r for r in rerun_results if r.get("error")]

    print("\n## Results")
    print(f"  Fixed (now complete): {len(new_complete)}")
    print(f"  Still truncated: {len(still_trunc)}")
    print(f"  Errors: {len(errors)}")

    # Save rerun results
    with open(args.output, "w") as f:
        json.dump(rerun_results, f, indent=2)
    print(f"\nSaved rerun results to: {args.output}")

    # Optionally merge with original
    if args.merge:
        print("\nMerging with original predictions...")

        # Create lookup for rerun results
        rerun_keys = {(r["model"], r["test_case_id"], r["condition"], r.get("bootstrap_run", 0)): r
                      for r in rerun_results}

        merged = []
        replaced = 0
        for p in predictions:
            key = (p["model"], p["test_case_id"], p["condition"], p.get("bootstrap_run", 0))
            if key in rerun_keys:
                merged.append(rerun_keys[key])
                replaced += 1
            else:
                merged.append(p)

        # Backup original
        backup = args.predictions.with_suffix(".json.bak")
        args.predictions.rename(backup)
        print(f"  Backed up original to: {backup}")

        with open(args.predictions, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"  Merged {replaced} predictions into: {args.predictions}")

    print("\nDone!")


if __name__ == "__main__":
    main()
