#!/usr/bin/env python3
"""
Rerun blocked predictions with higher max_tokens.

Usage:
    python scripts/rerun_blocked.py --input results/gemini_run --max-tokens 10000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from curation.omip_extractor import load_all_test_cases  # noqa: E402
from experiments.conditions import get_all_conditions  # noqa: E402
from experiments.llm_client import create_client  # noqa: E402
from experiments.prompts import build_prompt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Rerun blocked predictions")
    parser.add_argument("--input", type=Path, required=True, help="Input results directory")
    parser.add_argument("--test-cases", type=Path, default=PROJECT_ROOT / "data" / "staging")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max tokens for rerun")
    # NOTE: dry-run disabled to prevent accidental mock runs
    # parser.add_argument("--dry-run", action="store_true", help="Don't actually call API")
    args = parser.parse_args()
    args.dry_run = False  # Force real API calls

    # Load blocked keys
    blocked_keys_file = args.input / "blocked_keys.json"
    if not blocked_keys_file.exists():
        print(f"ERROR: No blocked_keys.json found in {args.input}")
        return

    with open(blocked_keys_file) as f:
        blocked_keys = [tuple(k) for k in json.load(f)]

    print(f"Found {len(blocked_keys)} blocked predictions to rerun")
    print(f"Using max_tokens={args.max_tokens}")

    # Load predictions
    predictions_file = args.input / "predictions.json"
    with open(predictions_file) as f:
        all_predictions = json.load(f)

    # Load test cases
    test_cases = load_all_test_cases(args.test_cases)
    test_case_map = {tc.test_case_id: tc for tc in test_cases}

    # Generate conditions
    conditions = get_all_conditions(
        models=["gemini-2.5-pro"],
        context_levels=["minimal", "standard"],
        prompt_strategies=["direct", "cot"],
    )
    condition_map = {c.name: c for c in conditions}

    # Create client
    client = create_client("gemini-2.5-pro", dry_run=args.dry_run)

    # Rerun blocked predictions
    new_predictions = []
    for i, key in enumerate(blocked_keys):
        bootstrap_run, test_case_id, model, condition_name = key

        test_case = test_case_map.get(test_case_id)
        condition = condition_map.get(condition_name)

        if not test_case or not condition:
            print(f"  [{i+1}/{len(blocked_keys)}] SKIP: {test_case_id} - missing test case or condition")
            continue

        prompt = build_prompt(
            test_case,
            template_name=condition.prompt_strategy,
            context_level=condition.context_level,
            rag_mode=condition.rag_mode,
        )

        print(f"  [{i+1}/{len(blocked_keys)}] {test_case_id} | {condition_name} | run {bootstrap_run}")

        try:
            response = client.call(prompt, max_tokens=args.max_tokens)

            new_pred = {
                "test_case_id": test_case_id,
                "model": model,
                "condition": condition_name,
                "bootstrap_run": bootstrap_run,
                "raw_response": response.content,
                "prompt": prompt[:500],
                "tokens_used": response.tokens_used,
                "timestamp": datetime.now().isoformat(),
                "error": None,
                "run_id": "",
                "rerun_max_tokens": args.max_tokens,
            }

            if '[BLOCKED' in response.content or '[EMPTY_RESPONSE' in response.content:
                print(f"    Still blocked: {response.content[:50]}")
            else:
                print(f"    Success: {len(response.content)} chars")

            new_predictions.append(new_pred)

        except Exception as e:
            print(f"    ERROR: {e}")
            new_predictions.append({
                "test_case_id": test_case_id,
                "model": model,
                "condition": condition_name,
                "bootstrap_run": bootstrap_run,
                "raw_response": "",
                "prompt": prompt[:500],
                "tokens_used": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "run_id": "",
                "rerun_max_tokens": args.max_tokens,
            })

    # Save rerun results
    rerun_file = args.input / "rerun_predictions.json"
    with open(rerun_file, "w") as f:
        json.dump(new_predictions, f, indent=2)
    print(f"\nSaved {len(new_predictions)} rerun predictions to {rerun_file}")

    # Merge with original
    blocked_key_set = set(blocked_keys)
    merged = [p for p in all_predictions
              if (p['bootstrap_run'], p['test_case_id'], p['model'], p['condition']) not in blocked_key_set]
    merged.extend(new_predictions)

    merged_file = args.input / "predictions_merged.json"
    with open(merged_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged predictions to {merged_file}")

    # Stats
    still_blocked = len([p for p in new_predictions
                        if '[BLOCKED' in p.get('raw_response', '') or '[EMPTY_RESPONSE' in p.get('raw_response', '')])
    print(f"\nResults: {len(new_predictions) - still_blocked} recovered, {still_blocked} still blocked")


if __name__ == "__main__":
    main()
