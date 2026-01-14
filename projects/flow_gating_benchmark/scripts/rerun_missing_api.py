#!/usr/bin/env python3
"""
Rerun missing haiku and opus predictions via API.

Identifies which (condition, test_case, bootstrap) combinations are missing
or errored, and reruns only those via API.
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from curation.omip_extractor import load_all_test_cases  # noqa: E402
from experiments.conditions import get_all_conditions  # noqa: E402
from experiments.llm_client import AnthropicClient  # noqa: E402
from experiments.prompts import build_prompt  # noqa: E402


def normalize_condition(condition: str) -> str:
    """Normalize condition name to treat CLI and API as equivalent."""
    # claude-opus-cli_minimal_direct_none -> claude-opus_minimal_direct_none
    return condition.replace('-cli_', '_')


def load_existing_predictions(path: Path) -> dict:
    """Load existing predictions and identify successful ones."""
    with open(path) as f:
        predictions = json.load(f)

    # Group by (normalized_condition, test_case, bootstrap)
    successful = defaultdict(list)
    for p in predictions:
        # Check if this is a successful prediction (not an error)
        is_error = (
            not p.get('raw_response') or
            '[ERROR' in p.get('raw_response', '') or
            '[BLOCKED' in p.get('raw_response', '') or
            p.get('error')
        )

        # Normalize condition to treat CLI and API as equivalent
        norm_cond = normalize_condition(p['condition'])
        key = (norm_cond, p['test_case_id'], p['bootstrap_run'])
        if not is_error:
            successful[key].append(p)

    return successful


def get_missing_keys(
    successful: dict,
    conditions: list,
    test_case_ids: list[str],
    n_bootstrap: int,
) -> list[tuple]:
    """Find which keys are missing for a model."""
    missing = []
    for condition in conditions:
        # Use normalized condition name for matching
        cond_name = normalize_condition(condition.name)
        for tc_id in test_case_ids:
            # Bootstrap values in predictions are 1-indexed
            for bootstrap in range(1, n_bootstrap + 1):
                key = (cond_name, tc_id, bootstrap)
                if key not in successful:
                    missing.append((condition, tc_id, bootstrap))
    return missing


def run_prediction(
    client,
    condition,
    test_case,  # TestCase object
    bootstrap: int,
    run_id: str,
) -> dict:
    """Run a single prediction."""
    # build_prompt expects: test_case, template_name, context_level, reference
    prompt = build_prompt(
        test_case,
        template_name=condition.prompt_strategy,
        context_level=condition.context_level,
        reference=condition.reference,
    )

    try:
        response = client.call(prompt, max_tokens=6000, temperature=0.7)
        return {
            'test_case_id': test_case.test_case_id,
            'model': client.model_id,
            'condition': condition.name,
            'bootstrap_run': bootstrap,
            'raw_response': response.content,
            'tokens_used': response.tokens_used,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt[:500] + '...' if len(prompt) > 500 else prompt,
            'error': None,
            'run_id': run_id,
        }
    except Exception as e:
        return {
            'test_case_id': test_case.test_case_id,
            'model': client.model_id,
            'condition': condition.name,
            'bootstrap_run': bootstrap,
            'raw_response': f'[ERROR: {str(e)}]',
            'tokens_used': 0,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt[:500] + '...' if len(prompt) > 500 else prompt,
            'error': str(e),
            'run_id': run_id,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True,
                        help='Path to existing predictions.json')
    parser.add_argument('--test-cases', type=Path, default=PROJECT_ROOT / 'data/verified',
                        help='Path to test cases directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output path for new predictions')
    parser.add_argument('--models', nargs='+', default=['claude-haiku', 'claude-opus'],
                        help='Models to rerun (haiku, opus)')
    parser.add_argument('--n-bootstrap', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-workers', type=int, default=5)
    args = parser.parse_args()

    # Load existing predictions
    print(f"Loading existing predictions from {args.predictions}")
    successful = load_existing_predictions(args.predictions)
    print(f"Found {len(successful)} successful prediction keys")

    # Load test cases (keep as TestCase objects, not dicts)
    test_cases = load_all_test_cases(args.test_cases)
    test_case_map = {tc.test_case_id: tc for tc in test_cases}
    test_case_ids = list(test_case_map.keys())
    print(f"Loaded {len(test_cases)} test cases")

    print(f"Test case IDs: {test_case_ids}")

    # Model mapping
    model_map = {
        'claude-haiku': 'claude-3-5-haiku-20241022',
        'claude-opus': 'claude-opus-4-20250514',
    }

    run_id = f"rerun_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_new_predictions = []

    for model_short in args.models:
        model_full = model_map.get(model_short, model_short)

        # Regenerate conditions with this model
        conditions = get_all_conditions(
            models=[model_short],
            context_levels=['minimal', 'standard', 'rich'],
            prompt_strategies=['direct', 'cot'],
            references=['none', 'hipc'],
        )

        # Find missing keys
        missing = get_missing_keys(
            successful, conditions, test_case_ids, args.n_bootstrap
        )

        print(f"\n{model_full}: {len(missing)} missing predictions")

        if args.dry_run:
            print(f"  [DRY RUN] Would run {len(missing)} predictions")
            continue

        if not missing:
            continue

        # Create API client
        client = AnthropicClient(model=model_full)
        print(f"  Using Anthropic API for {model_full}")

        # Run predictions in parallel
        new_predictions = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for condition, tc_id, bootstrap in missing:
                tc = test_case_map[tc_id]
                future = executor.submit(
                    run_prediction, client, condition, tc, bootstrap, run_id
                )
                futures[future] = (condition.name, tc_id, bootstrap)

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                new_predictions.append(result)

                if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                    pct = (i + 1) / len(futures) * 100
                    errors = sum(1 for p in new_predictions if p.get('error'))
                    print(f"  [{i+1}/{len(futures)}] ({pct:.0f}%) - {errors} errors")

        all_new_predictions.extend(new_predictions)
        print(f"  Completed {len(new_predictions)} predictions for {model_full}")

    if all_new_predictions:
        # Save new predictions
        with open(args.output, 'w') as f:
            json.dump(all_new_predictions, f, indent=2)
        print(f"\nSaved {len(all_new_predictions)} new predictions to {args.output}")

        # Also merge with existing predictions
        merge_path = args.predictions.parent / 'predictions_merged.json'
        with open(args.predictions) as f:
            existing = json.load(f)
        merged = existing + all_new_predictions
        with open(merge_path, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"Saved merged predictions ({len(merged)} total) to {merge_path}")


if __name__ == '__main__':
    main()
