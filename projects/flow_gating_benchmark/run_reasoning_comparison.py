#!/usr/bin/env python3
"""
Quick comparison of Opus vs Sonnet on reasoning benchmark.

Tests a subset of reasoning test cases to demonstrate the difference
between pattern matching and genuine biological reasoning.
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, "src")

from reasoning_benchmark.test_cases import (
    LINEAGE_NEGATIVE_TESTS,
    BIOLOGICAL_IMPOSSIBILITY_TESTS,
    CONTEXT_SWITCH_TESTS,
)
from reasoning_benchmark.evaluator import ReasoningEvaluator
from reasoning_benchmark.schemas import ReasoningQuality

try:
    from anthropic import Anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    sys.exit(1)


def run_test(client, model: str, prompt: str, max_tokens: int = 2048) -> str:
    """Call model with a prompt."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def main():
    # Check API key - try multiple sources
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Try loading from .env files
    env_paths = [
        "/home/user/claude-projects/.env",
        ".env",
        os.path.expanduser("~/.env"),
    ]

    if not api_key:
        for env_path in env_paths:
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("ANTHROPIC_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if api_key:
                    break

    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        print("Please run: export ANTHROPIC_API_KEY=your_key")
        print("Or create a .env file with ANTHROPIC_API_KEY=your_key")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    evaluator = ReasoningEvaluator()

    # Models to compare
    models = {
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-20250514",
    }

    # Select key test cases (one from each category)
    test_cases = [
        LINEAGE_NEGATIVE_TESTS[0],   # NK cells - dump channel
        BIOLOGICAL_IMPOSSIBILITY_TESTS[0],  # Triple positive doublet
        CONTEXT_SWITCH_TESTS[0],     # Lung autofluorescence
    ]

    print("=" * 70)
    print("REASONING BENCHMARK: OPUS vs SONNET COMPARISON")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Test cases: {len(test_cases)}")
    print()

    results = {model: [] for model in models}

    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {test.test_id}")
        print(f"TYPE: {test.test_type.value}")
        print(f"{'='*70}")
        print(f"\nPrompt: {test.prompt[:200]}...")
        print()

        for model_name, model_id in models.items():
            print(f"\n--- {model_name.upper()} ---")

            try:
                response = run_test(client, model_id, test.prompt)
                result = evaluator.evaluate(test, response)

                print(f"Quality: {result.quality.value}")
                print(f"Reasoning Score: {result.reasoning_score:.2f}")
                print(f"Failure Indicators: {len(result.failure_indicators_found)}")
                print(f"Concepts Found: {len(result.reasoning_concepts_found)}/{len(test.criteria.required_reasoning_concepts)}")

                # Show key excerpts
                if result.failure_indicators_found:
                    print(f"⚠️  Pattern matching detected: {result.failure_indicators_found[:2]}")
                if result.reasoning_concepts_found:
                    print(f"✓  Reasoning concepts: {result.reasoning_concepts_found[:3]}")

                # Show brief response excerpt
                excerpt = response[:300].replace("\n", " ")
                print(f"\nResponse excerpt: {excerpt}...")

                results[model_name].append(result)

            except Exception as e:
                print(f"Error: {e}")
                results[model_name].append(None)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name in models:
        model_results = [r for r in results[model_name] if r is not None]
        if not model_results:
            continue

        pass_count = sum(1 for r in model_results if r.quality == ReasoningQuality.PASS)
        partial_count = sum(1 for r in model_results if r.quality == ReasoningQuality.PARTIAL)
        fail_count = sum(1 for r in model_results if r.quality == ReasoningQuality.FAIL)
        avg_reasoning = sum(r.reasoning_score for r in model_results) / len(model_results)
        pattern_matching = sum(1 for r in model_results if len(r.failure_indicators_found) > 0)

        print(f"\n{model_name.upper()}:")
        print(f"  Pass: {pass_count}, Partial: {partial_count}, Fail: {fail_count}")
        print(f"  Avg Reasoning Score: {avg_reasoning:.2f}")
        print(f"  Pattern Matching Detected: {pattern_matching}/{len(model_results)}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "models": list(models.keys()),
        "test_ids": [t.test_id for t in test_cases],
        "results": {
            model: [r.to_dict() if r else None for r in model_results]
            for model, model_results in results.items()
        },
    }

    output_path = f"results/reasoning_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
