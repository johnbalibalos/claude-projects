#!/usr/bin/env python3
"""
Run synthetic ablation study on Sonnet.

Tests all combinations of:
- Reasoning: direct, cot, wot
- Context: minimal, standard, rich
- RAG: none, oracle

With 1 synthetic test case.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add libs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypothesis_pipeline import (
    HypothesisPipeline,
    TrialInput,
    RAGMode,
)
from hypothesis_pipeline.base import Evaluator
from hypothesis_pipeline.config import ConfigLoader
from hypothesis_pipeline.rag import OracleRAGProvider


class SyntheticEvaluator(Evaluator):
    """Simple evaluator for synthetic test cases."""

    def extract(self, response: str) -> Any:
        """Extract answer from response."""
        # Try to find JSON in response
        try:
            # Look for JSON block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                return json.loads(response[start:end].strip())
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        # Fall back to looking for answer pattern
        response_lower = response.lower()
        if "42" in response or "forty-two" in response_lower:
            return {"answer": 42}

        return {"answer": response.strip()[:100]}

    def score(self, extracted: Any, ground_truth: Any) -> dict[str, float]:
        """Score extracted vs ground truth."""
        if isinstance(extracted, dict) and isinstance(ground_truth, dict):
            # Check if answer matches
            if extracted.get("answer") == ground_truth.get("answer"):
                return {"accuracy": 1.0, "partial": 1.0}

            # Partial credit for numeric closeness
            try:
                ext_val = float(extracted.get("answer", 0))
                gt_val = float(ground_truth.get("answer", 0))
                if gt_val != 0:
                    error = abs(ext_val - gt_val) / abs(gt_val)
                    partial = max(0, 1 - error)
                    return {"accuracy": 0.0, "partial": partial}
            except (ValueError, TypeError):
                pass

        return {"accuracy": 0.0, "partial": 0.0}


def create_synthetic_trial() -> TrialInput:
    """Create a synthetic test case."""
    return TrialInput(
        id="synthetic_math_001",
        raw_input={
            "question": "What is the answer to life, the universe, and everything?",
            "domain": "mathematics",
            "difficulty": "easy",
        },
        prompt="""You are a helpful assistant. Answer the following question.

Question: What is the answer to life, the universe, and everything?

Respond with JSON in this format:
{"answer": <your answer as a number>}""",
        ground_truth={"answer": 42},
        metadata={
            "source": "synthetic",
            "category": "pop_culture_math",
            "oracle_context": "According to Douglas Adams' 'The Hitchhiker's Guide to the Galaxy', the Answer to the Ultimate Question of Life, the Universe, and Everything is 42.",
        },
    )


def main():
    """Run the synthetic ablation experiment."""
    # Load config
    config_path = Path(__file__).parent / "configs" / "sonnet_synthetic_ablation.yaml"
    loader = ConfigLoader(config_path.parent)
    config = loader.load(config_path)

    print(f"Loaded config: {config.name}")
    print(f"Data source: {config.data_source.value}")
    print(f"Bootstrap runs: {config.n_bootstrap_runs}")

    # Create trial input
    trial = create_synthetic_trial()
    trials = [trial]

    print(f"\nTrial: {trial.id}")
    print(f"Ground truth: {trial.ground_truth}")

    # Create evaluator
    evaluator = SyntheticEvaluator()

    # Create oracle RAG provider
    oracle_rag = OracleRAGProvider()

    # Create pipeline
    pipeline = HypothesisPipeline(
        config=config,
        evaluator=evaluator,
        trial_inputs=trials,
        rag_providers={RAGMode.ORACLE: oracle_rag},
    )

    # Show conditions
    print(f"\nConditions to test: {len(pipeline.conditions)}")
    for cond in pipeline.conditions:
        print(f"  - {cond.name}")

    # Estimate cost
    n_conditions = len(pipeline.conditions)
    n_trials = len(trials)
    n_runs = config.n_bootstrap_runs
    total_calls = n_conditions * n_trials * n_runs

    # Sonnet pricing
    avg_input = 1500
    avg_output = 500
    input_cost = (total_calls * avg_input / 1_000_000) * 3.0
    output_cost = (total_calls * avg_output / 1_000_000) * 15.0
    total_cost = input_cost + output_cost

    print(f"\n{'='*50}")
    print("COST ESTIMATE")
    print(f"{'='*50}")
    print(f"API calls: {total_calls}")
    print(f"Est. input tokens: {total_calls * avg_input:,}")
    print(f"Est. output tokens: {total_calls * avg_output:,}")
    print(f"Est. cost: ${total_cost:.2f}")
    print(f"{'='*50}\n")

    # Confirm before running
    confirm = input("Run experiment? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return 1

    # Run!
    results = pipeline.run(verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for cond_name, metrics in results.metrics_by_condition.items():
        print(f"\n{cond_name}:")
        print(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
        print(f"  Accuracy: {metrics.get('accuracy_mean', 0):.2f}")
        print(f"  Latency: {metrics.get('latency_mean', 0):.1f}s")
        print(f"  Input tokens: {metrics.get('input_tokens_mean', 0):.0f}")
        print(f"  Output tokens: {metrics.get('output_tokens_mean', 0):.0f}")

    # Generate report
    report = pipeline.generate_report(results)
    report_path = config.output_dir / f"{config.name}_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
