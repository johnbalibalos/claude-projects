#!/usr/bin/env python3
"""
Run synthetic ablation study on Sonnet.

Tests all combinations of:
- Reasoning: direct, cot, wot
- Context: minimal, standard, rich
- RAG: none, oracle

With 1 synthetic test case.

Usage:
    python run_synthetic_ablation.py           # Interactive mode (requires confirmation)
    python run_synthetic_ablation.py --force   # Skip confirmation (use with caution)
"""

import argparse
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
from hypothesis_pipeline.cost import confirm_experiment_cost
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run synthetic ablation experiment on Sonnet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_synthetic_ablation.py              # Interactive mode
    python run_synthetic_ablation.py --force      # Skip confirmation
    python run_synthetic_ablation.py --dry-run    # Show cost only
        """,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation and run immediately (use with caution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cost estimate without running the experiment",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: sonnet_synthetic_ablation.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the synthetic ablation experiment."""
    args = parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
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

    # Dry run - just show cost estimate
    if args.dry_run:
        from hypothesis_pipeline.cost import estimate_experiment_cost
        estimate = estimate_experiment_cost(config, n_test_cases=len(trials))
        print("\n" + estimate.format_summary())
        return 0

    # Cost confirmation (unless --force is used)
    if args.force:
        print("\n[--force] Skipping cost confirmation...")
    else:
        confirmed = confirm_experiment_cost(
            config=config,
            n_test_cases=len(trials),
        )
        if not confirmed:
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
