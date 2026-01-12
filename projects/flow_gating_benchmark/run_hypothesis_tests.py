#!/usr/bin/env python3
"""
Run hypothesis tests to rigorously evaluate LLM reasoning capabilities.

This script implements falsifiable hypothesis tests to distinguish between:
1. Token frequency effects vs. genuine reasoning (Frequency Confound)
2. Prose parsing issues vs. reasoning deficits (Format Ablation)
3. Prior hallucination vs. context processing (CoT Mechanistic)
4. Context blindness vs. safety over-triggering (Cognitive Refusal)

Example usage:
    # Run all tests on existing results
    python run_hypothesis_tests.py --results results/benchmark_results.json

    # Run specific tests
    python run_hypothesis_tests.py --tests frequency_confound alien_cell

    # Run with live model calls (requires API key)
    python run_hypothesis_tests.py --live --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hypothesis_tests import (
    AblationConfig,
    AlienCellTest,
    CoTAnnotator,
    CognitiveRefusalTest,
    FormatAblationTest,
    FrequencyCorrelation,
    HypothesisTestRunner,
    HypothesisTestResult,
    PromptFormat,
    PubMedFrequencyLookup,
)
from hypothesis_tests.runner import HypothesisType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hypothesis tests for LLM reasoning evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze existing benchmark results
    python run_hypothesis_tests.py --results results/benchmark_results.json

    # Run frequency correlation analysis
    python run_hypothesis_tests.py --tests frequency_confound --results results/

    # Run all tests with live model
    python run_hypothesis_tests.py --live --model claude-sonnet-4-20250514

    # Estimate cost before running
    python run_hypothesis_tests.py --estimate-cost --tests alien_cell format_ablation
        """,
    )

    parser.add_argument(
        "--results",
        type=str,
        help="Path to existing benchmark results (JSON file or directory)",
    )

    parser.add_argument(
        "--tests",
        nargs="+",
        choices=[t.value for t in HypothesisType],
        default=[t.value for t in HypothesisType],
        help="Which hypothesis tests to run (default: all)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./hypothesis_test_results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run with live model calls (requires API key)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for live tests",
    )

    parser.add_argument(
        "--test-cases",
        type=str,
        default="data/ground_truth",
        help="Directory containing test cases",
    )

    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate cost without running tests",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actual API calls",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_existing_results(results_path: str) -> dict:
    """Load existing benchmark results."""
    path = Path(results_path)

    if path.is_file():
        with open(path) as f:
            return json.load(f)

    elif path.is_dir():
        # Find the most recent results file
        result_files = sorted(path.glob("*results*.json"), reverse=True)
        if result_files:
            with open(result_files[0]) as f:
                return json.load(f)
        raise FileNotFoundError(f"No result files found in {path}")

    raise FileNotFoundError(f"Results not found: {results_path}")


def extract_population_scores(results: dict) -> dict[str, float]:
    """Extract per-population F1 scores from benchmark results."""
    population_scores = {}

    for result in results.get("results", []):
        if result.get("evaluation"):
            # Extract scores for matched populations
            matched = result["evaluation"].get("matched_gates", [])
            for gate in matched:
                name = gate.get("predicted", gate.get("ground_truth", "Unknown"))
                score = gate.get("similarity", 1.0)
                if name not in population_scores:
                    population_scores[name] = []
                population_scores[name].append(score)

    # Average scores for each population
    return {
        name: sum(scores) / len(scores)
        for name, scores in population_scores.items()
        if scores
    }


def estimate_cost(config: AblationConfig, n_test_cases: int) -> None:
    """Estimate the cost of running hypothesis tests."""
    print("\n" + "=" * 60)
    print("COST ESTIMATE")
    print("=" * 60)

    # Rough estimates based on typical token usage
    MODEL_COSTS = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    }

    model_pricing = MODEL_COSTS.get(
        config.model,
        MODEL_COSTS["claude-sonnet-4-20250514"],
    )

    # Estimate calls per test
    calls_per_test = {
        HypothesisType.FREQUENCY_CONFOUND: 0,  # No LLM calls
        HypothesisType.ALIEN_CELL: n_test_cases * 2,  # Original + alien
        HypothesisType.FORMAT_ABLATION: n_test_cases * 3,  # 3 formats
        HypothesisType.COT_MECHANISTIC: 0,  # Analysis only
        HypothesisType.COGNITIVE_REFUSAL: n_test_cases * 4,  # 4 variants
    }

    total_calls = sum(
        calls_per_test.get(t, 0)
        for t in config.tests
    )

    # Estimate tokens per call (rough)
    avg_input_tokens = 2000
    avg_output_tokens = 1000

    total_input_tokens = total_calls * avg_input_tokens
    total_output_tokens = total_calls * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    print(f"\nModel: {config.model}")
    print(f"Test cases: {n_test_cases}")
    print(f"Tests to run: {[t.value for t in config.tests]}")
    print(f"\nEstimated API calls: {total_calls}")
    print(f"Estimated input tokens: {total_input_tokens:,}")
    print(f"Estimated output tokens: {total_output_tokens:,}")
    print(f"\nEstimated cost: ${total_cost:.2f}")
    print(f"  Input: ${input_cost:.2f}")
    print(f"  Output: ${output_cost:.2f}")
    print("\n" + "=" * 60)


def run_offline_analysis(
    results: dict,
    config: AblationConfig,
) -> HypothesisTestResult:
    """Run hypothesis tests on existing results (no live API calls)."""
    runner = HypothesisTestRunner(config)

    # Frequency correlation
    if HypothesisType.FREQUENCY_CONFOUND in config.tests:
        logger.info("Running frequency correlation analysis...")
        population_scores = extract_population_scores(results)
        if population_scores:
            runner.run_frequency_correlation(population_scores)
        else:
            logger.warning("No population scores found for frequency analysis")

    # CoT analysis
    if HypothesisType.COT_MECHANISTIC in config.tests:
        logger.info("Running CoT mechanistic analysis...")
        # Extract CoT responses from results
        cot_responses = []
        for result in results.get("results", []):
            if "cot" in result.get("condition", "").lower():
                raw_response = result.get("raw_response")
                if raw_response:
                    # Need to reconstruct test case - simplified version
                    cot_responses.append((raw_response, None))

        if cot_responses:
            logger.info(f"Found {len(cot_responses)} CoT responses to analyze")
            # Note: Full analysis requires test case objects
        else:
            logger.warning("No CoT responses found for mechanistic analysis")

    return runner.finalize()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert test names to HypothesisType
    tests = [HypothesisType(t) for t in args.tests]

    config = AblationConfig(
        tests=tests,
        model=args.model,
        output_dir=args.output,
        dry_run=args.dry_run,
    )

    # Count test cases
    test_cases_path = Path(args.test_cases)
    if test_cases_path.exists():
        n_test_cases = len(list(test_cases_path.glob("*.json")))
    else:
        n_test_cases = 8  # Default estimate

    # Cost estimation
    if args.estimate_cost:
        estimate_cost(config, n_test_cases)
        return

    # Run tests
    if args.results:
        # Offline analysis of existing results
        logger.info(f"Loading results from {args.results}")
        results = load_existing_results(args.results)
        test_result = run_offline_analysis(results, config)

    elif args.live:
        # Live model testing
        if args.dry_run:
            logger.info("Dry run mode - no actual API calls")

        logger.warning("Live testing requires model integration - not yet implemented")
        logger.info("Use --results to analyze existing benchmark results")
        return

    else:
        logger.error("Either --results or --live must be specified")
        return

    # Print report
    print(test_result.format_report())


if __name__ == "__main__":
    main()
