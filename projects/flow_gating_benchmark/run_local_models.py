#!/usr/bin/env python3
"""
Run gating benchmark with local LLMs via Ollama.

Usage:
    python run_local_models.py --check                  # Verify Ollama is running
    python run_local_models.py --list-models            # Show available models
    python run_local_models.py --test plan              # Show execution plan
    python run_local_models.py --test quick             # 2 test cases, 1 model
    python run_local_models.py --test full              # Full benchmark

    # Specify models
    python run_local_models.py --test quick --model llama3.1:8b
    python run_local_models.py --test full --model llama3.1:70b --model qwen2.5:72b

    # Ablate context or strategy
    python run_local_models.py --test quick --context minimal --context standard
    python run_local_models.py --test quick --strategy cot
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from src.experiments.conditions import (
    ExperimentCondition,
    MODELS,
    CONTEXT_LEVELS,
    PROMPT_STRATEGIES,
    RECOMMENDED_LOCAL_MODELS,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def check_ollama(base_url: str = DEFAULT_OLLAMA_URL) -> tuple[bool, list[str]]:
    """Check if Ollama is running and return available models."""
    if not HTTPX_AVAILABLE:
        return False, []

    try:
        client = httpx.Client(timeout=5.0)
        response = client.get(f"{base_url}/api/tags")
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        return True, models
    except Exception:
        return False, []


def cmd_check(args) -> int:
    """Check if Ollama is running."""
    available, models = check_ollama(args.ollama_url)

    if available:
        print(f"Ollama is running at {args.ollama_url}")
        print(f"  {len(models)} model(s) installed")
        return 0
    else:
        print(f"Ollama not available at {args.ollama_url}")
        if not HTTPX_AVAILABLE:
            print("  httpx not installed. Run: pip install httpx")
        else:
            print("  Start Ollama with: ollama serve")
        return 1


def cmd_list_models(args) -> int:
    """List available local models."""
    available, installed = check_ollama(args.ollama_url)

    if not available:
        print(f"ERROR: Ollama is not running at {args.ollama_url}")
        print()
        print("To start Ollama:")
        print("  ollama serve")
        return 1

    print("=" * 60)
    print("AVAILABLE LOCAL MODELS")
    print("=" * 60)
    print()

    if installed:
        print("Installed models:")
        for model in sorted(installed):
            rec = RECOMMENDED_LOCAL_MODELS.get(model, "")
            marker = "*" if rec else " "
            note = f"  ({rec})" if rec else ""
            print(f"  {marker} {model}{note}")
    else:
        print("No models installed.")

    print()
    print("Recommended models to install:")
    for model, desc in RECOMMENDED_LOCAL_MODELS.items():
        if model not in installed:
            print(f"  ollama pull {model:<20}  # {desc}")

    print()
    print("* = Recommended for scientific reasoning")
    return 0


def cmd_plan(args, models: list[str], contexts: list[str], strategies: list[str]) -> int:
    """Show execution plan without running."""
    from src.curation.omip_extractor import load_all_test_cases

    # Load test cases to get count
    ground_truth_dir = Path(__file__).parent / "data" / "ground_truth"

    try:
        all_test_cases = load_all_test_cases(ground_truth_dir)
        total_available = len(all_test_cases)
    except Exception:
        total_available = "unknown"

    if args.test == "quick":
        n_cases = 2
    elif args.test == "standard":
        n_cases = 5
    else:  # full
        n_cases = total_available if isinstance(total_available, int) else 10

    n_conditions = len(contexts) * len(strategies)
    total_trials = n_cases * n_conditions * len(models)

    print("=" * 60)
    print("EXECUTION PLAN")
    print("=" * 60)
    print()
    print(f"Test level: {args.test}")
    print(f"Test cases: {n_cases}" + (f" (of {total_available} available)" if total_available != "unknown" else ""))
    print()
    print(f"Models ({len(models)}):")
    for m in models:
        rec = RECOMMENDED_LOCAL_MODELS.get(m, "")
        note = f"  ({rec})" if rec else ""
        print(f"  - {m}{note}")
    print()
    print(f"Context levels ({len(contexts)}): {', '.join(contexts)}")
    print(f"Prompt strategies ({len(strategies)}): {', '.join(strategies)}")
    print()
    print(f"Conditions per model: {n_conditions}")
    print(f"Total trials: {total_trials}")
    print()

    # Time estimate (local models are slower than cloud)
    est_time_per_trial = 45  # seconds, conservative for local inference
    total_seconds = total_trials * est_time_per_trial
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    print(f"Estimated time: {hours}h {minutes}m (at ~{est_time_per_trial}s/trial)")
    print()
    print("-" * 60)
    print("To run:")
    model_args = " ".join(f"--model {m}" for m in models)
    print(f"  python run_local_models.py --test {args.test} {model_args}")

    return 0


def cmd_run(args, models: list[str], contexts: list[str], strategies: list[str]) -> int:
    """Run the benchmark."""
    from src.curation.omip_extractor import load_all_test_cases
    from src.experiments.runner import ExperimentRunner, ExperimentConfig
    from src.evaluation.scorer import compute_aggregate_metrics

    # Check Ollama first
    available, installed = check_ollama(args.ollama_url)
    if not available:
        print("ERROR: Ollama is not running")
        print(f"  URL: {args.ollama_url}")
        print("  Start with: ollama serve")
        return 1

    # Verify models are installed
    missing = [m for m in models if m not in installed]
    if missing:
        print(f"ERROR: Models not installed: {missing}")
        print()
        print("Install with:")
        for m in missing:
            print(f"  ollama pull {m}")
        return 1

    # Load test cases
    ground_truth_dir = Path(__file__).parent / "data" / "ground_truth"
    all_test_cases = load_all_test_cases(ground_truth_dir)

    if not all_test_cases:
        print("ERROR: No test cases found!")
        print(f"  Directory: {ground_truth_dir}")
        return 1

    # Limit based on test level
    if args.test == "quick":
        test_cases = all_test_cases[:2]
    elif args.test == "standard":
        test_cases = all_test_cases[:5]
    else:
        test_cases = all_test_cases

    # Build conditions
    conditions = []
    for model in models:
        for context in contexts:
            for strategy in strategies:
                conditions.append(ExperimentCondition(
                    name=f"{model.replace(':', '-')}_{context}_{strategy}",
                    model=model,
                    context_level=context,
                    prompt_strategy=strategy,
                ))

    total_trials = len(test_cases) * len(conditions)

    print("=" * 60)
    print("LOCAL MODEL GATING BENCHMARK")
    print("=" * 60)
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Models: {len(models)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Total trials: {total_trials}")
    print()

    # Set up environment for runner
    os.environ["OLLAMA_BASE_URL"] = args.ollama_url

    # Create output directory
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = ExperimentConfig(
        name=f"local_models_{timestamp}",
        test_cases_dir=str(ground_truth_dir),
        output_dir=str(output_dir),
        conditions=conditions,
        dry_run=False,
        log_level="INFO" if args.verbose else "WARNING",
    )

    # Run experiment
    runner = ExperimentRunner(config)
    result = runner.run()

    # Summary by model
    print()
    print("=" * 60)
    print("RESULTS BY MODEL")
    print("=" * 60)

    for model in models:
        model_results = [r for r in result.results if model in r.model]
        if model_results:
            metrics = compute_aggregate_metrics(model_results)
            print(f"\n{model}:")
            print(f"  Hierarchy F1:        {metrics.get('hierarchy_f1_mean', 0):.3f}")
            print(f"  Structure Accuracy:  {metrics.get('structure_accuracy_mean', 0):.3f}")
            print(f"  Critical Gate Recall:{metrics.get('critical_gate_recall_mean', 0):.3f}")
            print(f"  Parse Success Rate:  {metrics.get('parse_success_rate', 0):.1%}")

    # Summary by context level
    print()
    print("-" * 60)
    print("RESULTS BY CONTEXT LEVEL")
    print("-" * 60)

    for context in contexts:
        context_results = [r for r in result.results if context in r.condition]
        if context_results:
            metrics = compute_aggregate_metrics(context_results)
            print(f"\n{context}:")
            print(f"  Hierarchy F1:        {metrics.get('hierarchy_f1_mean', 0):.3f}")
            print(f"  Parse Success Rate:  {metrics.get('parse_success_rate', 0):.1%}")

    # Save combined results
    results_file = output_dir / f"local_models_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print()
    print(f"Results saved to: {results_file}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run gating benchmark with local LLMs via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_local_models.py --check
  python run_local_models.py --list-models
  python run_local_models.py --test plan
  python run_local_models.py --test quick --model llama3.1:8b
  python run_local_models.py --test full --model llama3.1:70b --model qwen2.5:72b
  python run_local_models.py --test quick --context minimal --context standard
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="Check if Ollama is running"
    )
    mode_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available local models"
    )
    mode_group.add_argument(
        "--test",
        choices=["plan", "quick", "standard", "full"],
        help="Test level: plan (dry run), quick (2 cases), standard (5), full (all)"
    )

    # Model selection
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model to test (can specify multiple). Default: llama3.1:8b"
    )

    # Condition selection
    parser.add_argument(
        "--context",
        action="append",
        dest="contexts",
        choices=CONTEXT_LEVELS,
        help="Context levels to test. Default: all"
    )
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        choices=PROMPT_STRATEGIES,
        help="Prompt strategies to test. Default: all"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results/local_models"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Handle info commands
    if args.check:
        return cmd_check(args)

    if args.list_models:
        return cmd_list_models(args)

    # Set defaults for benchmark run
    models = args.models or ["llama3.1:8b"]
    contexts = args.contexts or CONTEXT_LEVELS
    strategies = args.strategies or PROMPT_STRATEGIES

    if args.test == "plan":
        return cmd_plan(args, models, contexts, strategies)

    return cmd_run(args, models, contexts, strategies)


if __name__ == "__main__":
    sys.exit(main())
