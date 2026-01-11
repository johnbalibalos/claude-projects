#!/usr/bin/env python3
"""
Full Gating Benchmark with LLM Judge

Runs Sonnet, Opus (via CLI), and Gemini models with 10 bootstrap runs.
Results are evaluated by Gemini 2.5 Pro judge.

Cost estimate: ~$31 (all Anthropic via CLI = $0)
- Gemini test calls: ~$6.50
- Gemini 2.5 Pro judge: ~$25

Usage:
    python scripts/run_full_benchmark.py --dry-run  # Test without API calls
    python scripts/run_full_benchmark.py --estimate  # Just show cost estimate
    python scripts/run_full_benchmark.py  # Full run (requires confirmation)
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent / "libs"))

# Load .env
env_file = PROJECT_ROOT.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value)


@dataclass
class BenchmarkConfig:
    """Configuration for the full benchmark."""

    # Models to test
    models: list[str] = field(default_factory=lambda: [
        "claude-sonnet-cli",  # Via Max subscription
        "claude-opus-cli",    # Via Max subscription
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ])

    # Experimental conditions
    context_levels: list[str] = field(default_factory=lambda: [
        "minimal", "standard", "rich"
    ])
    prompt_strategies: list[str] = field(default_factory=lambda: [
        "direct", "cot"
    ])

    # Bootstrap runs for statistical significance
    n_bootstrap: int = 10

    # Judge configuration
    judge_model: str = "gemini-2.5-pro"
    enable_judge: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results" / "full_benchmark")

    # Rate limiting
    cli_delay_seconds: float = 2.0

    @property
    def conditions_per_model(self) -> int:
        return len(self.context_levels) * len(self.prompt_strategies)

    @property
    def n_conditions(self) -> int:
        return len(self.models) * self.conditions_per_model


def estimate_cost(config: BenchmarkConfig, n_test_cases: int) -> dict[str, Any]:
    """Estimate API costs."""

    # Token estimates
    input_tokens = 2000
    output_tokens = 1000
    judge_input_tokens = 3500
    judge_output_tokens = 500

    # Pricing (per 1M tokens)
    pricing = {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    }

    calls_per_model = n_test_cases * config.n_bootstrap * config.conditions_per_model

    costs = {}
    total = 0.0
    cli_models = []

    for model in config.models:
        if model.endswith("-cli"):
            costs[model] = 0.0
            cli_models.append(model)
        elif model in pricing:
            p = pricing[model]
            cost = calls_per_model * (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
            costs[model] = cost
            total += cost

    # Judge costs
    judge_calls = calls_per_model * len(config.models)
    judge_pricing = pricing.get(config.judge_model, {"input": 1.25, "output": 5.00})
    judge_cost = judge_calls * (
        judge_input_tokens * judge_pricing["input"] +
        judge_output_tokens * judge_pricing["output"]
    ) / 1_000_000

    # CLI time estimate
    cli_calls = calls_per_model * len(cli_models)
    cli_time_mins = (cli_calls * config.cli_delay_seconds) / 60

    return {
        "n_test_cases": n_test_cases,
        "n_conditions": config.n_conditions,
        "conditions_per_model": config.conditions_per_model,
        "n_bootstrap": config.n_bootstrap,
        "calls_per_model": calls_per_model,
        "total_test_calls": calls_per_model * len(config.models),
        "costs_by_model": costs,
        "test_cost": total,
        "judge_calls": judge_calls,
        "judge_cost": judge_cost if config.enable_judge else 0,
        "total_cost": total + (judge_cost if config.enable_judge else 0),
        "cli_models": cli_models,
        "cli_time_mins": cli_time_mins,
    }


def print_cost_estimate(config: BenchmarkConfig, n_test_cases: int):
    """Print formatted cost estimate."""
    est = estimate_cost(config, n_test_cases)

    print("=" * 60)
    print("FULL GATING BENCHMARK - COST ESTIMATE")
    print("=" * 60)
    print(f"Test cases:         {est['n_test_cases']}")
    print(f"Bootstrap runs:     {est['n_bootstrap']}")
    print(f"Conditions/model:   {est['conditions_per_model']} (3 context × 2 strategy)")
    print(f"Total conditions:   {est['n_conditions']}")
    print(f"Calls per model:    {est['calls_per_model']}")
    print(f"Total test calls:   {est['total_test_calls']}")
    print()

    print("MODEL COSTS:")
    print("-" * 40)
    for model, cost in est["costs_by_model"].items():
        if model in est["cli_models"]:
            print(f"  {model:25} $    0.00  (via CLI/Max)")
        else:
            print(f"  {model:25} ${cost:>8.2f}")
    print()
    print(f"  SUBTOTAL (test calls):   ${est['test_cost']:>8.2f}")

    if config.enable_judge:
        print()
        print(f"JUDGE ({config.judge_model}):")
        print("-" * 40)
        print(f"  Judge calls:             {est['judge_calls']}")
        print(f"  Judge cost:              ${est['judge_cost']:>8.2f}")

    print()
    print("=" * 60)
    print(f"TOTAL ESTIMATED COST:      ${est['total_cost']:>8.2f}")
    print("=" * 60)

    if est["total_cost"] > 200:
        print(f"\n⚠️  WARNING: Estimated cost ${est['total_cost']:.2f} EXCEEDS $200 budget!")
    else:
        print(f"\n✓ Estimated cost ${est['total_cost']:.2f} is within $200 budget")

    if est["cli_models"]:
        print(f"\n⏱️  Anthropic CLI time: ~{est['cli_time_mins']:.0f} min (with {config.cli_delay_seconds}s delay)")
        print(f"   Models using CLI: {', '.join(est['cli_models'])}")


def verify_cli_setup():
    """Verify Claude CLI is working."""
    import subprocess

    print("\nVerifying Claude CLI setup...")
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"  ✓ Claude CLI: {result.stdout.strip()}")
            return True
        else:
            print(f"  ✗ Claude CLI error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  ✗ Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def verify_api_keys():
    """Verify required API keys."""
    print("\nVerifying API keys...")

    has_google = bool(os.environ.get("GOOGLE_API_KEY"))
    print(f"  {'✓' if has_google else '✗'} GOOGLE_API_KEY: {'set' if has_google else 'NOT SET'}")

    # Anthropic key not needed for CLI mode
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"  {'✓' if has_anthropic else '○'} ANTHROPIC_API_KEY: {'set' if has_anthropic else 'not needed (using CLI)'}")

    return has_google


def run_benchmark(config: BenchmarkConfig, test_cases_dir: Path, dry_run: bool = False):
    """Run the full benchmark."""
    from curation.omip_extractor import load_all_test_cases
    from experiments.conditions import get_all_conditions, MODELS
    from experiments.llm_client import create_client
    from experiments.prompts import build_prompt
    from evaluation.scorer import GatingScorer

    # Load test cases
    test_cases = load_all_test_cases(test_cases_dir)
    print(f"\nLoaded {len(test_cases)} test cases")

    # Generate conditions
    conditions = get_all_conditions(
        models=config.models,
        context_levels=config.context_levels,
        prompt_strategies=config.prompt_strategies,
    )
    print(f"Generated {len(conditions)} conditions")

    # Initialize
    scorer = GatingScorer()
    results = []

    config.output_dir.mkdir(parents=True, exist_ok=True)

    total_calls = len(test_cases) * len(conditions) * config.n_bootstrap
    current_call = 0

    for bootstrap_run in range(1, config.n_bootstrap + 1):
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP RUN {bootstrap_run}/{config.n_bootstrap}")
        print('='*60)

        for condition in conditions:
            client = create_client(condition.model, dry_run=dry_run)

            for test_case in test_cases:
                current_call += 1
                progress = f"[{current_call}/{total_calls}]"

                print(f"{progress} {condition.model} | {condition.name} | {test_case.test_case_id}")

                try:
                    # Build prompt
                    prompt = build_prompt(
                        test_case,
                        template_name=condition.prompt_strategy,
                        context_level=condition.context_level,
                        rag_mode=condition.rag_mode,
                    )

                    # Get prediction
                    response = client.call(prompt)

                    # Score
                    score_result = scorer.score(
                        response=response.content,
                        test_case=test_case,
                        model=condition.model,
                        condition=condition.name,
                    )

                    results.append({
                        "bootstrap_run": bootstrap_run,
                        "test_case_id": test_case.test_case_id,
                        "model": condition.model,
                        "condition": condition.name,
                        "hierarchy_f1": score_result.hierarchy_f1,
                        "structure_accuracy": score_result.structure_accuracy,
                        "critical_gate_recall": score_result.critical_gate_recall,
                        "parse_success": score_result.parse_success,
                        "raw_response": response.content[:500],  # Truncate for storage
                        "tokens_used": response.tokens_used,
                    })

                    print(f"         F1={score_result.hierarchy_f1:.3f} | Struct={score_result.structure_accuracy:.3f}")

                except Exception as e:
                    print(f"         ERROR: {e}")
                    results.append({
                        "bootstrap_run": bootstrap_run,
                        "test_case_id": test_case.test_case_id,
                        "model": condition.model,
                        "condition": condition.name,
                        "error": str(e),
                    })

        # Checkpoint after each bootstrap run
        checkpoint_file = config.output_dir / f"checkpoint_run_{bootstrap_run}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nCheckpoint saved: {checkpoint_file}")

    # Save final results
    final_file = config.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_file, "w") as f:
        json.dump({
            "config": {
                "models": config.models,
                "context_levels": config.context_levels,
                "prompt_strategies": config.prompt_strategies,
                "n_bootstrap": config.n_bootstrap,
            },
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n✓ Results saved: {final_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full gating benchmark")
    parser.add_argument("--test-cases", type=Path,
                        default=PROJECT_ROOT / "data" / "ground_truth",
                        help="Directory with test cases")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock API calls")
    parser.add_argument("--estimate", action="store_true",
                        help="Only show cost estimate")
    parser.add_argument("--n-bootstrap", type=int, default=10,
                        help="Number of bootstrap runs")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judge evaluation")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation")
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_bootstrap=args.n_bootstrap,
        enable_judge=not args.no_judge,
    )

    # Count test cases
    test_cases_dir = args.test_cases
    n_test_cases = len(list(test_cases_dir.glob("*.json")))

    # Show cost estimate
    print_cost_estimate(config, n_test_cases)

    if args.estimate:
        return

    # Verify setup
    if not verify_cli_setup():
        print("\n✗ CLI setup failed. Fix issues above and retry.")
        return

    if not verify_api_keys():
        print("\n✗ Missing required API keys.")
        return

    # Confirm
    if not args.dry_run and not args.yes:
        print()
        response = input("Proceed with benchmark? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            return

    # Run
    run_benchmark(config, test_cases_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
