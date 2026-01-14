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
    python scripts/run_full_benchmark.py --resume  # Resume from checkpoint
    python scripts/run_full_benchmark.py  # Full run (requires confirmation)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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

    # Parallelization
    parallel_workers: int = 5  # For Gemini API calls

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


def load_checkpoint(config: BenchmarkConfig) -> tuple[list[dict], set[tuple]]:
    """Load the most recent checkpoint(s) and return results + completed keys.

    Handles both old format (checkpoint_run_N.json) and new format
    (checkpoint_cli.json, checkpoint_api.json).
    """
    results = []
    completed = set()

    # Check for new format checkpoints first
    cli_checkpoint = config.output_dir / "checkpoint_cli.json"
    api_checkpoint = config.output_dir / "checkpoint_api.json"

    loaded_files = []

    if cli_checkpoint.exists():
        with open(cli_checkpoint) as f:
            cli_results = json.load(f)
            results.extend(cli_results)
            loaded_files.append(cli_checkpoint.name)

    if api_checkpoint.exists():
        with open(api_checkpoint) as f:
            api_results = json.load(f)
            # Merge, avoiding duplicates
            existing_keys = {
                (r["bootstrap_run"], r["test_case_id"], r["model"], r["condition"])
                for r in results
            }
            for r in api_results:
                key = (r["bootstrap_run"], r["test_case_id"], r["model"], r["condition"])
                if key not in existing_keys:
                    results.append(r)
            loaded_files.append(api_checkpoint.name)

    # Fall back to old format if no new format found
    if not loaded_files:
        checkpoint_files = sorted(config.output_dir.glob("checkpoint_run_*.json"))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            with open(latest_checkpoint) as f:
                results = json.load(f)
            loaded_files.append(latest_checkpoint.name)

    if not loaded_files:
        print("\nNo checkpoints found")
        return results, completed

    print(f"\nLoading checkpoint(s): {', '.join(loaded_files)}")

    # Build set of completed (bootstrap_run, test_case_id, model, condition)
    for r in results:
        key = (r["bootstrap_run"], r["test_case_id"], r["model"], r["condition"])
        completed.add(key)

    print(f"  Loaded {len(results)} results, {len(completed)} unique completions")
    return results, completed


def run_single_call(
    test_case,
    condition,
    bootstrap_run: int,
    build_prompt_fn,
    scorer,
    dry_run: bool,
) -> dict:
    """Execute a single benchmark call. Used for parallel execution."""
    from experiments.llm_client import create_client

    client = create_client(condition.model, dry_run=dry_run)

    try:
        prompt = build_prompt_fn(
            test_case,
            template_name=condition.prompt_strategy,
            context_level=condition.context_level,
            reference=condition.reference,
        )

        response = client.call(prompt)

        score_result = scorer.score(
            response=response.content,
            test_case=test_case,
            model=condition.model,
            condition=condition.name,
        )

        return {
            "bootstrap_run": bootstrap_run,
            "test_case_id": test_case.test_case_id,
            "model": condition.model,
            "condition": condition.name,
            "hierarchy_f1": score_result.hierarchy_f1,
            "structure_accuracy": score_result.structure_accuracy,
            "critical_gate_recall": score_result.critical_gate_recall,
            "parse_success": score_result.parse_success,
            "raw_response": response.content,  # Full response for judge
            "tokens_used": response.tokens_used,
        }

    except Exception as e:
        return {
            "bootstrap_run": bootstrap_run,
            "test_case_id": test_case.test_case_id,
            "model": condition.model,
            "condition": condition.name,
            "error": str(e),
        }


def run_cli_batch(
    cli_conditions,
    test_cases,
    config: BenchmarkConfig,
    scorer,
    completed: set,
    total_calls: int,
    update_progress,
    results_lock,
    dry_run: bool,
) -> list[dict]:
    """Run CLI conditions sequentially with rate limiting.

    Returns list of results from CLI calls.
    """
    from experiments.llm_client import create_client
    from experiments.prompts import build_prompt

    cli_results = []

    print(f"\n{'='*60}")
    print("CLI MODELS (sequential, cache-optimized)")
    print('='*60)

    for condition in cli_conditions:
        print(f"\n--- {condition.model} | {condition.name} ---")
        client = create_client(condition.model, dry_run=dry_run)

        for test_case in test_cases:
            # Build prompt once for all bootstrap runs (cache optimization)
            prompt = build_prompt(
                test_case,
                template_name=condition.prompt_strategy,
                context_level=condition.context_level,
                reference=condition.reference,
            )

            for bootstrap_run in range(1, config.n_bootstrap + 1):
                key = (bootstrap_run, test_case.test_case_id, condition.model, condition.name)

                with results_lock:
                    if key in completed:
                        update_progress()
                        continue

                current = update_progress()
                progress = f"[{current}/{total_calls}]"
                print(f"{progress} {test_case.test_case_id} run={bootstrap_run}")

                try:
                    response = client.call(prompt)

                    score_result = scorer.score(
                        response=response.content,
                        test_case=test_case,
                        model=condition.model,
                        condition=condition.name,
                    )

                    result = {
                        "bootstrap_run": bootstrap_run,
                        "test_case_id": test_case.test_case_id,
                        "model": condition.model,
                        "condition": condition.name,
                        "hierarchy_f1": score_result.hierarchy_f1,
                        "structure_accuracy": score_result.structure_accuracy,
                        "critical_gate_recall": score_result.critical_gate_recall,
                        "parse_success": score_result.parse_success,
                        "raw_response": response.content,  # Full response for judge
                        "tokens_used": response.tokens_used,
                    }
                    cli_results.append(result)
                    with results_lock:
                        completed.add(key)

                    print(f"         F1={score_result.hierarchy_f1:.3f} | Struct={score_result.structure_accuracy:.3f}")

                    # Rate limit for CLI
                    time.sleep(config.cli_delay_seconds)

                except Exception as e:
                    print(f"         ERROR: {e}")
                    result = {
                        "bootstrap_run": bootstrap_run,
                        "test_case_id": test_case.test_case_id,
                        "model": condition.model,
                        "condition": condition.name,
                        "error": str(e),
                    }
                    cli_results.append(result)
                    with results_lock:
                        completed.add(key)

        # Checkpoint after each CLI condition
        checkpoint_file = config.output_dir / "checkpoint_cli.json"
        with open(checkpoint_file, "w") as f:
            json.dump(cli_results, f, indent=2)
        print(f"  Checkpoint saved: {checkpoint_file.name}")

    return cli_results


def run_api_batch(
    api_conditions,
    test_cases,
    config: BenchmarkConfig,
    scorer,
    completed: set,
    total_calls: int,
    update_progress,
    results_lock,
    dry_run: bool,
) -> list[dict]:
    """Run API conditions in parallel with ThreadPoolExecutor.

    Returns list of results from API calls.
    """
    from experiments.prompts import build_prompt

    api_results = []

    print(f"\n{'='*60}")
    print(f"API MODELS (parallel, {config.parallel_workers} workers)")
    print('='*60)

    # Build all pending tasks
    pending_tasks = []
    with results_lock:
        for condition in api_conditions:
            for test_case in test_cases:
                for bootstrap_run in range(1, config.n_bootstrap + 1):
                    key = (bootstrap_run, test_case.test_case_id, condition.model, condition.name)
                    if key not in completed:
                        pending_tasks.append((test_case, condition, bootstrap_run))

    if not pending_tasks:
        print("\n  All API calls already completed")
        return api_results

    print(f"\n  {len(pending_tasks)} pending API calls")

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {
            executor.submit(
                run_single_call,
                task[0],  # test_case
                task[1],  # condition
                task[2],  # bootstrap_run
                build_prompt,
                scorer,
                dry_run,
            ): task
            for task in pending_tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            test_case, condition, bootstrap_run = task
            current = update_progress()
            progress = f"[{current}/{total_calls}]"

            try:
                result = future.result()
                api_results.append(result)

                with results_lock:
                    key = (bootstrap_run, test_case.test_case_id, condition.model, condition.name)
                    completed.add(key)

                if "error" in result:
                    print(f"{progress} {condition.model} | {test_case.test_case_id} run={bootstrap_run}")
                    print(f"         ERROR: {result['error']}")
                else:
                    print(f"{progress} {condition.model} | {test_case.test_case_id} run={bootstrap_run}")
                    print(f"         F1={result['hierarchy_f1']:.3f} | Struct={result['structure_accuracy']:.3f}")

            except Exception as e:
                print(f"{progress} {condition.model} | {test_case.test_case_id} run={bootstrap_run}")
                print(f"         FUTURE ERROR: {e}")
                result = {
                    "bootstrap_run": bootstrap_run,
                    "test_case_id": test_case.test_case_id,
                    "model": condition.model,
                    "condition": condition.name,
                    "error": str(e),
                }
                api_results.append(result)

    # Checkpoint after API batch
    checkpoint_file = config.output_dir / "checkpoint_api.json"
    with open(checkpoint_file, "w") as f:
        json.dump(api_results, f, indent=2)
    print(f"\nCheckpoint saved: {checkpoint_file.name}")

    return api_results


def run_benchmark(
    config: BenchmarkConfig,
    test_cases_dir: Path,
    dry_run: bool = False,
    resume: bool = False,
):
    """
    Run the full benchmark with parallelization and checkpoint resume.

    Optimized for prompt caching:
    - Iterates condition -> test_case -> bootstrap_runs
    - All bootstrap runs for same (condition, test_case) use identical prompts
    - Runs 2+ hit the prompt cache for significant cost savings

    Parallelization:
    - CLI models: Sequential with rate limiting (Claude Max subscription)
    - API models: Parallel with ThreadPoolExecutor (Gemini, Anthropic API)
    - CLI and API run CONCURRENTLY (API in background thread)
    """
    import threading

    from curation.omip_extractor import load_all_test_cases
    from evaluation.scorer import GatingScorer
    from experiments.conditions import get_all_conditions

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

    # Split conditions into CLI (sequential) and API (parallel)
    cli_conditions = [c for c in conditions if c.model.endswith("-cli")]
    api_conditions = [c for c in conditions if not c.model.endswith("-cli")]
    print(f"  CLI conditions: {len(cli_conditions)} (sequential)")
    print(f"  API conditions: {len(api_conditions)} (parallel, {config.parallel_workers} workers)")

    if cli_conditions and api_conditions:
        print("  Execution: CONCURRENT (CLI + API run simultaneously)")

    # Initialize
    scorer = GatingScorer()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    if resume:
        results, completed = load_checkpoint(config)
    else:
        results = []
        completed = set()

    total_calls = len(test_cases) * len(conditions) * config.n_bootstrap
    completed_count = len(completed)

    if completed_count > 0:
        print(f"\nResuming: {completed_count}/{total_calls} already completed")

    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    progress_counter = [completed_count]
    results_lock = threading.Lock()

    def update_progress():
        with progress_lock:
            progress_counter[0] += 1
            return progress_counter[0]

    # === CONCURRENT EXECUTION: CLI + API ===
    cli_results = []
    api_results = []
    api_future = None

    # Start API calls in background thread immediately (if any)
    if api_conditions:
        with ThreadPoolExecutor(max_workers=1) as background_executor:
            api_future = background_executor.submit(
                run_api_batch,
                api_conditions,
                test_cases,
                config,
                scorer,
                completed,
                total_calls,
                update_progress,
                results_lock,
                dry_run,
            )

            # Run CLI calls in foreground while API runs in background
            if cli_conditions:
                cli_results = run_cli_batch(
                    cli_conditions,
                    test_cases,
                    config,
                    scorer,
                    completed,
                    total_calls,
                    update_progress,
                    results_lock,
                    dry_run,
                )

            # Wait for API to complete
            print(f"\n{'='*60}")
            print("Waiting for API calls to complete...")
            print('='*60)
            api_results = api_future.result()

    elif cli_conditions:
        # Only CLI conditions, no API
        cli_results = run_cli_batch(
            cli_conditions,
            test_cases,
            config,
            scorer,
            completed,
            total_calls,
            update_progress,
            results_lock,
            dry_run,
        )

    # Merge results (checkpoint results + new CLI + new API)
    all_results = results + cli_results + api_results

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
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n✓ Results saved: {final_file}")
    print(f"  Total results: {len(all_results)}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run full gating benchmark")
    parser.add_argument("--test-cases", type=Path,
                        default=PROJECT_ROOT / "data" / "verified",
                        help="Directory with test cases")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock API calls")
    parser.add_argument("--estimate", action="store_true",
                        help="Only show cost estimate")
    parser.add_argument("--n-bootstrap", type=int, default=10,
                        help="Number of bootstrap runs")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judge evaluation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--parallel-workers", type=int, default=5,
                        help="Number of parallel workers for API calls")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation")
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_bootstrap=args.n_bootstrap,
        enable_judge=not args.no_judge,
        parallel_workers=args.parallel_workers,
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

    # Show resume info
    if args.resume:
        print("\n📂 Resume mode: Will load from latest checkpoint")

    # Confirm
    if not args.dry_run and not args.yes:
        print()
        response = input("Proceed with benchmark? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            return

    # Run
    run_benchmark(config, test_cases_dir, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
