#!/usr/bin/env python3
"""
Pilot experiment: Test model consistency across temperatures.

Uses Anthropic API to test consistency at different temperature settings:
- temperature=0.0 (deterministic)
- temperature=0.5 (moderate sampling)
- temperature=1.0 (high sampling)

Usage:
    python scripts/run_temperature_pilot.py --n-runs 10 --models haiku sonnet opus
    python scripts/run_temperature_pilot.py --n-runs 3 --models sonnet  # Quick test
    python scripts/run_temperature_pilot.py --temps 0 0.5 1  # Specific temperatures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curation.schemas import TestCase
from experiments.llm_client import AnthropicClient
from experiments.prompts import build_prompt


# Model mapping (API model IDs)
API_MODELS = {
    "haiku": "claude-3-5-haiku-20241022",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}

# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    "haiku": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
}


@dataclass
class PilotResult:
    """Result from a single pilot run."""
    model: str
    temperature: float
    run_idx: int
    response: str
    response_length: int
    tokens_used: int
    elapsed_seconds: float


@dataclass
class PilotExperiment:
    """Container for pilot experiment results."""
    test_case_id: str
    n_runs: int
    models: list[str]
    temperatures: list[float]
    results: list[PilotResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_cost: float = 0.0

    def add_result(self, result: PilotResult):
        self.results.append(result)

    def to_dict(self) -> dict:
        return {
            "test_case_id": self.test_case_id,
            "n_runs": self.n_runs,
            "models": self.models,
            "temperatures": self.temperatures,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_cost": round(self.total_cost, 4),
            "results": [
                {
                    "model": r.model,
                    "temperature": r.temperature,
                    "run_idx": r.run_idx,
                    "response_length": r.response_length,
                    "tokens_used": r.tokens_used,
                    "elapsed_seconds": r.elapsed_seconds,
                    "response": r.response,
                }
                for r in self.results
            ],
        }


def compute_consistency(responses: list[str]) -> dict:
    """Compute consistency metrics for a set of responses."""
    if len(responses) < 2:
        return {"n_unique": len(responses), "consistency": 1.0, "all_same": True}

    unique = set(responses)
    n_unique = len(unique)
    n_total = len(responses)

    # Consistency: 1.0 if all same, 0.0 if all different
    consistency = 1.0 - (n_unique - 1) / (n_total - 1)

    return {
        "n_unique": n_unique,
        "n_total": n_total,
        "consistency": round(consistency, 3),
        "all_same": n_unique == 1,
        "all_different": n_unique == n_total,
    }


def analyze_results(experiment: PilotExperiment) -> dict:
    """Analyze experiment results for consistency patterns."""
    # Group by model and temperature
    by_model_temp = defaultdict(list)
    for r in experiment.results:
        key = (r.model, r.temperature)
        by_model_temp[key].append(r.response)

    analysis = {
        "by_model_temp": {},
        "summary": {
            "models": experiment.models,
            "temperatures": experiment.temperatures,
            "n_runs": experiment.n_runs,
            "total_cost": experiment.total_cost,
        },
    }

    for (model, temp), responses in by_model_temp.items():
        consistency = compute_consistency(responses)
        avg_length = sum(len(r) for r in responses) / len(responses)

        analysis["by_model_temp"][f"{model}_t{temp}"] = {
            "model": model,
            "temperature": temp,
            **consistency,
            "avg_response_length": round(avg_length, 0),
        }

    # Compare temperatures for each model
    comparisons = []
    for model in experiment.models:
        model_data = {
            "model": model,
            "by_temperature": {},
        }
        for temp in experiment.temperatures:
            key = f"{model}_t{temp}"
            if key in analysis["by_model_temp"]:
                model_data["by_temperature"][str(temp)] = analysis["by_model_temp"][key]["consistency"]

        comparisons.append(model_data)

    analysis["comparisons"] = comparisons

    return analysis


def estimate_cost(models: list[str], temperatures: list[float], n_runs: int, input_tokens: int = 1500, output_tokens: int = 9000) -> dict:
    """Estimate cost for the experiment."""
    costs = {}
    total = 0.0

    for model in models:
        input_price, output_price = MODEL_PRICING[model]
        n_calls = len(temperatures) * n_runs
        model_cost = n_calls * (input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000)
        costs[model] = round(model_cost, 2)
        total += model_cost

    return {"by_model": costs, "total": round(total, 2)}


def run_pilot(
    test_case_path: Path,
    models: list[str],
    temperatures: list[float],
    n_runs: int = 10,
    output_dir: Path | None = None,
    skip_confirm: bool = False,
) -> PilotExperiment:
    """Run the pilot experiment."""
    # Load test case
    with open(test_case_path) as f:
        test_case_data = json.load(f)
    test_case = TestCase(**test_case_data)

    # Build prompt
    prompt = build_prompt(
        test_case,
        template_name="cot",
        context_level="standard",
        reference="none",
    )

    # Estimate cost
    cost_est = estimate_cost(models, temperatures, n_runs)
    total_calls = len(models) * len(temperatures) * n_runs

    print(f"Temperature Consistency Pilot Experiment")
    print("=" * 60)
    print(f"  Test case: {test_case.test_case_id}")
    print(f"  Models: {models}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Runs per setting: {n_runs}")
    print(f"  Total API calls: {total_calls}")
    print()
    print("Estimated cost:")
    for model, cost in cost_est["by_model"].items():
        print(f"  {model}: ${cost:.2f}")
    print(f"  TOTAL: ${cost_est['total']:.2f}")
    print()

    if not skip_confirm:
        confirm = input("Proceed? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return None

    # Initialize experiment
    experiment = PilotExperiment(
        test_case_id=test_case.test_case_id,
        n_runs=n_runs,
        models=models,
        temperatures=temperatures,
        start_time=datetime.now().isoformat(),
    )

    print()
    print("Running experiment...")
    print("-" * 60)

    total_cost = 0.0

    # Run experiment
    for model_name in models:
        model_id = API_MODELS[model_name]
        client = AnthropicClient(model=model_id)
        input_price, output_price = MODEL_PRICING[model_name]

        for temp in temperatures:
            print(f"\n{model_name} @ temperature={temp}:")

            for run_idx in range(n_runs):
                start = time.time()

                try:
                    response = client.call(prompt, temperature=temp, max_tokens=8192)
                    content = response.content
                    tokens = response.tokens_used
                except Exception as e:
                    print(f"  Run {run_idx + 1}: ERROR - {e}")
                    content = f"ERROR: {e}"
                    tokens = 0

                elapsed = time.time() - start

                # Calculate cost for this call (rough estimate: 15% input, 85% output)
                est_input = int(tokens * 0.15)
                est_output = tokens - est_input
                call_cost = est_input * input_price / 1_000_000 + est_output * output_price / 1_000_000
                total_cost += call_cost

                result = PilotResult(
                    model=model_name,
                    temperature=temp,
                    run_idx=run_idx,
                    response=content,
                    response_length=len(content),
                    tokens_used=tokens,
                    elapsed_seconds=round(elapsed, 2),
                )
                experiment.add_result(result)

                # Progress indicator
                status = "✓" if not content.startswith("ERROR") else "✗"
                print(f"  Run {run_idx + 1}/{n_runs}: {status} ({elapsed:.1f}s, {len(content)} chars, {tokens} tok)")

    experiment.end_time = datetime.now().isoformat()
    experiment.total_cost = total_cost

    # Analyze results
    analysis = analyze_results(experiment)

    # Print summary
    print()
    print("=" * 60)
    print("CONSISTENCY ANALYSIS")
    print("=" * 60)
    print()

    print(f"{'Model':<10} {'Temp':<8} {'Unique':<8} {'Consistency':<12} {'All Same':<10}")
    print("-" * 60)

    for key, data in sorted(analysis["by_model_temp"].items()):
        print(f"{data['model']:<10} {data['temperature']:<8} {data['n_unique']:<8} {data['consistency']:<12.3f} {str(data['all_same']):<10}")

    print()
    print("Summary by model:")
    print("-" * 60)
    for comp in analysis["comparisons"]:
        temps_str = ", ".join(f"t{t}={c:.3f}" for t, c in comp["by_temperature"].items())
        print(f"  {comp['model']}: {temps_str}")

    print(f"\nActual cost: ${total_cost:.4f}")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results
        results_path = output_dir / "pilot_results.json"
        with open(results_path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Save analysis
        analysis_path = output_dir / "pilot_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {analysis_path}")

    return experiment


def main():
    parser = argparse.ArgumentParser(description="Run temperature consistency pilot")
    parser.add_argument(
        "--n-runs", type=int, default=10,
        help="Number of runs per temperature setting (default: 10)"
    )
    parser.add_argument(
        "--models", nargs="+", default=["haiku", "sonnet", "opus"],
        choices=["haiku", "sonnet", "opus"],
        help="Models to test (default: all three)"
    )
    parser.add_argument(
        "--temps", nargs="+", type=float, default=[0.0, 0.5, 1.0],
        help="Temperature settings to test (default: 0.0 0.5 1.0)"
    )
    parser.add_argument(
        "--test-case", type=str, default="data/verified/omip_095.json",
        help="Path to test case JSON (default: OMIP-095)"
    )
    parser.add_argument(
        "--output", type=str, default="results/temperature_pilot",
        help="Output directory (default: results/temperature_pilot)"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    test_case_path = project_root / args.test_case
    output_dir = project_root / args.output

    if not test_case_path.exists():
        print(f"Error: Test case not found: {test_case_path}")
        sys.exit(1)

    run_pilot(
        test_case_path=test_case_path,
        models=args.models,
        temperatures=args.temps,
        n_runs=args.n_runs,
        output_dir=output_dir,
        skip_confirm=args.yes,
    )


if __name__ == "__main__":
    main()
