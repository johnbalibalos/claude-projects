#!/usr/bin/env python3
"""
Run MCP vs Retrieval ablation study.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python run_ablation_study.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from flow_panel_optimizer.evaluation.test_cases import build_ablation_test_suite, TestCaseType
from flow_panel_optimizer.evaluation.conditions import CONDITIONS, CORE_CONDITIONS
from flow_panel_optimizer.evaluation.runner import AblationRunner, ExperimentResults
from flow_panel_optimizer.evaluation.analysis import (
    compute_mcp_lift,
    analyze_by_case_type,
    find_optimal_retrieval_weight,
    generate_report,
    quick_summary,
)


def run_sonnet_ablation(
    n_in_dist: int = 20,
    n_out_dist: int = 20,
    n_adversarial: int = 10,
    use_core_conditions: bool = False,
    model: str = "claude-sonnet-4-20250514"
):
    """Run full ablation study with specified model."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    model_display = model.upper().replace("CLAUDE-", "").replace("-", " ")
    print("=" * 80)
    print(f"MCP vs RETRIEVAL ABLATION STUDY - {model_display}")
    print("=" * 80)

    # Build test suite
    print(f"\nGenerating test cases...")
    suite = build_ablation_test_suite(
        n_in_dist=n_in_dist,
        n_out_dist=n_out_dist,
        n_adversarial=n_adversarial
    )

    print(f"  In-distribution: {len(suite.filter_by_type(TestCaseType.IN_DISTRIBUTION))}")
    print(f"  Out-of-distribution: {len(suite.filter_by_type(TestCaseType.OUT_OF_DISTRIBUTION))}")
    print(f"  Adversarial: {len(suite.filter_by_type(TestCaseType.ADVERSARIAL))}")
    print(f"  Total: {len(suite.test_cases)}")

    # Select conditions
    conditions = CORE_CONDITIONS if use_core_conditions else CONDITIONS
    print(f"\nConditions to test: {len(conditions)}")
    for c in conditions:
        print(f"  - {c.name}: {c.description}")

    total_calls = len(suite.test_cases) * len(conditions)
    print(f"\nTotal API calls: {total_calls}")

    # Estimate cost based on model
    if "opus" in model.lower():
        cost_per_call = 0.20  # Rough estimate for Opus
        model_name = "Opus"
    else:
        cost_per_call = 0.07  # Sonnet estimate
        model_name = "Sonnet"

    print(f"Estimated cost: ${total_calls * cost_per_call:.2f} ({model_name})")

    # Initialize runner
    runner = AblationRunner(
        api_key=api_key,
        model=model
    )

    # Run study
    print("\n" + "=" * 80)
    print("RUNNING ABLATION STUDY")
    print("=" * 80 + "\n")

    results = runner.run_full_study(suite, conditions, verbose=True)

    # Save raw results
    output_dir = Path(__file__).parent
    model_short = model.split("-")[1] if "-" in model else "model"  # Extract "sonnet" or "opus"
    results_file = output_dir / f"ablation_results_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results.to_json(results_file)
    print(f"\nResults saved to: {results_file}")

    # Generate and print report
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    report = generate_report(results)
    print(report)

    # Save report
    report_file = output_dir / f"ablation_report_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Print quick summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(quick_summary(results))

    return results


def calculate_opus_sample_size(sonnet_results: ExperimentResults) -> dict:
    """
    Calculate optimal sample size for Opus comparison based on Sonnet results.

    Uses effect size from Sonnet to determine power requirements.
    """
    # Get Sonnet effect size (MCP lift)
    mcp_lift = compute_mcp_lift(sonnet_results)

    if "error" in mcp_lift:
        return {"error": mcp_lift["error"]}

    effect_size = abs(mcp_lift["mcp_lift"])

    # Calculate variance from Sonnet data
    mcp_cis = [t.complexity_index for t in sonnet_results.trials if t.condition_name == "mcp_only"]
    baseline_cis = [t.complexity_index for t in sonnet_results.trials if t.condition_name == "baseline"]

    if not mcp_cis or not baseline_cis:
        return {"error": "Missing condition data"}

    import math

    # Pooled standard deviation
    mcp_var = sum((x - sum(mcp_cis)/len(mcp_cis))**2 for x in mcp_cis) / len(mcp_cis)
    baseline_var = sum((x - sum(baseline_cis)/len(baseline_cis))**2 for x in baseline_cis) / len(baseline_cis)
    pooled_std = math.sqrt((mcp_var + baseline_var) / 2)

    # Cohen's d
    mean_diff = abs(sum(baseline_cis)/len(baseline_cis) - sum(mcp_cis)/len(mcp_cis))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Sample size for 80% power, alpha=0.05 (two-tailed)
    # n = 2 * ((z_alpha + z_beta) / d)^2
    # For 80% power: z_alpha=1.96, z_beta=0.84
    if cohens_d > 0:
        n_per_group = math.ceil(2 * ((1.96 + 0.84) / cohens_d) ** 2)
    else:
        n_per_group = 50  # Default

    # Cap at reasonable limits
    n_per_group = min(max(n_per_group, 10), 50)

    # Recommended conditions for Opus (focus on key comparison)
    recommended_conditions = ["baseline", "mcp_only"]

    # Cost estimate
    tokens_per_call = 15000  # Opus with MCP tools
    cost_per_call = tokens_per_call / 1_000_000 * (15 + 75)  # Opus pricing
    total_cost = n_per_group * len(recommended_conditions) * cost_per_call

    return {
        "sonnet_effect_size": round(effect_size, 4),
        "cohens_d": round(cohens_d, 4),
        "pooled_std": round(pooled_std, 4),
        "recommended_n_per_condition": n_per_group,
        "recommended_conditions": recommended_conditions,
        "total_opus_calls": n_per_group * len(recommended_conditions),
        "estimated_opus_cost": f"${total_cost:.2f}",
        "statistical_power": "80%",
        "alpha": 0.05
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MCP ablation study")
    parser.add_argument("--in-dist", type=int, default=20, help="Number of in-distribution cases")
    parser.add_argument("--out-dist", type=int, default=20, help="Number of out-of-distribution cases")
    parser.add_argument("--adversarial", type=int, default=10, help="Number of adversarial cases")
    parser.add_argument("--core-only", action="store_true", help="Use only core conditions (4 instead of 8)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Model to use (e.g., claude-opus-4-20250514)")

    args = parser.parse_args()

    results = run_sonnet_ablation(
        n_in_dist=args.in_dist,
        n_out_dist=args.out_dist,
        n_adversarial=args.adversarial,
        use_core_conditions=args.core_only,
        model=args.model
    )

    # Calculate Opus requirements
    print("\n" + "=" * 80)
    print("OPUS SAMPLE SIZE CALCULATION")
    print("=" * 80)

    opus_calc = calculate_opus_sample_size(results)
    print(json.dumps(opus_calc, indent=2))
