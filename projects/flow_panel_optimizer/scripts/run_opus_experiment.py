#!/usr/bin/env python3
"""
Run panel design ablation study using Claude Opus.

Evaluates Claude Opus's ability to design flow cytometry panels across:
- Baseline (no tools, no retrieval)
- MCP tools only
- Retrieval only
- MCP + Retrieval

Saves results to results/ directory.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flow_panel_optimizer.evaluation.runner import AblationRunner, ExperimentResults
from flow_panel_optimizer.evaluation.test_cases import (
    build_ablation_test_suite,
    TestCaseType,
)
from flow_panel_optimizer.evaluation.conditions import CORE_CONDITIONS


def main():
    """Run ablation study and save results."""
    print("=" * 60)
    print("FLOW PANEL OPTIMIZER - OPUS ABLATION STUDY")
    print("=" * 60)
    print()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Build test suite with representative cases
    print("Building test suite...")
    suite = build_ablation_test_suite(
        n_in_dist=6,      # OMIP-derived panels
        n_near_dist=4,    # Modified OMIP panels
        n_out_dist=4,     # Novel combinations
        n_adversarial=2,  # Edge cases
    )

    print(f"Test suite: {suite.name}")
    print(f"  Total cases: {len(suite.test_cases)}")
    print(f"  - In-distribution: {len(suite.filter_by_type(TestCaseType.IN_DISTRIBUTION))}")
    print(f"  - Near-distribution: {len(suite.filter_by_type(TestCaseType.NEAR_DISTRIBUTION))}")
    print(f"  - Out-of-distribution: {len(suite.filter_by_type(TestCaseType.OUT_OF_DISTRIBUTION))}")
    print(f"  - Adversarial: {len(suite.filter_by_type(TestCaseType.ADVERSARIAL))}")
    print()

    # Print conditions
    print("Experimental conditions:")
    for cond in CORE_CONDITIONS:
        print(f"  - {cond.name}: {cond.description}")
    print()

    # Initialize runner with Opus
    runner = AblationRunner(
        model="claude-opus-4-20250514",
        max_concurrent=2  # More conservative for Opus (higher cost/latency)
    )

    # Run study
    print(f"Running {len(suite.test_cases)} test cases × {len(CORE_CONDITIONS)} conditions...")
    print("-" * 60)

    results = runner.run_full_study(
        test_suite=suite,
        conditions=CORE_CONDITIONS,
        verbose=True
    )

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Save raw results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"opus_ablation_{timestamp}.json"
    results.to_json(results_file)
    print(f"\nRaw results saved to: {results_file}")

    # Calculate and print summary
    df = results.to_dataframe()

    print("\n--- By Condition ---")
    condition_summary = df.groupby("condition").agg({
        "accuracy": ["mean", "std", "count"],
        "complexity_index": "mean",
        "ci_improvement": "mean",
        "latency": "mean",
        "tool_calls": "sum"
    }).round(3)
    print(condition_summary)

    print("\n--- By Case Type ---")
    case_type_summary = df.groupby(["condition", "case_type"]).agg({
        "accuracy": "mean",
        "complexity_index": "mean",
    }).round(3)
    print(case_type_summary)

    # Save summary
    summary_file = results_dir / f"opus_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("FLOW PANEL OPTIMIZER - OPUS ABLATION STUDY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model: claude-opus-4-20250514\n")
        f.write(f"Test cases: {len(suite.test_cases)}\n")
        f.write(f"Conditions: {len(CORE_CONDITIONS)}\n\n")

        f.write("RESULTS BY CONDITION\n")
        f.write("-" * 60 + "\n")
        f.write(condition_summary.to_string())
        f.write("\n\n")

        f.write("RESULTS BY CASE TYPE\n")
        f.write("-" * 60 + "\n")
        f.write(case_type_summary.to_string())
        f.write("\n\n")

        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 60 + "\n")

        baseline = df[df["condition"] == "baseline"]["accuracy"].mean()
        mcp_only = df[df["condition"] == "mcp_only"]["accuracy"].mean()
        retrieval = df[df["condition"] == "retrieval_standard"]["accuracy"].mean()
        mcp_plus = df[df["condition"] == "mcp_plus_retrieval"]["accuracy"].mean()

        f.write(f"1. Baseline accuracy: {baseline:.1%}\n")
        f.write(f"2. MCP tools only: {mcp_only:.1%} ({(mcp_only-baseline)*100:+.1f}pp vs baseline)\n")
        f.write(f"3. Retrieval only: {retrieval:.1%} ({(retrieval-baseline)*100:+.1f}pp vs baseline)\n")
        f.write(f"4. MCP + Retrieval: {mcp_plus:.1%} ({(mcp_plus-baseline)*100:+.1f}pp vs baseline)\n")

        # Tool usage
        tool_calls = df[df["condition"] == "mcp_only"]["tool_calls"].sum()
        f.write(f"\n5. Total tool calls (MCP only): {tool_calls}\n")

        # Complexity index analysis
        f.write("\nCOMPLEXITY INDEX (lower is better):\n")
        for cond in ["baseline", "mcp_only", "retrieval_standard", "mcp_plus_retrieval"]:
            ci = df[df["condition"] == cond]["complexity_index"].mean()
            f.write(f"  {cond}: {ci:.2f}\n")

    print(f"Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    main()
