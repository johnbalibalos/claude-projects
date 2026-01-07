#!/usr/bin/env python3
"""
Run gating strategy prediction benchmark using Claude Opus.

Evaluates Claude Opus's ability to predict flow cytometry gating hierarchies
across different context levels and prompting strategies.

Saves results to results/ directory.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.runner import run_experiment, ExperimentConfig
from experiments.conditions import get_opus_ablation_conditions, ExperimentCondition


def main():
    """Run gating benchmark and save results."""
    print("=" * 60)
    print("FLOW GATING BENCHMARK - OPUS EVALUATION")
    print("=" * 60)
    print()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Paths
    project_dir = Path(__file__).parent.parent
    test_cases_dir = project_dir / "data" / "ground_truth"
    output_dir = project_dir / "results"

    # Count test cases
    test_case_files = list(test_cases_dir.glob("omip_*.json"))
    print(f"Found {len(test_case_files)} OMIP test cases")
    print()

    # Get conditions (Opus ablation: 3 context × 2 strategy = 6 conditions)
    conditions = get_opus_ablation_conditions()

    print("Experimental conditions:")
    for cond in conditions:
        print(f"  - {cond.name}: {cond.context_level} context, {cond.prompt_strategy} prompting")
    print()

    # Run experiment
    print(f"Running {len(test_case_files)} test cases × {len(conditions)} conditions...")
    print("-" * 60)

    result = run_experiment(
        test_cases_dir=str(test_cases_dir),
        output_dir=str(output_dir),
        name="opus_gating_benchmark",
        conditions=conditions,
        dry_run=False,
    )

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total evaluations: {len(result.results)}")
    print(f"Errors: {len(result.errors)}")

    # Calculate summary statistics
    if result.results:
        successful = [r for r in result.results if r.parse_success]

        if successful:
            avg_f1 = sum(r.hierarchy_f1 for r in successful) / len(successful)
            avg_structure = sum(r.structure_accuracy for r in successful) / len(successful)
            avg_critical = sum(r.critical_gate_recall for r in successful) / len(successful)
            avg_hallucination = sum(r.hallucination_rate for r in successful) / len(successful)
            parse_rate = len(successful) / len(result.results)

            print(f"\nOverall Metrics:")
            print(f"  Hierarchy F1: {avg_f1:.3f}")
            print(f"  Structure Accuracy: {avg_structure:.3f}")
            print(f"  Critical Gate Recall: {avg_critical:.3f}")
            print(f"  Hallucination Rate: {avg_hallucination:.3f}")
            print(f"  Parse Success Rate: {parse_rate:.1%}")

            # By condition
            print("\nBy Condition:")
            for cond in conditions:
                cond_results = [r for r in successful if r.condition == cond.name]
                if cond_results:
                    cond_f1 = sum(r.hierarchy_f1 for r in cond_results) / len(cond_results)
                    print(f"  {cond.name}: F1={cond_f1:.3f} (n={len(cond_results)})")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
