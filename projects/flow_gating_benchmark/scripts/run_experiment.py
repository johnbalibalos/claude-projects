#!/usr/bin/env python3
"""
Run gating strategy prediction benchmark.

Evaluates LLM ability to predict flow cytometry gating hierarchies
across different context levels and prompting strategies.

Usage:
    python scripts/run_experiment.py --model opus
    python scripts/run_experiment.py --model sonnet --report outliers
    python scripts/run_experiment.py --model sonnet --context standard --strategy cot
    python scripts/run_experiment.py --model haiku --dry-run --report full
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.manual_review_report import (
    OutlierThresholds,
    generate_manual_review_report,
)
from curation.schemas import TestCase
from experiments.conditions import (
    CONTEXT_LEVELS,
    MODELS,
    PROMPT_STRATEGIES,
    get_all_conditions,
)
from experiments.runner import run_experiment

# Model-specific configurations
MODEL_CONFIGS = {
    "opus": {
        "model_key": "claude-opus",
        "max_concurrent": 2,
    },
    "sonnet": {
        "model_key": "claude-sonnet",
        "max_concurrent": 3,
    },
    "haiku": {
        "model_key": "claude-haiku",
        "max_concurrent": 5,
    },
    "gpt-4o": {
        "model_key": "gpt-4o",
        "max_concurrent": 3,
    },
    "gpt-4o-mini": {
        "model_key": "gpt-4o-mini",
        "max_concurrent": 5,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gating strategy prediction benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Report Levels:
    summary   - Overall metrics table + worst performers (default)
    outliers  - Summary + detailed reports for outliers (F1 < 30%, halluc > 20%)
    full      - Detailed report for every test case

Examples:
    python scripts/run_experiment.py --model opus
    python scripts/run_experiment.py --model sonnet --report outliers
    python scripts/run_experiment.py --model sonnet --context standard
    python scripts/run_experiment.py --model haiku --strategy cot --dry-run
        """,
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="sonnet",
        help="Model to use (default: sonnet)",
    )
    parser.add_argument(
        "--context",
        choices=CONTEXT_LEVELS + ["all"],
        default="all",
        help="Context level (default: all)",
    )
    parser.add_argument(
        "--strategy",
        choices=PROMPT_STRATEGIES + ["all"],
        default="all",
        help="Prompting strategy (default: all)",
    )
    parser.add_argument(
        "--report",
        choices=["summary", "outliers", "full", "none"],
        default="summary",
        help="Report detail level (default: summary)",
    )
    parser.add_argument(
        "--outlier-f1",
        type=float,
        default=0.3,
        help="F1 threshold for outlier detection (default: 0.3)",
    )
    parser.add_argument(
        "--outlier-halluc",
        type=float,
        default=0.2,
        help="Hallucination threshold for outlier detection (default: 0.2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation (for hooks)",
    )
    return parser.parse_args()


def load_test_cases(test_cases_dir: Path) -> dict[str, TestCase]:
    """Load all test cases into a dictionary."""
    test_cases = {}
    for path in test_cases_dir.glob("omip_*.json"):
        with open(path) as f:
            data = json.load(f)

        # Fix null fluorophores
        for entry in data.get("panel", {}).get("entries", []):
            if entry.get("fluorophore") is None:
                entry["fluorophore"] = "Unknown"

        try:
            tc = TestCase(**data)
            test_cases[tc.omip_id] = tc
        except Exception as e:
            print(f"Warning: Could not load {path.name}: {e}")

    return test_cases


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    model_key = config["model_key"]

    print("=" * 60)
    print(f"FLOW GATING BENCHMARK - {args.model.upper()} EVALUATION")
    print("=" * 60)
    print()

    # Check API key
    api_key_var = "OPENAI_API_KEY" if "gpt" in args.model else "ANTHROPIC_API_KEY"
    if not os.environ.get(api_key_var):
        print(f"ERROR: {api_key_var} not set")
        sys.exit(1)

    # Paths
    project_dir = Path(__file__).parent.parent
    test_cases_dir = project_dir / "data" / "ground_truth"
    output_dir = project_dir / "results"

    # Count test cases
    test_case_files = list(test_cases_dir.glob("omip_*.json"))
    print(f"Found {len(test_case_files)} OMIP test cases")
    print()

    # Build conditions based on args
    context_levels = CONTEXT_LEVELS if args.context == "all" else [args.context]
    prompt_strategies = PROMPT_STRATEGIES if args.strategy == "all" else [args.strategy]

    conditions = get_all_conditions(
        models=[model_key],
        context_levels=context_levels,
        prompt_strategies=prompt_strategies,
    )

    print("Experimental conditions:")
    for cond in conditions:
        print(f"  - {cond.name}: {cond.context_level} context, {cond.prompt_strategy} prompting")
    print()

    print(f"Model: {MODELS[model_key]}")
    print(f"Report level: {args.report}")
    print()

    if args.dry_run:
        print("DRY RUN - exiting without running experiment")
        return

    # Run experiment
    print(f"Running {len(test_case_files)} test cases × {len(conditions)} conditions...")
    print("-" * 60)

    result = run_experiment(
        test_cases_dir=str(test_cases_dir),
        output_dir=str(output_dir),
        name=f"{args.model}_gating_benchmark",
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

            print("\nOverall Metrics:")
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

    # Generate manual review report
    if args.report != "none" and result.results:
        print("\nGenerating manual review report...")

        # Load test cases for report
        test_cases = load_test_cases(test_cases_dir)

        # Set up outlier thresholds
        thresholds = OutlierThresholds(
            min_f1=args.outlier_f1,
            max_hallucination=args.outlier_halluc,
        )

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"manual_review_{args.model}_{timestamp}.md"

        generate_manual_review_report(
            results=result.results,
            test_cases=test_cases,
            level=args.report,
            output_path=report_path,
            thresholds=thresholds,
        )

        print(f"Report saved to: {report_path}")

        # Count outliers for info
        from analysis.manual_review_report import is_outlier
        outliers = [r for r in result.results if is_outlier(r, thresholds)]
        if outliers:
            print(f"Found {len(outliers)} outlier(s) (F1 < {args.outlier_f1:.0%} or halluc > {args.outlier_halluc:.0%})")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
