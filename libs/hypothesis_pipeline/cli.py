"""
Command-line interface for hypothesis pipelines.

Supports:
- Running experiments from config files
- Config templates with CLI overrides
- Experiment tracking and comparison
- Adding conclusions to experiments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import ConfigLoader, PipelineConfig, parse_cli_overrides
from .tracker import ExperimentTracker


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hypothesis Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment from config file
  hypothesis-pipeline run --config configs/ablation.yaml

  # Run with CLI overrides
  hypothesis-pipeline run --config base.yaml --models=gpt-4o --reasoning_types=cot,wot

  # List experiments
  hypothesis-pipeline list --tags benchmark --status completed

  # Compare experiments
  hypothesis-pipeline compare exp1 exp2 --metrics hierarchy_f1

  # Add conclusion
  hypothesis-pipeline conclude exp_id --outcome success --summary "CoT outperformed direct"

  # Generate report
  hypothesis-pipeline report exp_id
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to config file or config name",
    )
    run_parser.add_argument(
        "--config-dir",
        default="./configs",
        help="Directory containing config templates",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (overrides config)",
    )
    run_parser.add_argument(
        "--experiments-dir",
        default="./experiments",
        help="Directory for experiment tracking",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument(
        "--experiments-dir",
        default="./experiments",
        help="Directory containing experiments",
    )
    list_parser.add_argument(
        "--tags", "-t",
        nargs="+",
        help="Filter by tags",
    )
    list_parser.add_argument(
        "--status", "-s",
        choices=["running", "completed"],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--name", "-n",
        help="Filter by name (substring match)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment IDs to compare",
    )
    compare_parser.add_argument(
        "--experiments-dir",
        default="./experiments",
        help="Directory containing experiments",
    )
    compare_parser.add_argument(
        "--metrics", "-m",
        nargs="+",
        help="Specific metrics to compare",
    )
    compare_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # Conclude command
    conclude_parser = subparsers.add_parser("conclude", help="Add conclusion to experiment")
    conclude_parser.add_argument(
        "experiment_id",
        help="Experiment ID",
    )
    conclude_parser.add_argument(
        "--experiments-dir",
        default="./experiments",
        help="Directory containing experiments",
    )
    conclude_parser.add_argument(
        "--outcome", "-o",
        choices=["success", "partial", "failed", "inconclusive"],
        default="inconclusive",
        help="Experiment outcome",
    )
    conclude_parser.add_argument(
        "--summary", "-s",
        required=True,
        help="Brief summary of conclusions",
    )
    conclude_parser.add_argument(
        "--findings", "-f",
        nargs="+",
        help="Key findings",
    )
    conclude_parser.add_argument(
        "--next-steps",
        nargs="+",
        help="Suggested next steps",
    )
    conclude_parser.add_argument(
        "--notes",
        help="Additional notes",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate experiment report")
    report_parser.add_argument(
        "experiment_id",
        help="Experiment ID",
    )
    report_parser.add_argument(
        "--experiments-dir",
        default="./experiments",
        help="Directory containing experiments",
    )
    report_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )

    # Show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument(
        "experiment_id",
        help="Experiment ID",
    )
    show_parser.add_argument(
        "--experiments-dir",
        default="./experiments",
        help="Directory containing experiments",
    )

    # Create config command
    create_parser = subparsers.add_parser("create-config", help="Create a config template")
    create_parser.add_argument(
        "name",
        help="Config name",
    )
    create_parser.add_argument(
        "--template",
        choices=["minimal", "ablation", "full"],
        default="ablation",
        help="Template type",
    )
    create_parser.add_argument(
        "--output", "-o",
        help="Output file (default: configs/<name>.yaml)",
    )

    # Parse known args (allow unknown for overrides)
    args, unknown = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        return 1

    # Parse override args
    overrides = parse_cli_overrides(unknown)

    # Dispatch
    if args.command == "run":
        return cmd_run(args, overrides)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "conclude":
        return cmd_conclude(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "create-config":
        return cmd_create_config(args)

    return 0


def cmd_run(args: argparse.Namespace, overrides: dict[str, Any]) -> int:
    """Run an experiment."""
    # Load config
    loader = ConfigLoader(args.config_dir)

    try:
        config = loader.load(args.config, overrides)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nAvailable configs: {', '.join(loader.list_configs())}", file=sys.stderr)
        return 1

    # Apply output dir override
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Configuration:")
        print("=" * 60)
        print(config.to_yaml())
        print("=" * 60)
        print(f"\nWould generate {_count_conditions(config)} conditions")
        return 0

    # Initialize tracker
    tracker = ExperimentTracker(args.experiments_dir)

    # Start tracking
    metadata = tracker.start_experiment(config.to_dict())
    print(f"Started experiment: {metadata.experiment_id}")

    # Import pipeline here to avoid circular imports
    from .pipeline import HypothesisPipeline

    # This is where you'd create your evaluator and trial inputs
    # For now, we print instructions
    print("\nTo run the pipeline, create your evaluator and trial inputs:")
    print("""
from hypothesis_pipeline import HypothesisPipeline

# Your custom evaluator and trial inputs
pipeline = HypothesisPipeline(config, evaluator, trial_inputs)
results = pipeline.run()

# Save results to tracker
tracker.save_results(metadata, results)
""")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List experiments."""
    tracker = ExperimentTracker(args.experiments_dir)

    experiments = tracker.list_experiments(
        tags=args.tags,
        status=args.status,
        name_contains=args.name,
    )

    if not experiments:
        print("No experiments found.")
        return 0

    if args.json:
        print(json.dumps(experiments, indent=2))
    else:
        print(f"\n{'ID':<50} {'Name':<20} {'Status':<12} {'Timestamp':<25}")
        print("-" * 110)
        for exp in experiments:
            print(
                f"{exp['experiment_id']:<50} "
                f"{exp.get('name', ''):<20} "
                f"{exp.get('status', ''):<12} "
                f"{exp.get('timestamp', '')[:19]:<25}"
            )

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare experiments."""
    tracker = ExperimentTracker(args.experiments_dir)

    comparison = tracker.compare_experiments(
        args.experiments,
        metrics=args.metrics,
    )

    if args.json:
        print(json.dumps(comparison, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPARISON")
        print("=" * 60)

        # Show experiments
        for exp in comparison["experiments"]:
            print(f"\n{exp['experiment_id']}:")
            print(f"  Name: {exp['name']}")
            print(f"  Hypothesis: {exp.get('hypothesis', 'N/A')}")
            for metric, value in exp.get("metrics", {}).items():
                print(f"  {metric}: {value:.3f}")

        # Show best by metric
        if comparison["best_by_metric"]:
            print("\n" + "-" * 40)
            print("Best by Metric:")
            for metric, best in comparison["best_by_metric"].items():
                print(f"  {metric}: {best['experiment_id']} ({best['value']:.3f})")

    return 0


def cmd_conclude(args: argparse.Namespace) -> int:
    """Add conclusion to experiment."""
    tracker = ExperimentTracker(args.experiments_dir)

    try:
        conclusion = tracker.add_conclusion(
            args.experiment_id,
            summary=args.summary,
            outcome=args.outcome,
            findings=args.findings,
            next_steps=args.next_steps,
            notes=args.notes or "",
        )
        print(f"Conclusion added to {args.experiment_id}")
        print(f"  Outcome: {conclusion.outcome}")
        print(f"  Summary: {conclusion.summary}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate experiment report."""
    tracker = ExperimentTracker(args.experiments_dir)

    report = tracker.generate_report(args.experiment_id)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show experiment details."""
    tracker = ExperimentTracker(args.experiments_dir)

    record = tracker.get_experiment(args.experiment_id)
    if not record:
        print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {record.metadata.name}")
    print("=" * 60)
    print(f"ID: {record.metadata.experiment_id}")
    print(f"Timestamp: {record.metadata.timestamp}")
    print(f"Git: {record.metadata.git_commit} ({record.metadata.git_branch})")
    print(f"Dirty: {record.metadata.git_dirty}")

    if record.metadata.hypothesis:
        print(f"\nHypothesis: {record.metadata.hypothesis}")

    print("\nSummary:")
    print(f"  Trials: {record.results_summary.get('n_trials', 0)}")
    print(f"  Success Rate: {record.results_summary.get('overall_success_rate', 0):.1%}")
    print(f"  Best Condition: {record.results_summary.get('best_condition', 'N/A')}")

    if record.conclusion:
        print(f"\nConclusion ({record.conclusion.outcome}):")
        print(f"  {record.conclusion.summary}")

    print(f"\nFiles:")
    print(f"  Config: {record.config_file}")
    print(f"  Results: {record.results_file}")

    return 0


def cmd_create_config(args: argparse.Namespace) -> int:
    """Create a config template."""
    from .config import create_minimal_config, create_ablation_config, create_full_config

    templates = {
        "minimal": create_minimal_config,
        "ablation": create_ablation_config,
        "full": create_full_config,
    }

    config = templates[args.template](args.name)

    output = args.output or f"configs/{args.name}.yaml"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    config.save(output)
    print(f"Created config: {output}")

    return 0


def _count_conditions(config: PipelineConfig) -> int:
    """Count total conditions from config."""
    return (
        len(config.models) *
        len(config.reasoning_types) *
        len(config.context_levels) *
        len(config.rag_modes) *
        len(config.tool_configs)
    )


if __name__ == "__main__":
    sys.exit(main())
