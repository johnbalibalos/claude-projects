"""
CLI entry point for results_processor.

Usage:
    python -m results_processor export results/*.json
    python -m results_processor summarize results/experiment.json
"""

import argparse
import sys
from pathlib import Path

from .exporter import ResultsExporter


def cmd_export(args):
    """Export JSON files to CSV."""
    exporter = ResultsExporter()

    for json_path in args.files:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"File not found: {json_path}")
            continue

        try:
            output_path = exporter.export_to_csv(json_path, args.output)
            print(f"Exported: {json_path} -> {output_path}")
        except Exception as e:
            print(f"Error processing {json_path}: {e}")


def cmd_summarize(args):
    """Generate summary from JSON file."""
    exporter = ResultsExporter()

    json_path = Path(args.file)
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return 1

    try:
        output_path = exporter.generate_summary(json_path, args.output, args.title)
        print(f"Summary saved to: {output_path}")

        # Also print to stdout if requested
        if args.print:
            with open(output_path) as f:
                print()
                print(f.read())
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Process experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export JSON results to CSV",
    )
    export_parser.add_argument(
        "files",
        nargs="+",
        help="JSON files to export",
    )
    export_parser.add_argument(
        "-o", "--output",
        help="Output CSV path (default: same name with .csv)",
    )
    export_parser.set_defaults(func=cmd_export)

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Generate summary from JSON results",
    )
    summarize_parser.add_argument(
        "file",
        help="JSON file to summarize",
    )
    summarize_parser.add_argument(
        "-o", "--output",
        help="Output text path",
    )
    summarize_parser.add_argument(
        "-t", "--title",
        help="Summary title",
    )
    summarize_parser.add_argument(
        "-p", "--print",
        action="store_true",
        help="Also print summary to stdout",
    )
    summarize_parser.set_defaults(func=cmd_summarize)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
