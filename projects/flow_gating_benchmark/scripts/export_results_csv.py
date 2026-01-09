#!/usr/bin/env python3
"""
Export experiment results to CSV for analysis.

This is a thin wrapper around the shared results_processor library.
"""

import sys
from pathlib import Path

# Add libs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs"))

from results_processor import ResultsExporter, GATING_BENCHMARK_COLUMNS


def main():
    results_dir = Path(__file__).parent.parent / "results"

    # Find all experiment result files
    json_files = list(results_dir.glob("experiment_results_*.json"))

    if not json_files:
        print("No experiment result files found")
        return 1

    exporter = ResultsExporter(columns=GATING_BENCHMARK_COLUMNS)

    for json_path in json_files:
        output_path = exporter.export_to_csv(json_path)
        print(f"Exported {json_path.name} -> {output_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
