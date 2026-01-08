"""
Results processing utilities for experiment data.

This library provides tools for exporting experiment results to CSV
and generating summary reports.

Usage:
    from results_processor import ResultsExporter

    exporter = ResultsExporter()
    exporter.export_to_csv("results.json", "results.csv")
    exporter.generate_summary("results.json", "summary.txt")
"""

from .exporter import (
    ResultsExporter,
    export_to_csv,
    generate_summary,
    GATING_BENCHMARK_COLUMNS,
    PANEL_OPTIMIZER_COLUMNS,
)

__all__ = [
    "ResultsExporter",
    "export_to_csv",
    "generate_summary",
    "GATING_BENCHMARK_COLUMNS",
    "PANEL_OPTIMIZER_COLUMNS",
]
