"""Phase 4: Analysis - result analysis and visualization."""

# Import with error handling for optional dependencies
try:
    from .failure_analysis import FailureCategory, analyze_failures
except ImportError:
    analyze_failures = None
    FailureCategory = None

try:
    from .visualization import create_summary_plots
except ImportError:
    create_summary_plots = None

from .manual_review_report import (
    ManualReviewReportGenerator,
    OutlierThresholds,
    generate_manual_review_report,
    is_outlier,
)

__all__ = [
    "analyze_failures",
    "FailureCategory",
    "create_summary_plots",
    "ManualReviewReportGenerator",
    "generate_manual_review_report",
    "OutlierThresholds",
    "is_outlier",
]
