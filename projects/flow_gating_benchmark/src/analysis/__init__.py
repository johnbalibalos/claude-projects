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

# Hypothesis testing modules
from .alien_cell import (
    AlienCellMapping,
    AlienCellResult,
    AlienCellTest,
    AlienCellTestCase,
    ALIEN_CELL_NAMES,
)
from .format_ablation import (
    FormatAblationAnalysis,
    FormatAblationResult,
    FormatAblationTest,
    FormattedPrompt,
    PromptFormat,
)
from .cognitive_refusal import (
    AggregateRefusalAnalysis,
    CognitiveRefusalAnalysis,
    CognitiveRefusalResult,
    CognitiveRefusalTest,
    PromptVariant,
    RefusalAnalysis,
    RefusalType,
    PROMPT_VARIANTS,
    REFUSAL_PATTERNS,
)

__all__ = [
    # Failure analysis
    "analyze_failures",
    "FailureCategory",
    "create_summary_plots",
    # Manual review
    "ManualReviewReportGenerator",
    "generate_manual_review_report",
    "OutlierThresholds",
    "is_outlier",
    # Alien Cell test
    "AlienCellMapping",
    "AlienCellResult",
    "AlienCellTest",
    "AlienCellTestCase",
    "ALIEN_CELL_NAMES",
    # Format Ablation test
    "FormatAblationAnalysis",
    "FormatAblationResult",
    "FormatAblationTest",
    "FormattedPrompt",
    "PromptFormat",
    # Cognitive Refusal test
    "AggregateRefusalAnalysis",
    "CognitiveRefusalAnalysis",
    "CognitiveRefusalResult",
    "CognitiveRefusalTest",
    "PromptVariant",
    "RefusalAnalysis",
    "RefusalType",
    "PROMPT_VARIANTS",
    "REFUSAL_PATTERNS",
]
