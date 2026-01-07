"""Evaluation framework for MCP ablation study."""

from .test_cases import (
    TestCaseType,
    PanelDesignTestCase,
    TestSuite,
    generate_in_distribution_cases,
    generate_out_of_distribution_cases,
    generate_adversarial_cases,
    build_ablation_test_suite,
)
from .conditions import (
    RetrievalMode,
    ExperimentalCondition,
    CONDITIONS,
)
from .runner import (
    TrialResult,
    ExperimentResults,
    AblationRunner,
)
from .analysis import (
    compute_mcp_lift,
    analyze_by_case_type,
    find_optimal_retrieval_weight,
    generate_report,
)

__all__ = [
    "TestCaseType",
    "PanelDesignTestCase",
    "TestSuite",
    "generate_in_distribution_cases",
    "generate_out_of_distribution_cases",
    "generate_adversarial_cases",
    "build_ablation_test_suite",
    "RetrievalMode",
    "ExperimentalCondition",
    "CONDITIONS",
    "TrialResult",
    "ExperimentResults",
    "AblationRunner",
    "compute_mcp_lift",
    "analyze_by_case_type",
    "find_optimal_retrieval_weight",
    "generate_report",
]
