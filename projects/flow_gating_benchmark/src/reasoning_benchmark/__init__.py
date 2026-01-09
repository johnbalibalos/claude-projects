"""
Reasoning Benchmark for Flow Cytometry Gating.

This module tests genuine biological reasoning, not pattern matching.

Test Types:
1. Lineage Negative (Dump Channel) - Defining cells by exclusion
2. Biological Impossibility - Detecting doublets/artifacts
3. Context Switch - Adapting to tissue-specific constraints
4. FMO Logic - Understanding gating boundary determination
5. Panel Subset Design - Optimizing under instrument constraints
"""

from .schemas import (
    ReasoningTestCase,
    ReasoningTestType,
    ExpectedBehavior,
    EvaluationCriteria,
)
from .test_cases import (
    LINEAGE_NEGATIVE_TESTS,
    BIOLOGICAL_IMPOSSIBILITY_TESTS,
    CONTEXT_SWITCH_TESTS,
    FMO_LOGIC_TESTS,
    PANEL_SUBSET_TESTS,
    get_all_reasoning_tests,
)
from .evaluator import ReasoningEvaluator

__all__ = [
    "ReasoningTestCase",
    "ReasoningTestType",
    "ExpectedBehavior",
    "EvaluationCriteria",
    "ReasoningEvaluator",
    "LINEAGE_NEGATIVE_TESTS",
    "BIOLOGICAL_IMPOSSIBILITY_TESTS",
    "CONTEXT_SWITCH_TESTS",
    "FMO_LOGIC_TESTS",
    "PANEL_SUBSET_TESTS",
    "get_all_reasoning_tests",
]
