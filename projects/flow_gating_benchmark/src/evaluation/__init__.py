"""Phase 2: Evaluation framework - metrics and scoring for gating predictions."""

from .metrics import (
    CELL_TYPE_SYNONYMS,
    POPULATION_REQUIRED_MARKERS,
    EvaluationResult,
    compute_critical_gate_recall,
    compute_hallucination_rate,
    compute_hierarchy_f1,
    compute_structure_accuracy,
    normalize_gate_name,
    normalize_gate_semantic,
)
from .response_parser import ParseResult, parse_llm_response
from .scorer import GatingScorer, score_prediction
from .task_failure import (
    TaskFailureResult,
    TaskFailureType,
    compute_task_failure_rate,
    detect_task_failure,
)

__all__ = [
    "compute_hierarchy_f1",
    "compute_structure_accuracy",
    "compute_critical_gate_recall",
    "compute_hallucination_rate",
    "EvaluationResult",
    "normalize_gate_name",
    "normalize_gate_semantic",
    "CELL_TYPE_SYNONYMS",
    "POPULATION_REQUIRED_MARKERS",
    "GatingScorer",
    "score_prediction",
    "parse_llm_response",
    "ParseResult",
    "TaskFailureType",
    "TaskFailureResult",
    "detect_task_failure",
    "compute_task_failure_rate",
]
