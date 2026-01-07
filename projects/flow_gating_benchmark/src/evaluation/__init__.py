"""Phase 2: Evaluation framework - metrics and scoring for gating predictions."""

from .metrics import (
    compute_hierarchy_f1,
    compute_structure_accuracy,
    compute_critical_gate_recall,
    compute_hallucination_rate,
    EvaluationResult,
    normalize_gate_name,
    normalize_gate_semantic,
    CELL_TYPE_SYNONYMS,
    POPULATION_REQUIRED_MARKERS,
)
from .scorer import GatingScorer, score_prediction
from .response_parser import parse_llm_response, ParseResult

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
]
