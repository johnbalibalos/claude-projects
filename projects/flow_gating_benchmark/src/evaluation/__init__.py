"""Phase 2: Evaluation framework - metrics and scoring for gating predictions."""

from .metrics import (
    POPULATION_REQUIRED_MARKERS,
    EvaluationResult,
    compute_critical_gate_recall,
    compute_hallucination_rate,
    compute_hierarchy_f1,
    compute_structure_accuracy,
    normalize_gate_name,
    normalize_gate_semantic,
)
from .normalization import CELL_TYPE_SYNONYMS
from .response_parser import ParseResult, parse_llm_response
from .scorer import GatingScorer, score_prediction
from .task_failure import (
    TaskFailureResult,
    TaskFailureType,
    compute_task_failure_rate,
    detect_task_failure,
)

# New evaluation modules
from .pass_at_k import (
    PassAtKResult,
    AggregatedPassAtK,
    compute_pass_at_k,
    compute_pass_k_power,
    analyze_pass_at_k,
    aggregate_pass_at_k,
    compare_models_pass_at_k,
)
from .graph_similarity import (
    HierarchyGraph,
    compute_tree_edit_distance,
    compute_tree_similarity,
    compute_structure_similarity,
    StructureSimilarityResult,
)

# Semantic similarity (lazy import due to heavy dependencies)
def get_semantic_matcher():
    """Get SemanticMatcher (lazy import to avoid loading model at startup)."""
    from .semantic_similarity import SemanticMatcher
    return SemanticMatcher

def compute_semantic_f1(predicted, ground_truth, matcher=None):
    """Compute F1 using semantic similarity."""
    from .semantic_similarity import compute_semantic_f1 as _compute_semantic_f1
    return _compute_semantic_f1(predicted, ground_truth, matcher)

__all__ = [
    # Core metrics
    "compute_hierarchy_f1",
    "compute_structure_accuracy",
    "compute_critical_gate_recall",
    "compute_hallucination_rate",
    "EvaluationResult",
    "normalize_gate_name",
    "normalize_gate_semantic",
    "CELL_TYPE_SYNONYMS",
    "POPULATION_REQUIRED_MARKERS",
    # Scoring
    "GatingScorer",
    "score_prediction",
    "parse_llm_response",
    "ParseResult",
    # Task failure
    "TaskFailureType",
    "TaskFailureResult",
    "detect_task_failure",
    "compute_task_failure_rate",
    # Pass@k metrics
    "PassAtKResult",
    "AggregatedPassAtK",
    "compute_pass_at_k",
    "compute_pass_k_power",
    "analyze_pass_at_k",
    "aggregate_pass_at_k",
    "compare_models_pass_at_k",
    # Graph similarity
    "HierarchyGraph",
    "compute_tree_edit_distance",
    "compute_tree_similarity",
    "compute_structure_similarity",
    "StructureSimilarityResult",
    # Semantic similarity
    "get_semantic_matcher",
    "compute_semantic_f1",
]
