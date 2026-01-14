"""
Batch scoring of predictions against ground truth.

This module scores collected predictions without making LLM calls.
Enables modular pipeline architecture:

    PredictionCollector → BatchScorer → LLMJudge → ResultsAggregator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curation.schemas import TestCase
from evaluation.scorer import GatingScorer
from evaluation.hierarchy import extract_gate_names
from evaluation.metrics import compute_hierarchy_f1
from evaluation.normalization import normalize_gate_semantic
from utils.checkpoint import CheckpointManager
from utils.serializable import SerializableMixin

from .prediction_collector import Prediction

# Lazy import for semantic similarity (heavy dependencies)
_semantic_matcher = None


def _get_semantic_matcher():
    """Lazy-load semantic matcher to avoid startup overhead."""
    global _semantic_matcher
    if _semantic_matcher is None:
        try:
            from evaluation.semantic_similarity import SemanticMatcher
            _semantic_matcher = SemanticMatcher()
        except ImportError:
            # sentence-transformers not installed
            _semantic_matcher = False
    return _semantic_matcher if _semantic_matcher else None


def _compute_synonym_f1(pred_gates: set[str], gt_gates: set[str]) -> float:
    """Compute F1 using dictionary-based semantic normalization."""
    if not pred_gates or not gt_gates:
        return 0.0

    # Normalize both sets using semantic synonyms
    pred_normalized = {normalize_gate_semantic(g) for g in pred_gates}
    gt_normalized = {normalize_gate_semantic(g) for g in gt_gates}

    # Compute overlap
    matching = pred_normalized & gt_normalized

    precision = len(matching) / len(pred_normalized) if pred_normalized else 0.0
    recall = len(matching) / len(gt_normalized) if gt_normalized else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class ScoringResult(SerializableMixin):
    """Result of scoring a prediction against ground truth."""

    # Identity
    test_case_id: str
    model: str
    condition: str
    bootstrap_run: int

    # Scores - Multiple F1 variants for comparison
    hierarchy_f1: float           # String normalization (baseline)
    synonym_f1: float = 0.0       # Dictionary-based semantic matching
    semantic_f1: float = 0.0      # Embedding-based (sentence-transformers)
    weighted_semantic_f1: float = 0.0  # Embedding-weighted by confidence
    structure_accuracy: float
    critical_gate_recall: float
    hallucination_rate: float
    parse_success: bool

    # Raw data for judge
    raw_response: str
    parsed_hierarchy: dict | None = None
    ground_truth_gates: list[str] = field(default_factory=list)

    # Metadata
    error: str | None = None

    @property
    def key(self) -> tuple:
        """Unique key for deduplication."""
        return (self.bootstrap_run, self.test_case_id, self.model, self.condition)


class BatchScorer:
    """Scores predictions in batch against ground truth.

    Supports:
    - Parallel scoring
    - Checkpoint/resume
    - Ground truth lookup
    """

    def __init__(
        self,
        test_cases: list[TestCase],
        checkpoint_dir: Path | None = None,
    ):
        self.test_cases = test_cases
        self.checkpoint_dir = checkpoint_dir
        self.scorer = GatingScorer()

        # Checkpoint manager for resume support
        self._checkpoint = CheckpointManager(checkpoint_dir)

        # Build lookup for fast ground truth access
        self._test_case_lookup = {tc.test_case_id: tc for tc in test_cases}

    def score_all(
        self,
        predictions: list[Prediction],
    ) -> list[ScoringResult]:
        """Score all predictions against ground truth.

        Args:
            predictions: List of raw predictions to score

        Returns:
            List of scoring results
        """
        results = []

        for prediction in predictions:
            result = self.score_one(prediction)
            results.append(result)

        # Save checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint(results)

        return results

    def score_one(self, prediction: Prediction) -> ScoringResult:
        """Score a single prediction."""
        test_case = self._test_case_lookup.get(prediction.test_case_id)

        if not test_case:
            return ScoringResult(
                test_case_id=prediction.test_case_id,
                model=prediction.model,
                condition=prediction.condition,
                bootstrap_run=prediction.bootstrap_run,
                hierarchy_f1=0.0,
                structure_accuracy=0.0,
                critical_gate_recall=0.0,
                hallucination_rate=0.0,
                parse_success=False,
                raw_response=prediction.raw_response,
                error=f"Test case not found: {prediction.test_case_id}",
            )

        if prediction.error:
            return ScoringResult(
                test_case_id=prediction.test_case_id,
                model=prediction.model,
                condition=prediction.condition,
                bootstrap_run=prediction.bootstrap_run,
                hierarchy_f1=0.0,
                structure_accuracy=0.0,
                critical_gate_recall=0.0,
                hallucination_rate=0.0,
                parse_success=False,
                raw_response=prediction.raw_response,
                error=prediction.error,
            )

        try:
            score_result = self.scorer.score(
                response=prediction.raw_response,
                test_case=test_case,
                model=prediction.model,
                condition=prediction.condition,
            )

            # Compute additional F1 variants if parsing succeeded
            synonym_f1 = 0.0
            semantic_f1 = 0.0
            weighted_semantic_f1 = 0.0

            if score_result.parse_success and score_result.parsed_hierarchy:
                pred_gates = extract_gate_names(score_result.parsed_hierarchy)
                gt_gates = extract_gate_names(test_case.gating_hierarchy)

                # Dictionary-based semantic F1
                synonym_f1 = _compute_synonym_f1(pred_gates, gt_gates)

                # Embedding-based semantic F1 (if available)
                matcher = _get_semantic_matcher()
                if matcher is not None:
                    try:
                        from evaluation.semantic_similarity import (
                            compute_semantic_f1 as _compute_semantic_f1,
                            compute_weighted_semantic_f1 as _compute_weighted_semantic_f1,
                        )
                        semantic_f1, _, _, _ = _compute_semantic_f1(pred_gates, gt_gates, matcher)
                        weighted_semantic_f1, _, _, _ = _compute_weighted_semantic_f1(pred_gates, gt_gates, matcher)
                    except Exception:
                        pass  # Fall back to 0.0 on any error

            return ScoringResult(
                test_case_id=prediction.test_case_id,
                model=prediction.model,
                condition=prediction.condition,
                bootstrap_run=prediction.bootstrap_run,
                hierarchy_f1=score_result.hierarchy_f1,
                synonym_f1=synonym_f1,
                semantic_f1=semantic_f1,
                weighted_semantic_f1=weighted_semantic_f1,
                structure_accuracy=score_result.structure_accuracy,
                critical_gate_recall=score_result.critical_gate_recall,
                hallucination_rate=getattr(score_result, 'hallucination_rate', 0.0),
                parse_success=score_result.parse_success,
                raw_response=prediction.raw_response,
                parsed_hierarchy=getattr(score_result, 'parsed_hierarchy', None),
                ground_truth_gates=test_case.gating_hierarchy.get_all_gates(),
            )

        except Exception as e:
            return ScoringResult(
                test_case_id=prediction.test_case_id,
                model=prediction.model,
                condition=prediction.condition,
                bootstrap_run=prediction.bootstrap_run,
                hierarchy_f1=0.0,
                structure_accuracy=0.0,
                critical_gate_recall=0.0,
                hallucination_rate=0.0,
                parse_success=False,
                raw_response=prediction.raw_response,
                error=str(e),
            )

    def load_checkpoint(self) -> tuple[list[ScoringResult], set[tuple]]:
        """Load scoring results from checkpoint.

        Returns:
            Tuple of (results list, set of completed keys for resume logic)
        """
        return self._checkpoint.load_with_keys(
            "scoring_results.json",
            ScoringResult,
            key_fn=lambda r: r.key,
        )

    def save_checkpoint(self, results: list[ScoringResult]) -> None:
        """Save scoring results to checkpoint."""
        self._checkpoint.save(results, "scoring_results.json")


def compute_aggregate_stats(
    results: list[ScoringResult],
) -> dict[str, Any]:
    """Compute aggregate statistics from scoring results.

    Returns:
        Dictionary with:
        - overall_stats: Mean F1, structure accuracy, etc.
        - by_model: Stats grouped by model
        - by_condition: Stats grouped by condition
        - by_test_case: Stats grouped by test case
    """
    if not results:
        return {"error": "No results to aggregate"}

    import statistics
    from collections import defaultdict

    def compute_stats(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "n": len(scores),
        }

    # Overall stats - all F1 variants
    valid_results = [r for r in results if not r.error]
    all_f1 = [r.hierarchy_f1 for r in valid_results]
    all_synonym_f1 = [r.synonym_f1 for r in valid_results]
    all_semantic_f1 = [r.semantic_f1 for r in valid_results]
    all_weighted_semantic_f1 = [r.weighted_semantic_f1 for r in valid_results]
    all_struct = [r.structure_accuracy for r in valid_results]
    all_critical = [r.critical_gate_recall for r in valid_results]
    parse_success_rate = sum(1 for r in results if r.parse_success) / len(results)

    overall = {
        "hierarchy_f1": compute_stats(all_f1),
        "synonym_f1": compute_stats(all_synonym_f1),
        "semantic_f1": compute_stats(all_semantic_f1),
        "weighted_semantic_f1": compute_stats(all_weighted_semantic_f1),
        "structure_accuracy": compute_stats(all_struct),
        "critical_gate_recall": compute_stats(all_critical),
        "parse_success_rate": parse_success_rate,
        "error_count": sum(1 for r in results if r.error),
        "total": len(results),
    }

    # By model - all F1 variants
    by_model = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)

    model_stats = {}
    for model, model_results in by_model.items():
        valid = [r for r in model_results if not r.error]
        model_stats[model] = {
            "hierarchy_f1": compute_stats([r.hierarchy_f1 for r in valid]),
            "synonym_f1": compute_stats([r.synonym_f1 for r in valid]),
            "semantic_f1": compute_stats([r.semantic_f1 for r in valid]),
            "weighted_semantic_f1": compute_stats([r.weighted_semantic_f1 for r in valid]),
            "n": len(model_results),
            "parse_success_rate": sum(1 for r in model_results if r.parse_success) / len(model_results),
        }

    # By condition - all F1 variants
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r)

    condition_stats = {}
    for condition, cond_results in by_condition.items():
        valid = [r for r in cond_results if not r.error]
        condition_stats[condition] = {
            "hierarchy_f1": compute_stats([r.hierarchy_f1 for r in valid]),
            "synonym_f1": compute_stats([r.synonym_f1 for r in valid]),
            "semantic_f1": compute_stats([r.semantic_f1 for r in valid]),
            "weighted_semantic_f1": compute_stats([r.weighted_semantic_f1 for r in valid]),
            "n": len(cond_results),
        }

    return {
        "overall": overall,
        "by_model": model_stats,
        "by_condition": condition_stats,
    }
