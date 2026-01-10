"""
High-level scoring interface for gating predictions.

Combines response parsing with metric computation to provide
a single entry point for evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from curation.schemas import TestCase

from .metrics import EvaluationResult, evaluate_prediction
from .response_parser import parse_llm_response
from .task_failure import (
    TaskFailureResult,
    TaskFailureType,
    detect_task_failure,
)


@dataclass
class ScoringResult:
    """Complete result of scoring a prediction."""

    test_case_id: str
    model: str
    condition: str

    # Parse result
    parse_success: bool
    parse_format: str | None = None
    parse_error: str | None = None

    # Evaluation result (None if parsing failed)
    evaluation: EvaluationResult | None = None

    # Task failure detection
    task_failure: TaskFailureResult | None = None

    # Raw data
    raw_response: str | None = None
    parsed_hierarchy: dict | None = None

    # Schema version for compatibility checking
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        task_failure_dict = None
        if self.task_failure:
            task_failure_dict = {
                "is_failure": self.task_failure.is_failure,
                "failure_type": self.task_failure.failure_type.value,
                "confidence": self.task_failure.confidence,
                "evidence": self.task_failure.evidence,
                "gate_count": self.task_failure.gate_count,
            }

        return {
            "schema_version": self.schema_version,
            "test_case_id": self.test_case_id,
            "model": self.model,
            "condition": self.condition,
            "parse_success": self.parse_success,
            "parse_format": self.parse_format,
            "parse_error": self.parse_error,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "task_failure": task_failure_dict,
            "parsed_hierarchy": self.parsed_hierarchy,
        }

    @property
    def hierarchy_f1(self) -> float:
        """Convenience accessor for main metric."""
        return self.evaluation.hierarchy_f1 if self.evaluation else 0.0

    @property
    def structure_accuracy(self) -> float:
        """Convenience accessor for structure metric."""
        return self.evaluation.structure_accuracy if self.evaluation else 0.0

    @property
    def critical_gate_recall(self) -> float:
        """Convenience accessor for critical gate metric."""
        return self.evaluation.critical_gate_recall if self.evaluation else 0.0

    @property
    def hallucination_rate(self) -> float:
        """Convenience accessor for hallucination metric."""
        return self.evaluation.hallucination_rate if self.evaluation else 0.0

    @property
    def is_task_failure(self) -> bool:
        """Whether this result represents a task failure."""
        return self.task_failure.is_failure if self.task_failure else False

    @property
    def task_failure_type(self) -> TaskFailureType:
        """The type of task failure, if any."""
        return self.task_failure.failure_type if self.task_failure else TaskFailureType.NONE


class GatingScorer:
    """
    Scorer for gating hierarchy predictions.

    Handles both parsing and evaluation in a single interface.
    """

    def __init__(self, critical_gates: list[str] | None = None):
        """
        Initialize scorer.

        Args:
            critical_gates: Optional list of critical gates to check
        """
        self.critical_gates = critical_gates

    def score(
        self,
        response: str,
        test_case: TestCase,
        model: str = "unknown",
        condition: str = "unknown",
    ) -> ScoringResult:
        """
        Score a single LLM response.

        Args:
            response: Raw LLM response text
            test_case: Test case with ground truth
            model: Model name for tracking
            condition: Experimental condition name

        Returns:
            Complete ScoringResult
        """
        parse_result = parse_llm_response(response)

        # Always run task failure detection on the raw response
        task_failure_result = detect_task_failure(
            response,
            parsed_hierarchy=parse_result.hierarchy if parse_result.success else None,
        )

        if not parse_result.success:
            return ScoringResult(
                test_case_id=test_case.test_case_id,
                model=model,
                condition=condition,
                parse_success=False,
                parse_error=parse_result.error,
                task_failure=task_failure_result,
                raw_response=response,
            )

        # parse_result.success is True here, so hierarchy should be set
        assert parse_result.hierarchy is not None
        evaluation = evaluate_prediction(
            predicted=parse_result.hierarchy,
            ground_truth=test_case.gating_hierarchy,
            panel=test_case.panel,
            critical_gates=self.critical_gates,
        )

        return ScoringResult(
            test_case_id=test_case.test_case_id,
            model=model,
            condition=condition,
            parse_success=True,
            parse_format=parse_result.format_detected,
            evaluation=evaluation,
            task_failure=task_failure_result,
            raw_response=response,
            parsed_hierarchy=parse_result.hierarchy,
        )

    def score_batch(
        self,
        items: list[tuple[str, TestCase, str, str]],
    ) -> list[ScoringResult]:
        """
        Score a batch of responses.

        Args:
            items: List of (response, test_case, model, condition) tuples

        Returns:
            List of ScoringResults
        """
        return [
            self.score(response, test_case, model, condition)
            for response, test_case, model, condition in items
        ]


def score_prediction(
    response: str,
    test_case: TestCase,
    model: str = "unknown",
    condition: str = "unknown",
) -> ScoringResult:
    """Convenience function to score a single prediction."""
    scorer = GatingScorer()
    return scorer.score(response, test_case, model, condition)


def compute_aggregate_metrics(results: list[ScoringResult]) -> dict[str, Any]:
    """
    Compute aggregate metrics across multiple results.

    Args:
        results: List of ScoringResults

    Returns:
        Dictionary of aggregate metrics
    """
    if not results:
        return {"error": "No results to aggregate"}

    valid_results = [r for r in results if r.parse_success and r.evaluation]

    if not valid_results:
        return {
            "total": len(results),
            "parse_success_rate": 0.0,
            "error": "No valid results after parsing",
        }

    metrics: dict[str, Any] = {
        "total": len(results),
        "valid": len(valid_results),
        "parse_success_rate": len(valid_results) / len(results),
    }

    metric_names = [
        "hierarchy_f1",
        "hierarchy_precision",
        "hierarchy_recall",
        "structure_accuracy",
        "critical_gate_recall",
        "hallucination_rate",
        "depth_accuracy",
    ]

    for metric_name in metric_names:
        values = [
            getattr(r.evaluation, metric_name)
            for r in valid_results
            if r.evaluation
        ]
        if values:
            metrics[f"{metric_name}_mean"] = sum(values) / len(values)
            metrics[f"{metric_name}_min"] = min(values)
            metrics[f"{metric_name}_max"] = max(values)

    # Add task failure metrics
    task_failure_counts = dict.fromkeys(TaskFailureType, 0)
    for r in results:
        if r.task_failure and r.task_failure.is_failure:
            task_failure_counts[r.task_failure.failure_type] += 1

    total_task_failures = sum(
        v for k, v in task_failure_counts.items() if k != TaskFailureType.NONE
    )
    metrics["task_failure_rate"] = total_task_failures / len(results)
    metrics["task_failure_count"] = total_task_failures
    metrics["task_failures_by_type"] = {
        "meta_questions": task_failure_counts[TaskFailureType.META_QUESTIONS],
        "refusals": task_failure_counts[TaskFailureType.REFUSAL],
        "instructions": task_failure_counts[TaskFailureType.INSTRUCTIONS],
        "empty": task_failure_counts[TaskFailureType.EMPTY],
        "malformed": task_failure_counts[TaskFailureType.MALFORMED],
    }

    return metrics


def _group_results(
    results: list[ScoringResult],
    key_fn: Callable[[ScoringResult], str],
) -> dict[str, list[ScoringResult]]:
    """Group results by a key function."""
    grouped: dict[str, list[ScoringResult]] = {}
    for result in results:
        key = key_fn(result)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    return grouped


def compute_metrics_by_condition(results: list[ScoringResult]) -> dict[str, dict[str, Any]]:
    """Compute aggregate metrics grouped by condition."""
    grouped = _group_results(results, lambda r: r.condition)
    return {
        condition: compute_aggregate_metrics(condition_results)
        for condition, condition_results in grouped.items()
    }


def compute_metrics_by_model(results: list[ScoringResult]) -> dict[str, dict[str, Any]]:
    """Compute aggregate metrics grouped by model."""
    grouped = _group_results(results, lambda r: r.model)
    return {
        model: compute_aggregate_metrics(model_results)
        for model, model_results in grouped.items()
    }


def compute_metrics_by_test_case(results: list[ScoringResult]) -> dict[str, dict[str, Any]]:
    """Compute aggregate metrics grouped by test case."""
    grouped = _group_results(results, lambda r: r.test_case_id)
    return {
        test_case_id: compute_aggregate_metrics(tc_results)
        for test_case_id, tc_results in grouped.items()
    }


def filter_results(
    results: list[ScoringResult],
    *,
    model: str | None = None,
    condition: str | None = None,
    test_case_id: str | None = None,
    parse_success: bool | None = None,
    min_f1: float | None = None,
) -> list[ScoringResult]:
    """
    Filter results by various criteria.

    Args:
        results: List of ScoringResults to filter
        model: Filter by model name
        condition: Filter by condition name
        test_case_id: Filter by test case ID
        parse_success: Filter by parse success status
        min_f1: Filter by minimum F1 score

    Returns:
        Filtered list of ScoringResults
    """
    filtered = results

    if model is not None:
        filtered = [r for r in filtered if r.model == model]

    if condition is not None:
        filtered = [r for r in filtered if r.condition == condition]

    if test_case_id is not None:
        filtered = [r for r in filtered if r.test_case_id == test_case_id]

    if parse_success is not None:
        filtered = [r for r in filtered if r.parse_success == parse_success]

    if min_f1 is not None:
        filtered = [r for r in filtered if r.hierarchy_f1 >= min_f1]

    return filtered
