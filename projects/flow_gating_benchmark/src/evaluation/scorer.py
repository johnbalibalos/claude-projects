"""
High-level scoring interface for gating predictions.

Combines response parsing with metric computation to provide
a single entry point for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from curation.schemas import TestCase, GatingHierarchy, Panel
from .metrics import EvaluationResult, evaluate_prediction
from .response_parser import parse_llm_response, ParseResult


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

    # Raw data
    raw_response: str | None = None
    parsed_hierarchy: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_case_id": self.test_case_id,
            "model": self.model,
            "condition": self.condition,
            "parse_success": self.parse_success,
            "parse_format": self.parse_format,
            "parse_error": self.parse_error,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
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
        # Parse response
        parse_result = parse_llm_response(response)

        if not parse_result.success:
            return ScoringResult(
                test_case_id=test_case.test_case_id,
                model=model,
                condition=condition,
                parse_success=False,
                parse_error=parse_result.error,
                raw_response=response,
            )

        # Evaluate prediction
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
            raw_response=response,
            parsed_hierarchy=parse_result.hierarchy,
        )

    def score_batch(
        self,
        responses: list[tuple[str, TestCase, str, str]],
    ) -> list[ScoringResult]:
        """
        Score a batch of responses.

        Args:
            responses: List of (response, test_case, model, condition) tuples

        Returns:
            List of ScoringResults
        """
        return [
            self.score(response, test_case, model, condition)
            for response, test_case, model, condition in responses
        ]


def score_prediction(
    response: str,
    test_case: TestCase,
    model: str = "unknown",
    condition: str = "unknown",
) -> ScoringResult:
    """
    Convenience function to score a single prediction.

    Args:
        response: Raw LLM response
        test_case: Test case with ground truth
        model: Model name
        condition: Experimental condition

    Returns:
        ScoringResult
    """
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

    # Filter to successful parses
    valid_results = [r for r in results if r.parse_success and r.evaluation]

    if not valid_results:
        return {
            "total": len(results),
            "parse_success_rate": 0.0,
            "error": "No valid results after parsing",
        }

    # Compute aggregates
    metrics = {
        "total": len(results),
        "valid": len(valid_results),
        "parse_success_rate": len(valid_results) / len(results),
    }

    # Aggregate each metric
    for metric_name in [
        "hierarchy_f1",
        "hierarchy_precision",
        "hierarchy_recall",
        "structure_accuracy",
        "critical_gate_recall",
        "hallucination_rate",
        "depth_accuracy",
    ]:
        values = [
            getattr(r.evaluation, metric_name)
            for r in valid_results
            if r.evaluation
        ]

        if values:
            metrics[f"{metric_name}_mean"] = sum(values) / len(values)
            metrics[f"{metric_name}_min"] = min(values)
            metrics[f"{metric_name}_max"] = max(values)

    return metrics


def compute_metrics_by_condition(
    results: list[ScoringResult],
) -> dict[str, dict[str, Any]]:
    """
    Compute aggregate metrics grouped by condition.

    Args:
        results: List of ScoringResults

    Returns:
        Dictionary mapping condition to aggregate metrics
    """
    # Group by condition
    by_condition: dict[str, list[ScoringResult]] = {}
    for result in results:
        condition = result.condition
        if condition not in by_condition:
            by_condition[condition] = []
        by_condition[condition].append(result)

    # Compute aggregates for each condition
    return {
        condition: compute_aggregate_metrics(condition_results)
        for condition, condition_results in by_condition.items()
    }


def compute_metrics_by_model(
    results: list[ScoringResult],
) -> dict[str, dict[str, Any]]:
    """
    Compute aggregate metrics grouped by model.

    Args:
        results: List of ScoringResults

    Returns:
        Dictionary mapping model to aggregate metrics
    """
    # Group by model
    by_model: dict[str, list[ScoringResult]] = {}
    for result in results:
        model = result.model
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(result)

    # Compute aggregates for each model
    return {
        model: compute_aggregate_metrics(model_results)
        for model, model_results in by_model.items()
    }
