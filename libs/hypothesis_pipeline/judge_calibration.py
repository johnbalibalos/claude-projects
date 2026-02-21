"""
Judge calibration against human annotations.

Provides calibrate_judge() to measure how well an LLM judge
correlates with human scores, plus bias and MAE metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .llm_judge import LLMJudge


@dataclass
class JudgeCalibrationResult:
    """Calibration analysis for a judge."""

    n_samples: int
    correlation_with_human: float
    mae: float  # Mean absolute error vs human
    bias: float  # Systematic over/under scoring
    is_well_calibrated: bool


def calibrate_judge(
    judge: LLMJudge,
    calibration_set: list[dict[str, Any]],
    question_field: str = "question",
    response_field: str = "response",
    human_score_field: str = "human_score",
    max_score: float = 1.0,
) -> JudgeCalibrationResult:
    """
    Calibrate a judge against human annotations.

    Args:
        judge: The LLM judge to calibrate
        calibration_set: List of examples with human scores
        question_field: Field name for questions
        response_field: Field name for responses
        human_score_field: Field name for human scores
        max_score: Maximum possible score

    Returns:
        JudgeCalibrationResult with calibration metrics
    """
    judge_scores = []
    human_scores = []

    for example in calibration_set:
        result = judge.evaluate(
            question=example[question_field],
            response=example[response_field],
        )
        judge_scores.append(result.normalized_score)
        human_scores.append(example[human_score_field] / max_score)

    judge_scores = np.array(judge_scores)
    human_scores = np.array(human_scores)

    # Compute metrics
    correlation = float(np.corrcoef(judge_scores, human_scores)[0, 1])
    mae = float(np.mean(np.abs(judge_scores - human_scores)))
    bias = float(np.mean(judge_scores - human_scores))

    # Well-calibrated if correlation > 0.7 and MAE < 0.15
    is_calibrated = correlation > 0.7 and mae < 0.15

    return JudgeCalibrationResult(
        n_samples=len(calibration_set),
        correlation_with_human=correlation,
        mae=mae,
        bias=bias,
        is_well_calibrated=is_calibrated,
    )

