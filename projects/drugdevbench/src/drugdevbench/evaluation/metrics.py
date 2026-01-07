"""Metrics computation for benchmark results."""

import statistics
from collections import defaultdict
from typing import Any

from drugdevbench.data.schemas import (
    AblationResult,
    BenchmarkResult,
    EvaluationResponse,
    FigureType,
    PromptCondition,
    QuestionType,
)


def compute_benchmark_metrics(
    responses: list[EvaluationResponse],
    run_id: str,
    model: str,
    condition: PromptCondition,
    figure_type: FigureType | None = None,
) -> BenchmarkResult:
    """Compute aggregated metrics for a benchmark run.

    Args:
        responses: List of evaluation responses with scores
        run_id: Identifier for this run
        model: Model used
        condition: Prompt condition
        figure_type: Optional figure type filter

    Returns:
        BenchmarkResult with aggregated metrics
    """
    # Filter responses with scores
    scored_responses = [r for r in responses if r.score is not None]

    if not scored_responses:
        return BenchmarkResult(
            run_id=run_id,
            model=model,
            condition=condition,
            figure_type=figure_type,
            n_figures=0,
            n_questions=0,
            mean_score=0.0,
            std_score=0.0,
            scores_by_question_type={},
            total_cost_usd=0.0,
            total_time_s=0.0,
        )

    # Basic metrics
    scores = [r.score for r in scored_responses]
    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # Unique figures
    unique_figures = set(r.figure_id for r in scored_responses)

    # Scores by question type (from metadata if available)
    scores_by_type: dict[str, list[float]] = defaultdict(list)
    for r in scored_responses:
        if r.metadata and "question_type" in r.metadata:
            q_type = r.metadata["question_type"]
            scores_by_type[q_type].append(r.score)

    mean_scores_by_type = {
        q_type: statistics.mean(type_scores) for q_type, type_scores in scores_by_type.items()
    }

    # Cost and time
    total_cost = sum(r.cost_usd or 0.0 for r in scored_responses)
    total_time = sum(r.latency_ms or 0 for r in scored_responses) / 1000.0

    return BenchmarkResult(
        run_id=run_id,
        model=model,
        condition=condition,
        figure_type=figure_type,
        n_figures=len(unique_figures),
        n_questions=len(scored_responses),
        mean_score=mean_score,
        std_score=std_score,
        scores_by_question_type=mean_scores_by_type,
        total_cost_usd=total_cost,
        total_time_s=total_time,
    )


def compute_ablation_metrics(
    results_by_condition: dict[PromptCondition, BenchmarkResult],
    run_id: str,
    model: str,
    baseline: PromptCondition = PromptCondition.VANILLA,
    figure_type: FigureType | None = None,
) -> AblationResult:
    """Compute ablation study metrics comparing conditions.

    Args:
        results_by_condition: Results for each prompt condition
        run_id: Identifier for this ablation
        model: Model used
        baseline: Baseline condition for comparison
        figure_type: Optional figure type filter

    Returns:
        AblationResult with comparison metrics
    """
    # Compute improvements over baseline
    improvements = {}
    baseline_score = results_by_condition.get(baseline)

    if baseline_score and baseline_score.mean_score > 0:
        for condition, result in results_by_condition.items():
            if condition != baseline:
                improvement = (
                    (result.mean_score - baseline_score.mean_score)
                    / baseline_score.mean_score
                    * 100
                )
                improvements[condition.value] = improvement

    return AblationResult(
        run_id=run_id,
        model=model,
        figure_type=figure_type,
        results_by_condition={c.value: r for c, r in results_by_condition.items()},
        baseline_condition=baseline,
        improvements=improvements,
    )


def compute_summary_statistics(responses: list[EvaluationResponse]) -> dict[str, Any]:
    """Compute summary statistics for a set of responses.

    Args:
        responses: List of evaluation responses

    Returns:
        Dictionary of summary statistics
    """
    scored = [r for r in responses if r.score is not None]
    scores = [r.score for r in scored]

    if not scores:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

    return {
        "n": len(scores),
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0,
        "min": min(scores),
        "max": max(scores),
        "median": statistics.median(scores),
        "q25": statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores),
        "q75": statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores),
    }


def compute_model_comparison(
    responses_by_model: dict[str, list[EvaluationResponse]],
) -> dict[str, dict[str, Any]]:
    """Compare performance across models.

    Args:
        responses_by_model: Responses grouped by model

    Returns:
        Comparison statistics by model
    """
    comparison = {}

    for model, responses in responses_by_model.items():
        comparison[model] = compute_summary_statistics(responses)

        # Add cost info
        costs = [r.cost_usd for r in responses if r.cost_usd is not None]
        comparison[model]["total_cost_usd"] = sum(costs)
        comparison[model]["avg_cost_per_query"] = (
            statistics.mean(costs) if costs else 0
        )

        # Add latency info
        latencies = [r.latency_ms for r in responses if r.latency_ms is not None]
        comparison[model]["avg_latency_ms"] = (
            statistics.mean(latencies) if latencies else 0
        )

    return comparison


def compute_figure_type_breakdown(
    responses: list[EvaluationResponse],
    figure_type_map: dict[str, FigureType],
) -> dict[str, dict[str, Any]]:
    """Break down performance by figure type.

    Args:
        responses: List of evaluation responses
        figure_type_map: Mapping from figure_id to FigureType

    Returns:
        Statistics broken down by figure type
    """
    by_type: dict[str, list[EvaluationResponse]] = defaultdict(list)

    for response in responses:
        if response.figure_id in figure_type_map:
            fig_type = figure_type_map[response.figure_id]
            by_type[fig_type.value].append(response)

    return {
        fig_type: compute_summary_statistics(type_responses)
        for fig_type, type_responses in by_type.items()
    }
