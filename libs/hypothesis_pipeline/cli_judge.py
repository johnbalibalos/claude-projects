"""CLI-based LLM Judge using Claude Max subscription.

Factory functions for creating judges that use the Claude CLI instead of API.
No per-token costs (subscription-based), but includes rate limiting.

Example:
    from hypothesis_pipeline.cli_judge import create_cli_judge

    # Default: Opus, 1s rate limit
    judge = create_cli_judge()

    # Evaluate
    result = judge.evaluate(
        question="What causes diabetes?",
        response="Diabetes is caused by...",
        ground_truth="Type 2 diabetes results from..."
    )
    print(f"Score: {result.normalized_score:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .llm_judge import (
    EnsembleJudge,
    EvaluationRubric,
    LLMJudge,
    PairwiseJudge,
)

if TYPE_CHECKING:
    pass


@dataclass
class CLIJudgeConfig:
    """Configuration for CLI-based judges."""

    # Model settings
    model: str = "claude-opus-4-20250514"
    timeout_seconds: int = 300
    max_retries: int = 3

    # Rate limiting
    rate_limit: float = 1.0  # Seconds between calls

    # Rubric
    rubric_type: Literal["qa", "scientific", "custom"] = "qa"
    custom_rubric: EvaluationRubric | None = None


def create_cli_judge(config: CLIJudgeConfig | None = None) -> LLMJudge:
    """
    Create an LLM judge using Claude CLI.

    Uses Claude Max subscription (no per-token costs).
    Includes rate limiting to avoid overwhelming the CLI.

    Args:
        config: Judge configuration (defaults: Opus, 1s rate limit)

    Returns:
        Configured LLMJudge instance
    """
    from model_client.cli_client import ClaudeCLIClient, CLIConfig

    config = config or CLIJudgeConfig()

    cli_config = CLIConfig(
        model=config.model,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        rate_limit=config.rate_limit,
    )
    client = ClaudeCLIClient(cli_config)

    # Select rubric
    if config.rubric_type == "scientific":
        rubric = EvaluationRubric.scientific_analysis_rubric()
    elif config.rubric_type == "custom" and config.custom_rubric:
        rubric = config.custom_rubric
    else:
        rubric = EvaluationRubric.default_qa_rubric()

    return LLMJudge(
        judge_model=client,
        rubric=rubric,
        model_name=config.model,
    )


def create_cli_pairwise_judge(
    config: CLIJudgeConfig | None = None,
    debias: bool = True,
) -> PairwiseJudge:
    """
    Create a pairwise comparison judge using CLI.

    Compares two responses and determines which is better.
    Uses position debiasing by default (runs comparison in both orders).

    Args:
        config: Judge configuration
        debias: Whether to run both orderings for position bias mitigation

    Returns:
        Configured PairwiseJudge instance
    """
    from model_client.cli_client import ClaudeCLIClient, CLIConfig

    config = config or CLIJudgeConfig()

    cli_config = CLIConfig(
        model=config.model,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        rate_limit=config.rate_limit,
    )
    client = ClaudeCLIClient(cli_config)

    return PairwiseJudge(judge_model=client, debias=debias)


def create_cli_ensemble_judge(
    models: list[str] | None = None,
    rate_limit: float = 1.0,
    rubric_type: Literal["qa", "scientific"] = "qa",
    aggregation: Literal["mean", "median", "majority"] = "mean",
) -> EnsembleJudge:
    """
    Create an ensemble of CLI judges with different models.

    Combines judgments from multiple models for more robust evaluation.

    Args:
        models: Model IDs to use (default: opus + sonnet)
        rate_limit: Seconds between calls per client
        rubric_type: Type of evaluation rubric
        aggregation: How to combine scores ("mean", "median", "majority")

    Returns:
        Configured EnsembleJudge instance
    """
    models = models or [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
    ]

    judges = []
    for model in models:
        config = CLIJudgeConfig(
            model=model,
            rate_limit=rate_limit,
            rubric_type=rubric_type,
        )
        judges.append(create_cli_judge(config))

    return EnsembleJudge(judges=judges, aggregation=aggregation)
