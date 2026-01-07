"""
Experimental conditions for the gating benchmark.

Defines the independent variables:
- Model: Which LLM to use
- Context level: How much information to provide
- Prompting strategy: Direct vs chain-of-thought
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal


@dataclass
class ExperimentCondition:
    """A single experimental condition."""

    name: str
    model: str
    context_level: Literal["minimal", "standard", "rich"]
    prompt_strategy: Literal["direct", "cot"]

    @property
    def condition_id(self) -> str:
        """Unique identifier for this condition."""
        return f"{self.model}_{self.context_level}_{self.prompt_strategy}"


# Available models for testing
MODELS = {
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}

# Context levels
CONTEXT_LEVELS = ["minimal", "standard", "rich"]

# Prompting strategies
PROMPT_STRATEGIES = ["direct", "cot"]


def get_all_conditions(
    models: list[str] | None = None,
    context_levels: list[str] | None = None,
    prompt_strategies: list[str] | None = None,
) -> list[ExperimentCondition]:
    """
    Generate all experimental conditions.

    Args:
        models: Models to include (default: all)
        context_levels: Context levels to include (default: all)
        prompt_strategies: Prompting strategies (default: all)

    Returns:
        List of all condition combinations
    """
    if models is None:
        models = list(MODELS.keys())

    if context_levels is None:
        context_levels = CONTEXT_LEVELS

    if prompt_strategies is None:
        prompt_strategies = PROMPT_STRATEGIES

    conditions = []
    for model, context, strategy in product(models, context_levels, prompt_strategies):
        condition = ExperimentCondition(
            name=f"{model}_{context}_{strategy}",
            model=MODELS.get(model, model),
            context_level=context,
            prompt_strategy=strategy,
        )
        conditions.append(condition)

    return conditions


def get_minimal_conditions() -> list[ExperimentCondition]:
    """
    Get minimal set of conditions for quick testing.

    1 model × 2 context levels × 1 strategy = 2 conditions
    """
    return get_all_conditions(
        models=["claude-sonnet"],
        context_levels=["minimal", "standard"],
        prompt_strategies=["direct"],
    )


def get_standard_conditions() -> list[ExperimentCondition]:
    """
    Get standard set of conditions for main experiment.

    3 models × 3 context levels × 2 strategies = 18 conditions
    """
    return get_all_conditions(
        models=["claude-sonnet", "claude-haiku", "gpt-4o"],
        context_levels=["minimal", "standard", "rich"],
        prompt_strategies=["direct", "cot"],
    )


def get_model_comparison_conditions() -> list[ExperimentCondition]:
    """
    Get conditions for model comparison (fixed context/strategy).

    All models × 1 context × 1 strategy = N models
    """
    return get_all_conditions(
        models=list(MODELS.keys()),
        context_levels=["standard"],
        prompt_strategies=["cot"],
    )


def get_ablation_conditions() -> list[ExperimentCondition]:
    """
    Get conditions for ablation study on a single model.

    1 model × all context × all strategies = 6 conditions
    """
    return get_all_conditions(
        models=["claude-sonnet"],
        context_levels=["minimal", "standard", "rich"],
        prompt_strategies=["direct", "cot"],
    )


def print_conditions(conditions: list[ExperimentCondition]) -> None:
    """Print conditions in a readable format."""
    print(f"\nExperimental Conditions ({len(conditions)} total)")
    print("=" * 60)
    print(f"{'Name':<40} {'Model':<25} {'Context':<10} {'Strategy':<8}")
    print("-" * 60)

    for cond in conditions:
        print(f"{cond.name:<40} {cond.model:<25} {cond.context_level:<10} {cond.prompt_strategy:<8}")


if __name__ == "__main__":
    print("\n=== Minimal Conditions ===")
    print_conditions(get_minimal_conditions())

    print("\n=== Standard Conditions ===")
    print_conditions(get_standard_conditions())

    print("\n=== Model Comparison Conditions ===")
    print_conditions(get_model_comparison_conditions())

    print("\n=== Ablation Conditions ===")
    print_conditions(get_ablation_conditions())
