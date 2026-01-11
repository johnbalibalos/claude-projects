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
    rag_mode: Literal["none", "oracle"] = "none"

    @property
    def condition_id(self) -> str:
        """Unique identifier for this condition."""
        return f"{self.model}_{self.context_level}_{self.prompt_strategy}_{self.rag_mode}"


# Available models for testing
MODELS = {
    # Cloud models - Anthropic
    "claude-opus": "claude-opus-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    # Cloud models - OpenAI
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    # Cloud models - Google Gemini
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-flash": "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
    # Local models (Ollama)
    "llama3.1-8b": "llama3.1:8b",
    "llama3.1-70b": "llama3.1:70b",
    "qwen2.5-7b": "qwen2.5:7b",
    "qwen2.5-72b": "qwen2.5:72b",
    "mistral-7b": "mistral:7b",
    "mixtral-8x7b": "mixtral:8x7b",
    "deepseek-r1-8b": "deepseek-r1:8b",
    "deepseek-r1-70b": "deepseek-r1:70b",
}

# Model categories
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
CLOUD_MODELS = ["claude-opus", "claude-sonnet", "claude-haiku", "gpt-4o", "gpt-4o-mini"] + GEMINI_MODELS
LOCAL_MODELS = ["llama3.1-8b", "llama3.1-70b", "qwen2.5-7b", "qwen2.5-72b",
                "mistral-7b", "mixtral-8x7b", "deepseek-r1-8b", "deepseek-r1-70b"]

# Recommended local models for scientific reasoning
RECOMMENDED_LOCAL_MODELS = {
    "llama3.1:70b": "Best overall for science tasks",
    "qwen2.5:72b": "Strong structured output, good on biology",
    "deepseek-r1:70b": "Reasoning-focused, good for CoT",
    "mixtral:8x7b": "Good balance of speed and quality",
}

# Context levels
CONTEXT_LEVELS = ["minimal", "standard", "rich"]

# Prompting strategies
PROMPT_STRATEGIES = ["direct", "cot"]

# RAG modes
RAG_MODES = ["none", "oracle"]


def get_all_conditions(
    models: list[str] | None = None,
    context_levels: list[str] | None = None,
    prompt_strategies: list[str] | None = None,
    rag_modes: list[str] | None = None,
) -> list[ExperimentCondition]:
    """
    Generate all experimental conditions.

    Args:
        models: Models to include (default: all)
        context_levels: Context levels to include (default: all)
        prompt_strategies: Prompting strategies (default: all)
        rag_modes: RAG modes to include (default: ["none"] for backward compat)

    Returns:
        List of all condition combinations
    """
    if models is None:
        models = list(MODELS.keys())

    if context_levels is None:
        context_levels = CONTEXT_LEVELS

    if prompt_strategies is None:
        prompt_strategies = PROMPT_STRATEGIES

    if rag_modes is None:
        rag_modes = ["none"]  # Default to no RAG for backward compatibility

    conditions = []
    for model, context, strategy, rag in product(models, context_levels, prompt_strategies, rag_modes):
        condition = ExperimentCondition(
            name=f"{model}_{context}_{strategy}_{rag}",
            model=MODELS.get(model, model),
            context_level=context,
            prompt_strategy=strategy,
            rag_mode=rag,
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


def get_opus_ablation_conditions() -> list[ExperimentCondition]:
    """
    Get conditions for Opus ablation study.

    1 model × all context × all strategies = 6 conditions
    """
    return get_all_conditions(
        models=["claude-opus"],
        context_levels=["minimal", "standard", "rich"],
        prompt_strategies=["direct", "cot"],
    )


def get_local_model_conditions(
    models: list[str] | None = None,
    context_levels: list[str] | None = None,
    prompt_strategies: list[str] | None = None,
) -> list[ExperimentCondition]:
    """
    Get conditions for local model comparison.

    Args:
        models: Local models to test (default: all LOCAL_MODELS)
        context_levels: Context levels to include (default: all)
        prompt_strategies: Prompting strategies (default: all)

    Returns:
        List of conditions for local model testing
    """
    if models is None:
        models = LOCAL_MODELS
    return get_all_conditions(models, context_levels, prompt_strategies)


def get_local_quick_conditions() -> list[ExperimentCondition]:
    """
    Quick test conditions for local models.

    1 model × 2 context × 1 strategy = 2 conditions
    """
    return get_all_conditions(
        models=["llama3.1-8b"],
        context_levels=["minimal", "standard"],
        prompt_strategies=["cot"],
    )


def get_gemini_conditions() -> list[ExperimentCondition]:
    """
    Get conditions for Gemini model comparison.

    3 models × 3 context × 2 strategies = 18 conditions
    """
    return get_all_conditions(
        models=GEMINI_MODELS,
        context_levels=["minimal", "standard", "rich"],
        prompt_strategies=["direct", "cot"],
    )


def get_gemini_quick_conditions() -> list[ExperimentCondition]:
    """
    Quick test conditions for Gemini (2.0 Flash only).

    1 model × 3 context × 2 strategies = 6 conditions
    """
    return get_all_conditions(
        models=["gemini-2.0-flash"],
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

    print("\n=== Gemini Conditions ===")
    print_conditions(get_gemini_conditions())
