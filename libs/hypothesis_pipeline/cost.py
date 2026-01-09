"""
Cost estimation for hypothesis pipeline experiments.

Estimates API costs based on model pricing and expected token usage.
Supports multiple model providers with extensible pricing configuration.

Current Support:
- Anthropic: Claude Sonnet 4, Claude Opus 4

Future Support (see ADDING_NEW_MODELS below):
- OpenAI: GPT-4o, GPT-4o-mini
- Google: Gemini 1.5 Pro, Gemini 1.5 Flash
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig


# =============================================================================
# MODEL PRICING (per 1M tokens, in USD)
# =============================================================================

@dataclass
class ModelPricing:
    """Pricing for a model."""
    input_per_million: float  # Cost per 1M input tokens
    output_per_million: float  # Cost per 1M output tokens
    provider: str  # Provider name for display

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_per_million
        return input_cost + output_cost


# Current model pricing (as of January 2025)
MODEL_PRICING: dict[str, ModelPricing] = {
    # Anthropic Models
    "claude-sonnet-4-20250514": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        provider="Anthropic",
    ),
    "claude-opus-4-20250514": ModelPricing(
        input_per_million=15.0,
        output_per_million=75.0,
        provider="Anthropic",
    ),
    # Legacy model names (aliases)
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        provider="Anthropic",
    ),
    "claude-3-opus-20240229": ModelPricing(
        input_per_million=15.0,
        output_per_million=75.0,
        provider="Anthropic",
    ),

    # =========================================================================
    # ADDING NEW MODELS
    # =========================================================================
    # To add OpenAI models, uncomment and update pricing:
    #
    # "gpt-4o": ModelPricing(
    #     input_per_million=2.50,
    #     output_per_million=10.0,
    #     provider="OpenAI",
    # ),
    # "gpt-4o-mini": ModelPricing(
    #     input_per_million=0.15,
    #     output_per_million=0.60,
    #     provider="OpenAI",
    # ),
    # "gpt-4-turbo": ModelPricing(
    #     input_per_million=10.0,
    #     output_per_million=30.0,
    #     provider="OpenAI",
    # ),
    #
    # To add Google Gemini models:
    #
    # "gemini-1.5-pro": ModelPricing(
    #     input_per_million=1.25,
    #     output_per_million=5.0,
    #     provider="Google",
    # ),
    # "gemini-1.5-flash": ModelPricing(
    #     input_per_million=0.075,
    #     output_per_million=0.30,
    #     provider="Google",
    # ),
    # "gemini-2.0-flash": ModelPricing(
    #     input_per_million=0.10,
    #     output_per_million=0.40,
    #     provider="Google",
    # ),
    # =========================================================================
}

# Default token estimates by reasoning type
DEFAULT_TOKEN_ESTIMATES: dict[str, dict[str, int]] = {
    "direct": {"input": 1000, "output": 300},
    "cot": {"input": 1500, "output": 600},
    "wot": {"input": 2000, "output": 1000},
    "few_shot": {"input": 2500, "output": 400},
    "self_consistency": {"input": 1500, "output": 1500},  # Multiple samples
    "react": {"input": 2000, "output": 1500},  # Tool calls
}

# Context level multipliers (relative to base tokens)
CONTEXT_MULTIPLIERS: dict[str, float] = {
    "none": 0.8,
    "minimal": 1.0,
    "standard": 1.3,
    "rich": 1.8,
    "oracle": 2.0,
}

# RAG mode additional tokens
RAG_ADDITIONAL_TOKENS: dict[str, int] = {
    "none": 0,
    "vector": 500,
    "hybrid": 600,
    "oracle": 400,
    "negative": 500,
}


# =============================================================================
# COST ESTIMATION
# =============================================================================

@dataclass
class CostEstimate:
    """Detailed cost estimate for an experiment."""

    # Counts
    n_conditions: int
    n_test_cases: int
    n_bootstrap_runs: int
    total_api_calls: int

    # Token estimates
    total_input_tokens: int
    total_output_tokens: int
    avg_input_tokens_per_call: int
    avg_output_tokens_per_call: int

    # Cost breakdown
    input_cost: float
    output_cost: float
    total_cost: float

    # Per-model breakdown
    cost_by_model: dict[str, float]

    # Metadata
    models: list[str]
    warnings: list[str]

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            "=" * 55,
            "COST ESTIMATE",
            "=" * 55,
            f"Conditions:        {self.n_conditions}",
            f"Test cases:        {self.n_test_cases}",
            f"Bootstrap runs:    {self.n_bootstrap_runs}",
            f"Total API calls:   {self.total_api_calls}",
            "",
            f"Est. input tokens:  {self.total_input_tokens:,}",
            f"Est. output tokens: {self.total_output_tokens:,}",
            "",
        ]

        # Cost by model
        if len(self.cost_by_model) > 1:
            lines.append("Cost by model:")
            for model, cost in self.cost_by_model.items():
                short_name = model.split("-")[1] if "-" in model else model
                lines.append(f"  {short_name}: ${cost:.2f}")
            lines.append("")

        lines.extend([
            f"Input cost:    ${self.input_cost:.2f}",
            f"Output cost:   ${self.output_cost:.2f}",
            f"{'─' * 25}",
            f"TOTAL COST:    ${self.total_cost:.2f}",
            "=" * 55,
        ])

        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ! {warning}")

        return "\n".join(lines)


def estimate_experiment_cost(
    config: PipelineConfig,
    n_test_cases: int,
    avg_input_tokens: int | None = None,
    avg_output_tokens: int | None = None,
) -> CostEstimate:
    """
    Estimate the cost of running an experiment.

    Args:
        config: Pipeline configuration
        n_test_cases: Number of test cases to run
        avg_input_tokens: Override average input tokens per call
        avg_output_tokens: Override average output tokens per call

    Returns:
        Detailed cost estimate
    """
    warnings = []

    # Calculate number of conditions
    n_conditions = (
        len(config.models) *
        len(config.reasoning_types) *
        len(config.context_levels) *
        len(config.rag_modes) *
        len(config.tool_configs)
    )

    n_bootstrap = config.n_bootstrap_runs
    total_calls = n_conditions * n_test_cases * n_bootstrap

    # Estimate tokens if not provided
    if avg_input_tokens is None or avg_output_tokens is None:
        est_input, est_output = _estimate_average_tokens(config)
        avg_input_tokens = avg_input_tokens or est_input
        avg_output_tokens = avg_output_tokens or est_output

    total_input = total_calls * avg_input_tokens
    total_output = total_calls * avg_output_tokens

    # Calculate costs by model
    cost_by_model = {}
    total_input_cost = 0.0
    total_output_cost = 0.0

    calls_per_model = total_calls // len(config.models)

    for model in config.models:
        pricing = MODEL_PRICING.get(model)

        if pricing is None:
            warnings.append(f"Unknown model '{model}' - using Sonnet pricing as fallback")
            pricing = MODEL_PRICING["claude-sonnet-4-20250514"]

        model_input = calls_per_model * avg_input_tokens
        model_output = calls_per_model * avg_output_tokens

        input_cost = (model_input / 1_000_000) * pricing.input_per_million
        output_cost = (model_output / 1_000_000) * pricing.output_per_million

        cost_by_model[model] = input_cost + output_cost
        total_input_cost += input_cost
        total_output_cost += output_cost

    return CostEstimate(
        n_conditions=n_conditions,
        n_test_cases=n_test_cases,
        n_bootstrap_runs=n_bootstrap,
        total_api_calls=total_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        avg_input_tokens_per_call=avg_input_tokens,
        avg_output_tokens_per_call=avg_output_tokens,
        input_cost=total_input_cost,
        output_cost=total_output_cost,
        total_cost=total_input_cost + total_output_cost,
        cost_by_model=cost_by_model,
        models=list(config.models),
        warnings=warnings,
    )


def _estimate_average_tokens(config: PipelineConfig) -> tuple[int, int]:
    """Estimate average tokens based on config dimensions."""
    # Base tokens from reasoning types
    input_tokens = []
    output_tokens = []

    for rt in config.reasoning_types:
        estimates = DEFAULT_TOKEN_ESTIMATES.get(rt.value, {"input": 1500, "output": 500})
        input_tokens.append(estimates["input"])
        output_tokens.append(estimates["output"])

    avg_input = sum(input_tokens) / len(input_tokens) if input_tokens else 1500
    avg_output = sum(output_tokens) / len(output_tokens) if output_tokens else 500

    # Apply context multipliers
    context_mult = []
    for cl in config.context_levels:
        context_mult.append(CONTEXT_MULTIPLIERS.get(cl.value, 1.0))
    avg_context_mult = sum(context_mult) / len(context_mult) if context_mult else 1.0

    avg_input *= avg_context_mult

    # Add RAG tokens
    rag_tokens = []
    for rm in config.rag_modes:
        rag_tokens.append(RAG_ADDITIONAL_TOKENS.get(rm.value, 0))
    avg_rag = sum(rag_tokens) / len(rag_tokens) if rag_tokens else 0

    avg_input += avg_rag

    # Add tool overhead if enabled
    if any(tools for tools in config.tool_configs):
        avg_input += 200  # Tool descriptions
        avg_output += 300  # Tool calls

    return int(avg_input), int(avg_output)


# =============================================================================
# USER CONFIRMATION
# =============================================================================

def confirm_experiment_cost(
    config: PipelineConfig,
    n_test_cases: int,
    avg_input_tokens: int | None = None,
    avg_output_tokens: int | None = None,
    auto_confirm_under: float | None = None,
) -> bool:
    """
    Show cost estimate and require user confirmation before running.

    Args:
        config: Pipeline configuration
        n_test_cases: Number of test cases
        avg_input_tokens: Override average input tokens
        avg_output_tokens: Override average output tokens
        auto_confirm_under: Auto-confirm if cost is under this amount (in USD)

    Returns:
        True if user confirms, False otherwise
    """
    estimate = estimate_experiment_cost(
        config=config,
        n_test_cases=n_test_cases,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
    )

    print(estimate.format_summary())

    # Auto-confirm for small costs
    if auto_confirm_under is not None and estimate.total_cost < auto_confirm_under:
        print(f"\nAuto-confirmed (cost < ${auto_confirm_under:.2f})")
        return True

    # Interactive confirmation
    print()
    try:
        response = input("Proceed with experiment? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return False


def get_model_pricing_table() -> str:
    """Get a formatted table of all model pricing."""
    lines = [
        "MODEL PRICING (per 1M tokens)",
        "=" * 60,
        f"{'Model':<35} {'Input':>10} {'Output':>10}",
        "-" * 60,
    ]

    for model, pricing in sorted(MODEL_PRICING.items()):
        lines.append(
            f"{model:<35} ${pricing.input_per_million:>8.2f} ${pricing.output_per_million:>8.2f}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
