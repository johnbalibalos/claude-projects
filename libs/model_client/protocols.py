"""
Protocol definitions for model clients.

Provides a unified interface for different LLM providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class TokenUsage:
    """Token usage statistics from an API call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Any = None  # Provider-specific response object

    @property
    def cost_usd(self) -> float:
        """Estimate cost based on token usage and model."""
        # Pricing per 1M tokens (as of 2025)
        pricing = {
            "claude-sonnet-4-20250514": (3.0, 15.0),
            "claude-opus-4-20250514": (15.0, 75.0),
            "claude-3-5-sonnet-20241022": (3.0, 15.0),
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
        }

        input_price, output_price = pricing.get(self.model, (3.0, 15.0))
        input_cost = (self.usage.input_tokens / 1_000_000) * input_price
        output_cost = (self.usage.output_tokens / 1_000_000) * output_price
        return input_cost + output_cost


@runtime_checkable
class ModelClient(Protocol):
    """Protocol for LLM client implementations."""

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options

        Returns:
            Standardized ModelResponse
        """
        ...


@runtime_checkable
class AsyncModelClient(Protocol):
    """Protocol for async LLM client implementations."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response asynchronously."""
        ...


Provider = Literal["anthropic", "openai", "litellm"]
