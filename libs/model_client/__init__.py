"""
Unified model client registry for LLM API access.

Provides a consistent interface for different LLM providers with:
- Automatic retry with exponential backoff
- Token usage tracking
- Cost estimation
- Connection management

Example usage:
    from model_client import ModelRegistry

    # Get a client
    client = ModelRegistry.get_client("anthropic", "claude-sonnet-4-20250514")

    # Make a request
    response = client.generate([{"role": "user", "content": "Hello!"}])
    print(response.content)
    print(f"Cost: ${response.cost_usd:.4f}")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from .cache import CachedClient, ResponseCache, get_global_cache
from .protocols import ModelClient, ModelResponse, Provider, TokenUsage
from .retry import with_retry, with_retry_async

__all__ = [
    "ModelRegistry",
    "ModelClient",
    "ModelResponse",
    "TokenUsage",
    "Provider",
    "with_retry",
    "with_retry_async",
    "CachedClient",
    "ResponseCache",
    "get_global_cache",
]

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for a model client."""

    api_key: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    base_url: str | None = None


class AnthropicClient:
    """Anthropic API client wrapper."""

    def __init__(self, config: ClientConfig):
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
        self.config = config

    @with_retry(max_attempts=3, initial_delay=2.0, max_delay=60.0)
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        start_time = time.time()

        response = self.client.messages.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract text from first content block
        content = ""
        if response.content:
            first_block = response.content[0]
            if hasattr(first_block, "text"):
                content = first_block.text

        return ModelResponse(
            content=content,
            model=model,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason or "stop",
            latency_ms=latency_ms,
            raw_response=response,
        )


class OpenAIClient:
    """OpenAI API client wrapper."""

    def __init__(self, config: ClientConfig):
        from openai import OpenAI  # type: ignore[import-not-found]

        self.client = OpenAI(
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=config.timeout,
            max_retries=0,
        )
        self.config = config

    @with_retry(max_attempts=3, initial_delay=2.0, max_delay=60.0)
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        latency_ms = (time.time() - start_time) * 1000
        choice = response.choices[0] if response.choices else None

        return ModelResponse(
            content=choice.message.content if choice else "",
            model=model,
            usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
            finish_reason=choice.finish_reason if choice else "stop",
            latency_ms=latency_ms,
            raw_response=response,
        )


class LiteLLMClient:
    """LiteLLM client wrapper for multi-provider support."""

    def __init__(self, config: ClientConfig):
        self.config = config

    @with_retry(max_attempts=3, initial_delay=2.0, max_delay=60.0)
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        import litellm  # type: ignore[import-not-found]

        start_time = time.time()

        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        latency_ms = (time.time() - start_time) * 1000
        choice = response.choices[0] if response.choices else None

        return ModelResponse(
            content=choice.message.content if choice else "",
            model=model,
            usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
            finish_reason=choice.finish_reason if choice else "stop",
            latency_ms=latency_ms,
            raw_response=response,
        )


class ModelRegistry:
    """
    Factory for creating model clients with consistent configuration.

    Provides singleton-like behavior - clients are cached and reused.
    """

    _clients: dict[str, Any] = {}

    @classmethod
    def get_client(
        cls,
        provider: Provider,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """
        Get or create a client for the specified provider.

        Args:
            provider: LLM provider ("anthropic", "openai", "litellm")
            model: Model name (used for cache key)
            api_key: Optional API key (defaults to env var)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional provider-specific options

        Returns:
            Configured client instance
        """
        cache_key = f"{provider}:{model or 'default'}"

        if cache_key not in cls._clients:
            config = ClientConfig(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )

            if provider == "anthropic":
                cls._clients[cache_key] = AnthropicClient(config)
            elif provider == "openai":
                cls._clients[cache_key] = OpenAIClient(config)
            elif provider == "litellm":
                cls._clients[cache_key] = LiteLLMClient(config)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            logger.debug(f"Created new client for {cache_key}")

        return cls._clients[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the client cache."""
        cls._clients.clear()

    @classmethod
    def generate(
        cls,
        provider: Provider,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Convenience method to generate a response directly.

        Args:
            provider: LLM provider
            model: Model name
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional options

        Returns:
            ModelResponse with content, usage, and cost
        """
        client = cls.get_client(provider, model)
        return client.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
