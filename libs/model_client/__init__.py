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
from .cli_client import ClaudeCLIClient, CLIConfig, CLIError
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
    # API clients
    "AnthropicClient",
    "OpenAIClient",
    "GeminiClient",
    "LiteLLMClient",
    "ClientConfig",
    # CLI client (for Max subscription)
    "ClaudeCLIClient",
    "CLIConfig",
    "CLIError",
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


class GeminiClient:
    """Google Gemini API client wrapper."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai  # type: ignore[import-not-found]

            api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    @with_retry(max_attempts=3, initial_delay=2.0, max_delay=60.0)
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **_kwargs: Any,
    ) -> ModelResponse:
        from google.genai import types  # type: ignore[import-not-found]

        start_time = time.time()
        client = self._get_client()

        # Convert messages to Gemini format (simple concatenation for now)
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" if m["role"] != "user" else m["content"]
            for m in messages
        )

        # Relaxed safety settings for biomedical/scientific content
        # pyright: ignore - google.genai types not properly stubbed
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",  # pyright: ignore
                threshold="BLOCK_ONLY_HIGH",  # pyright: ignore
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",  # pyright: ignore
                threshold="BLOCK_ONLY_HIGH",  # pyright: ignore
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",  # pyright: ignore
                threshold="BLOCK_ONLY_HIGH",  # pyright: ignore
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  # pyright: ignore
                threshold="BLOCK_ONLY_HIGH",  # pyright: ignore
            ),
        ]

        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=safety_settings,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=generation_config,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Handle blocked responses
        if not response.candidates:
            return ModelResponse(
                content="[BLOCKED: No candidates returned]",
                model=model,
                usage=TokenUsage(input_tokens=0, output_tokens=0),
                finish_reason="blocked",
                latency_ms=latency_ms,
                raw_response=response,
            )

        candidate = response.candidates[0]
        finish_reason = "stop"
        if candidate.finish_reason:
            finish_reason = candidate.finish_reason.name.lower()
            if finish_reason not in ("stop", "max_tokens"):
                return ModelResponse(
                    content=f"[BLOCKED: {candidate.finish_reason.name}]",
                    model=model,
                    usage=TokenUsage(input_tokens=0, output_tokens=0),
                    finish_reason=finish_reason,
                    latency_ms=latency_ms,
                    raw_response=response,
                )

        content = response.text if response.text else ""

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return ModelResponse(
            content=content,
            model=model,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            finish_reason=finish_reason,
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
        **_kwargs: Any,
    ) -> Any:
        """
        Get or create a client for the specified provider.

        Args:
            provider: LLM provider ("anthropic", "openai", "gemini", "litellm")
            model: Model name (used for cache key)
            api_key: Optional API key (defaults to env var)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **_kwargs: Additional provider-specific options (reserved for future use)

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
            elif provider == "gemini":
                cls._clients[cache_key] = GeminiClient(config)
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
