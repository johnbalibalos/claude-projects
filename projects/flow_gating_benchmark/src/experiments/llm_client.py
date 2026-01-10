"""
LLM client abstraction for experiment runner.

Provides a unified interface for calling different LLM providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Ollama (local models)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    tokens_used: int = 0


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        """Call the LLM with a prompt."""
        ...

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        ...


class AnthropicClient:
    """Client for Anthropic Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise RuntimeError("ANTHROPIC_API_KEY not set")
                self._client = Anthropic()
            except ImportError as err:
                raise RuntimeError("anthropic package not installed") from err
        return self._client

    @property
    def model_id(self) -> str:
        return self.model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            content=message.content[0].text,
            model=self.model,
            tokens_used=message.usage.input_tokens + message.usage.output_tokens,
        )


class OpenAIClient:
    """Client for OpenAI GPT models."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY not set")
                self._client = OpenAI()
            except ImportError as err:
                raise RuntimeError("openai package not installed") from err
        return self._client

    @property
    def model_id(self) -> str:
        return self.model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return LLMResponse(content=content, model=self.model, tokens_used=tokens)


class OllamaClient:
    """Client for local Ollama models."""

    def __init__(self, model: str = "llama3.1:8b", base_url: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=300.0)
            except ImportError as err:
                raise RuntimeError("httpx package not installed") from err
        return self._client

    @property
    def model_id(self) -> str:
        return self.model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        client = self._get_client()
        response = client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(content=content, model=self.model)


class MockClient:
    """Mock client for dry runs and testing."""

    def __init__(self, model: str = "mock"):
        self.model = model
        self._response_template = {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "children": [{"name": "Live", "children": []}]}
            ],
        }

    @property
    def model_id(self) -> str:
        return self.model

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        import json
        return LLMResponse(
            content=json.dumps(self._response_template),
            model=self.model,
        )


def create_client(model: str, dry_run: bool = False) -> LLMClient:
    """
    Create an LLM client based on model name.

    Args:
        model: Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o", "llama3.1:8b")
        dry_run: If True, return a mock client

    Returns:
        Appropriate LLM client instance
    """
    if dry_run:
        return MockClient(model)

    model_lower = model.lower()

    if "claude" in model_lower:
        return AnthropicClient(model)
    elif "gpt" in model_lower:
        return OpenAIClient(model)
    else:
        # Default to Ollama for local models
        return OllamaClient(model)


# Model registry for easy lookup
MODEL_REGISTRY = {
    # Anthropic
    "claude-opus": "claude-opus-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    # OpenAI
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    # Ollama (local)
    "llama3.1-8b": "llama3.1:8b",
    "llama3.1-70b": "llama3.1:70b",
    "qwen2.5-7b": "qwen2.5:7b",
    "qwen2.5-72b": "qwen2.5:72b",
    "mistral-7b": "mistral:7b",
    "mixtral-8x7b": "mixtral:8x7b",
    "deepseek-r1-8b": "deepseek-r1:8b",
    "deepseek-r1-70b": "deepseek-r1:70b",
}


def resolve_model(model_name: str) -> str:
    """Resolve a model shorthand to full model ID."""
    return MODEL_REGISTRY.get(model_name, model_name)
