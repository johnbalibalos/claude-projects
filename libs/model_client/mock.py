"""
Mock LLM client for testing and development.

Supports multiple response modes:
- echo: Returns the prompt back as the response
- fixed: Returns a fixed string
- random: Returns randomly sampled text
- error: Always raises an error (for testing error handling)
"""

from __future__ import annotations

import random
import string
import time
from typing import Any, Literal

from .protocols import ModelResponse, TokenUsage

MockMode = Literal["echo", "fixed", "random", "error"]


class MockClient:
    """Mock LLM client for testing without API calls.

    Usage::

        client = MockClient(mode="echo")
        response = client.complete("Hello world")
        # response.text == "Hello world"

        client = MockClient(mode="fixed", fixed_response="42")
        response = client.complete("What is the answer?")
        # response.text == "42"
    """

    def __init__(
        self,
        mode: MockMode = "echo",
        fixed_response: str = "Mock response",
        latency: float = 0.0,
        error_rate: float = 0.0,
    ) -> None:
        self.mode = mode
        self.fixed_response = fixed_response
        self.latency = latency
        self.error_rate = error_rate
        self._call_count = 0

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a mock response."""
        self._call_count += 1

        if self.latency > 0:
            time.sleep(self.latency)

        if self.error_rate > 0 and random.random() < self.error_rate:
            raise RuntimeError(f"Mock error (call #{self._call_count})")

        text = self._generate_response(prompt)

        return ModelResponse(
            content=text,
            model="mock",
            usage=TokenUsage(
                input_tokens=len(prompt) // 4,
                output_tokens=len(text) // 4,
            ),
            raw_response={"mock": True, "mode": self.mode},
        )

    def _generate_response(self, prompt: str) -> str:
        if self.mode == "echo":
            return prompt
        elif self.mode == "fixed":
            return self.fixed_response
        elif self.mode == "random":
            length = random.randint(50, 200)
            return "".join(random.choices(string.ascii_letters + " ", k=length))
        elif self.mode == "error":
            raise RuntimeError("MockClient in error mode")
        else:
            return self.fixed_response

    @property
    def call_count(self) -> int:
        """Number of calls made to this mock."""
        return self._call_count

    def reset(self) -> None:
        """Reset call counter."""
        self._call_count = 0
