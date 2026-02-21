"""
Pre-flight token counting for LLM API calls.

Estimates prompt token counts to catch truncation issues before
making expensive API calls. Uses tiktoken for OpenAI models and
character-based heuristics for others.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Model context windows (input tokens)
# Source: Provider documentation as of 2025
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic Claude
    "claude-opus-4-5-20251101": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    # OpenAI GPT
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # Google Gemini
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
}


class PromptTooLongError(Exception):
    """Raised when a prompt exceeds the model's context window."""

    def __init__(self, model: str, estimated_tokens: int, max_tokens: int) -> None:
        self.model = model
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Prompt too long for {model}: ~{estimated_tokens:,} tokens "
            f"(max: {max_tokens:,})"
        )


@dataclass
class TokenEstimate:
    """Result of a token count estimation."""

    estimated_tokens: int
    model: str
    context_window: int | None
    method: str  # "tiktoken", "heuristic"
    fits: bool  # Whether it fits in context window


class TokenCounter:
    """Estimates token counts for LLM prompts.

    Uses tiktoken for OpenAI models when available, falls back to
    character-based heuristics (roughly 4 chars per token for English).
    """

    CHARS_PER_TOKEN = 4  # Conservative estimate

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self._encoder = None
        self._try_load_tiktoken()

    def _try_load_tiktoken(self) -> None:
        """Try to load tiktoken encoder for accurate counting."""
        if self.model and ("gpt" in self.model or "o1" in self.model):
            try:
                import tiktoken  # type: ignore[reportMissingImports]

                self._encoder = tiktoken.encoding_for_model(self.model)
            except (ImportError, KeyError):
                pass

    def estimate(self, text: str) -> TokenEstimate:
        """Estimate token count for the given text."""
        if self._encoder:
            count = len(self._encoder.encode(text))
            method = "tiktoken"
        else:
            count = len(text) // self.CHARS_PER_TOKEN
            method = "heuristic"

        context_window = MODEL_CONTEXT_WINDOWS.get(self.model or "", None)

        return TokenEstimate(
            estimated_tokens=count,
            model=self.model or "unknown",
            context_window=context_window,
            method=method,
            fits=count <= context_window if context_window else True,
        )

    def check_fits(self, text: str, *, safety_margin: float = 0.9) -> TokenEstimate:
        """Check if text fits in model context, raise if not.

        Args:
            text: The prompt text to check.
            safety_margin: Fraction of context window to use (default 90%).

        Raises:
            PromptTooLongError: If estimated tokens exceed the safe limit.
        """
        result = self.estimate(text)
        if result.context_window:
            safe_limit = int(result.context_window * safety_margin)
            if result.estimated_tokens > safe_limit:
                raise PromptTooLongError(
                    model=result.model,
                    estimated_tokens=result.estimated_tokens,
                    max_tokens=safe_limit,
                )
        return result
