"""
LLM client abstraction for experiment runner.

Provides a unified interface for calling different LLM providers:
- Anthropic (Claude) via API
- OpenAI (GPT) via API
- Google Gemini via API
- Ollama (local models)
- Claude CLI (experimental, local dev only)
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import warnings
from dataclasses import dataclass
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from experiments.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Response from an LLM call.

    Attributes:
        content: The text response from the model.
        model: Model identifier that generated this response.
        tokens_used: Total tokens consumed (input + output). May be estimated for some providers.
    """

    content: str
    model: str
    tokens_used: int = 0


class LLMClient(Protocol):
    """
    Protocol defining the interface for LLM clients.

    All LLM client implementations must provide:
    - call(): Make a synchronous call to the LLM
    - model_id: Return a unique identifier for the model

    This protocol enables dependency injection and makes testing easier
    by allowing mock implementations.
    """

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        """Call the LLM with a prompt."""
        ...

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        ...


class AnthropicClient:
    """
    Client for Anthropic Claude models via the official API.

    Requires ANTHROPIC_API_KEY environment variable to be set.

    Attributes:
        model: Model identifier (e.g., "claude-sonnet-4-20250514").
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic

                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise ConfigurationError(
                        "ANTHROPIC_API_KEY not set. "
                        "Set this environment variable with your Anthropic API key."
                    )
                self._client = Anthropic()
            except ImportError as err:
                raise ConfigurationError(
                    "anthropic package not installed. Install with: pip install anthropic"
                ) from err
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
    """
    Client for OpenAI GPT models via the official API.

    Requires OPENAI_API_KEY environment variable to be set.

    Attributes:
        model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini").
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                if not os.environ.get("OPENAI_API_KEY"):
                    raise ConfigurationError(
                        "OPENAI_API_KEY not set. "
                        "Set this environment variable with your OpenAI API key."
                    )
                self._client = OpenAI()
            except ImportError as err:
                raise ConfigurationError(
                    "openai package not installed. Install with: pip install openai"
                ) from err
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


class GeminiClient:
    """
    Client for Google Gemini models via the official API.

    Requires GOOGLE_API_KEY environment variable to be set.

    Attributes:
        model: Model identifier (e.g., "gemini-2.0-flash", "gemini-2.5-pro").
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai

                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ConfigurationError(
                        "GOOGLE_API_KEY not set. "
                        "Set this environment variable with your Google API key."
                    )
                self._client = genai.Client(api_key=api_key)
            except ImportError as err:
                raise ConfigurationError(
                    "google-genai package not installed. Install with: pip install google-genai"
                ) from err
        return self._client

    @property
    def model_id(self) -> str:
        return self.model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        from google.genai import types

        client = self._get_client()

        # Relaxed safety settings for biomedical content
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]

        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=safety_settings,
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=generation_config,
        )

        # Handle blocked responses
        if not response.candidates:
            return LLMResponse(
                content="[BLOCKED: No candidates returned]",
                model=self.model,
                tokens_used=0,
            )

        candidate = response.candidates[0]
        if candidate.finish_reason and candidate.finish_reason.name not in ("STOP", "MAX_TOKENS"):
            return LLMResponse(
                content=f"[BLOCKED: {candidate.finish_reason.name}]",
                model=self.model,
                tokens_used=0,
            )

        content = response.text if response.text else ""

        # Handle empty response (likely safety filtering that didn't trigger finish_reason)
        if not content.strip():
            # Try to get more info from the response
            finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
            content = f"[EMPTY_RESPONSE: finish_reason={finish_reason}]"

        tokens = 0
        if response.usage_metadata:
            tokens = (response.usage_metadata.prompt_token_count or 0) + \
                     (response.usage_metadata.candidates_token_count or 0)

        return LLMResponse(content=content, model=self.model, tokens_used=tokens)


# Map model names to CLI aliases
CLI_MODEL_MAP = {
    "claude-sonnet-4-20250514": "sonnet",
    "claude-sonnet": "sonnet",
    "claude-opus-4-20250514": "opus",
    "claude-opus": "opus",
    "claude-3-5-haiku-20241022": "haiku",
    "claude-haiku": "haiku",
}


class ClaudeCLIClient:
    """
    EXPERIMENTAL: Client for Claude via CLI (uses Claude Max OAuth subscription).

    ⚠️  WARNING: This client is for LOCAL DEVELOPMENT ONLY. ⚠️

    Known limitations:
    - Wraps the Claude CLI tool meant for interactive terminal use
    - No control over max_tokens or temperature parameters
    - Token counts are estimated, not actual
    - Dependent on CLI output format (may break if Anthropic updates CLI)
    - Requires Node.js and global npm package installation
    - Not suitable for production or CI environments

    For production use, prefer AnthropicClient with proper API authentication.

    Attributes:
        model: Model identifier (e.g., "claude-sonnet-4-20250514").
        delay_seconds: Minimum delay between calls for rate limiting.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        delay_seconds: float = 0.5,
    ):
        warnings.warn(
            "ClaudeCLIClient is experimental and intended for local development only. "
            "For production use, prefer AnthropicClient with API authentication.",
            category=UserWarning,
            stacklevel=2,
        )
        self.model = model
        self.delay_seconds = delay_seconds
        self._cli_model = CLI_MODEL_MAP.get(model, model)
        self._last_call_time = 0.0
        self._verify_cli()

    def _verify_cli(self):
        """Verify claude CLI is available and authenticated."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise ConfigurationError("claude CLI not found or not working")
        except FileNotFoundError as err:
            raise ConfigurationError(
                "claude CLI not installed. Install with: npm install -g @anthropic-ai/claude-code"
            ) from err

    @property
    def model_id(self) -> str:
        return f"{self.model}-cli"

    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limiting between calls."""
        if self.delay_seconds > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.delay_seconds:
                time.sleep(self.delay_seconds - elapsed)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        """Call Claude via CLI using --print flag for non-interactive output."""
        _ = max_tokens, temperature  # CLI doesn't support these directly

        self._wait_for_rate_limit()

        try:
            # Use stdin for prompt (handles long prompts, avoids shell arg limits)
            result = subprocess.run(
                ["claude", "-p", "--model", self._cli_model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            self._last_call_time = time.time()

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                raise RuntimeError(f"Claude CLI error (code {result.returncode}): {error_msg}")

            content = result.stdout.strip()

            # Estimate tokens (CLI doesn't return actual counts)
            # Rough estimate: 1 token ≈ 4 chars
            estimated_tokens = (len(prompt) + len(content)) // 4

            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=estimated_tokens,
            )

        except subprocess.TimeoutExpired as err:
            raise RuntimeError("Claude CLI call timed out after 5 minutes") from err


class OllamaClient:
    """
    Client for local Ollama models.

    Connects to a local Ollama server for inference. No API key required.

    Attributes:
        model: Ollama model identifier (e.g., "llama3.1:8b", "mistral:7b").
        base_url: Ollama server URL. Defaults to OLLAMA_BASE_URL env var or localhost.
    """

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
                raise ConfigurationError(
                    "httpx package not installed. Install with: pip install httpx"
                ) from err
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


class MockMode:
    """Mock response modes for different testing scenarios."""
    VALID = "valid"          # Return valid parseable response
    EMPTY = "empty"          # Return empty string (test parse error handling)
    ECHO = "echo"            # Echo back the prompt (test prompt construction)
    MALFORMED = "malformed"  # Return invalid JSON (test parse error handling)
    ERROR = "error"          # Simulate API error


class MockClient:
    """
    Mock client for dry runs and testing.

    Returns configurable responses without making any API calls.
    Useful for testing different scenarios:

    - VALID: Returns valid gating hierarchy (tests happy path)
    - EMPTY: Returns empty string (tests parse failure handling)
    - ECHO: Returns the prompt (tests prompt construction - inspect what was sent)
    - MALFORMED: Returns invalid JSON (tests parse error recovery)
    - ERROR: Raises exception (tests error handling)

    Features:
    - Zero latency by default (no sleeps, no rate limiting)
    - Multiple response modes for testing different scenarios
    - Optional simulated latency for timing tests

    Attributes:
        model: Model identifier (always "mock" or custom string).
        mode: Response mode (see MockMode).
        simulate_latency: If > 0, sleep this many seconds per call.
    """

    # Class-level flag to disable ALL delays globally for dry runs
    FAST_MODE: bool = True

    def __init__(
        self,
        model: str = "mock",
        mode: str = MockMode.VALID,
        simulate_latency: float = 0.0,
    ):
        self.model = model
        self.mode = mode
        self.simulate_latency = simulate_latency if not MockClient.FAST_MODE else 0.0
        self._call_count = 0
        self._last_prompt: str | None = None

        # Valid response template
        self._valid_response = {
            "name": "All Events",
            "children": [
                {
                    "name": "Singlets",
                    "markers": ["FSC-A", "FSC-H"],
                    "children": [
                        {
                            "name": "Live",
                            "markers": ["Live/Dead"],
                            "children": [
                                {
                                    "name": "CD3+ T cells",
                                    "markers": ["CD3"],
                                    "children": []
                                }
                            ]
                        }
                    ]
                }
            ],
        }

    @property
    def model_id(self) -> str:
        return f"{self.model}-{self.mode}"

    @property
    def call_count(self) -> int:
        """Number of calls made (useful for testing)."""
        return self._call_count

    @property
    def last_prompt(self) -> str | None:
        """Last prompt received (useful for testing prompt construction)."""
        return self._last_prompt

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        """
        Return mock response based on configured mode.

        In ECHO mode, returns the prompt itself - useful for verifying
        prompt construction without API calls.
        """
        import json

        self._call_count += 1
        self._last_prompt = prompt

        # Optional simulated latency for timing tests
        if self.simulate_latency > 0:
            time.sleep(self.simulate_latency)

        # Generate response based on mode
        if self.mode == MockMode.ERROR:
            raise RuntimeError("Simulated API error (MockClient in ERROR mode)")

        if self.mode == MockMode.EMPTY:
            content = ""
        elif self.mode == MockMode.ECHO:
            # Return prompt wrapped in a way that shows what was sent
            content = f"=== PROMPT RECEIVED ({len(prompt)} chars) ===\n{prompt[:2000]}"
            if len(prompt) > 2000:
                content += f"\n... (truncated, {len(prompt) - 2000} more chars)"
        elif self.mode == MockMode.MALFORMED:
            content = '{"name": "All Events", "children": [INVALID JSON HERE'
        else:  # VALID
            content = json.dumps(self._valid_response, indent=2)

        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=0,
        )


def create_client(
    model: str,
    dry_run: bool = False,
    use_cli: bool = False,
    mock_mode: str = MockMode.VALID,
) -> LLMClient:
    """
    Create an LLM client based on model name.

    Args:
        model: Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o", "llama3.1:8b")
               Use "-cli" suffix (e.g., "claude-sonnet-cli") to use Claude Max OAuth
        dry_run: If True, return a mock client (no API calls, instant responses)
        use_cli: If True and model is Claude, use CLI client (Claude Max OAuth)
        mock_mode: When dry_run=True, controls mock behavior:
            - MockMode.VALID: Return valid parseable response (default)
            - MockMode.EMPTY: Return empty string (test parse errors)
            - MockMode.ECHO: Return the prompt (inspect what would be sent)
            - MockMode.MALFORMED: Return invalid JSON (test error handling)
            - MockMode.ERROR: Raise exception (test error handling)

    Returns:
        Appropriate LLM client instance
    """
    if dry_run:
        return MockClient(model, mode=mock_mode)

    model_lower = model.lower()

    # Check for -cli suffix to auto-enable CLI mode
    if model_lower.endswith("-cli"):
        use_cli = True
        model = model[:-4]  # Strip -cli suffix
        model_lower = model.lower()
        # Resolve shorthand to full model ID
        model = resolve_model(model)

    if "claude" in model_lower:
        if use_cli:
            return ClaudeCLIClient(model)
        return AnthropicClient(model)
    elif "gpt" in model_lower:
        return OpenAIClient(model)
    elif "gemini" in model_lower:
        return GeminiClient(model)
    else:
        # Default to Ollama for local models
        return OllamaClient(model)


# Model registry for shorthand → full model ID resolution
#
# NOTE: The "-cli" suffix distinguishes API vs CLI routing:
#   - "claude-opus"     → AnthropicClient (API, billed per token)
#   - "claude-opus-cli" → ClaudeCLIClient (OAuth, Max subscription)
#
# The create_client() function strips "-cli" suffix BEFORE calling resolve_model(),
# so the -cli entries below are only for documentation/test consistency.
# See conditions.py:MODELS for the authoritative experiment condition list.
#
MODEL_REGISTRY = {
    # Anthropic API
    "claude-opus": "claude-opus-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    # Anthropic CLI (these keys exist for test coverage but are never looked up -
    # create_client() strips -cli suffix before resolution)
    "claude-opus-cli": "claude-opus-4-20250514",
    "claude-sonnet-cli": "claude-sonnet-4-20250514",
    "claude-haiku-cli": "claude-3-5-haiku-20241022",
    # OpenAI
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    # Google Gemini
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
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
