"""Claude CLI client for Max subscription users."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CLIConfig:
    """Configuration for Claude CLI client."""

    model: str = "claude-opus-4-20250514"
    timeout_seconds: int = 300
    max_retries: int = 3
    rate_limit: float = 1.0  # Seconds between calls


class CLIRateLimiter:
    """Thread-safe rate limiter for CLI calls."""

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_call: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Wait if needed to respect rate limit."""
        with self._lock:
            elapsed = time.time() - self._last_call
            wait_time = self.min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_call = time.time()


class ClaudeCLIClient:
    """
    Claude CLI client for Max subscription.

    Features:
    - Rate limiting (default 1s between calls)
    - Retry with exponential backoff
    - Implements JudgeModel protocol

    Usage:
        from model_client.cli_client import ClaudeCLIClient, CLIConfig

        # Default config (Opus, 1s rate limit)
        client = ClaudeCLIClient()

        # Custom config
        client = ClaudeCLIClient(CLIConfig(rate_limit=2.0, model="claude-sonnet-4-20250514"))

        # Generate (implements JudgeModel protocol)
        response = client.generate("What is 2+2?")
    """

    def __init__(self, config: CLIConfig | None = None):
        self.config = config or CLIConfig()
        self._rate_limiter = CLIRateLimiter(self.config.rate_limit)
        self._validate_cli_available()

    def _validate_cli_available(self) -> None:
        """Check that claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise CLIError(f"Claude CLI error: {result.stderr}")
        except FileNotFoundError:
            raise CLIError("Claude CLI not found")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate response using Claude CLI.

        Implements JudgeModel protocol from hypothesis_pipeline.llm_judge.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (ignored by CLI, always deterministic)

        Returns:
            Generated text response
        """
        self._rate_limiter.wait()
        return self._call_with_retry(prompt, max_tokens)

    def _call_with_retry(self, prompt: str, max_tokens: int) -> str:
        """Execute CLI call with exponential backoff retry."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return self._call_cli(prompt, max_tokens)
            except CLIError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    backoff = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"CLI attempt {attempt + 1} failed, retrying in {backoff}s: {e}"
                    )
                    time.sleep(backoff)
        raise last_error or CLIError("Max retries exceeded")

    def _call_cli(self, prompt: str, max_tokens: int) -> str:
        """Execute single CLI call."""
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-tokens",
            str(max_tokens),
            "--model",
            self.config.model,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            if result.returncode != 0:
                raise CLIError(f"CLI failed: {result.stderr}")

            return self._parse_response(result.stdout)

        except subprocess.TimeoutExpired:
            raise CLIError(f"Timeout after {self.config.timeout_seconds}s")

    def _parse_response(self, stdout: str) -> str:
        """Parse JSON response from CLI."""
        try:
            data = json.loads(stdout)
            return data.get("result", data.get("content", ""))
        except json.JSONDecodeError:
            # Fall back to raw output
            return stdout.strip()


class CLIError(Exception):
    """Exception for CLI-related errors."""

    pass
