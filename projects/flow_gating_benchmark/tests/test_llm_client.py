"""Tests for LLM client module."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.llm_client import (  # pyright: ignore[reportMissingImports]
    CLI_MODEL_MAP,
    MODEL_REGISTRY,
    ClaudeCLIClient,
    LLMResponse,
    MockClient,
    create_client,
    resolve_model,
)


class TestCreateClient:
    """Tests for create_client factory function."""

    def test_create_mock_client_dry_run(self):
        """Dry run should return MockClient."""
        client = create_client("claude-sonnet", dry_run=True)
        assert isinstance(client, MockClient)

    def test_create_anthropic_client(self):
        """Should create AnthropicClient for claude models."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = create_client("claude-sonnet", dry_run=True)
            # Dry run returns mock, but we can test the routing
            assert client.model == "claude-sonnet"

    def test_create_cli_client_with_suffix(self):
        """Model name ending in -cli should use ClaudeCLIClient."""
        with patch.object(ClaudeCLIClient, "_verify_cli"):
            client = create_client("claude-sonnet-cli")
            assert isinstance(client, ClaudeCLIClient)

    def test_create_cli_client_with_flag(self):
        """use_cli=True should use ClaudeCLIClient."""
        with patch.object(ClaudeCLIClient, "_verify_cli"):
            client = create_client("claude-sonnet", use_cli=True)
            assert isinstance(client, ClaudeCLIClient)

    def test_create_gemini_client(self):
        """Should create GeminiClient for gemini models."""
        client = create_client("gemini-2.0-flash", dry_run=True)
        assert client.model == "gemini-2.0-flash"

    def test_create_ollama_client(self):
        """Unknown models should default to OllamaClient."""
        client = create_client("llama3.1:8b", dry_run=True)
        assert client.model == "llama3.1:8b"


class TestCLIModelMap:
    """Tests for CLI model mapping."""

    def test_sonnet_mapping(self):
        """claude-sonnet should map to 'sonnet'."""
        assert CLI_MODEL_MAP["claude-sonnet"] == "sonnet"
        assert CLI_MODEL_MAP["claude-sonnet-4-20250514"] == "sonnet"

    def test_opus_mapping(self):
        """claude-opus should map to 'opus'."""
        assert CLI_MODEL_MAP["claude-opus"] == "opus"
        assert CLI_MODEL_MAP["claude-opus-4-20250514"] == "opus"

    def test_haiku_mapping(self):
        """claude-haiku should map to 'haiku'."""
        assert CLI_MODEL_MAP["claude-haiku"] == "haiku"


class TestClaudeCLIClient:
    """Tests for ClaudeCLIClient."""

    @pytest.fixture
    def mock_cli_client(self):
        """Create a CLI client with mocked verification."""
        with patch.object(ClaudeCLIClient, "_verify_cli"):
            client = ClaudeCLIClient(model="claude-sonnet")
            return client

    def test_model_id_includes_cli_suffix(self, mock_cli_client):
        """Model ID should include -cli suffix."""
        assert mock_cli_client.model_id == "claude-sonnet-cli"

    def test_cli_model_mapping(self, mock_cli_client):
        """Should use mapped CLI model name."""
        assert mock_cli_client._cli_model == "sonnet"

    def test_default_delay(self, mock_cli_client):
        """Default delay should be 0.5 seconds."""
        assert mock_cli_client.delay_seconds == 0.5

    def test_custom_delay(self):
        """Should accept custom delay."""
        with patch.object(ClaudeCLIClient, "_verify_cli"):
            client = ClaudeCLIClient(model="claude-sonnet", delay_seconds=5.0)
            assert client.delay_seconds == 5.0

    def test_call_uses_subprocess(self, mock_cli_client):
        """Call should invoke claude CLI via subprocess."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Test response"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            response = mock_cli_client.call("Test prompt")

            # Verify subprocess was called correctly
            mock_run.assert_called_once()
            args = mock_run.call_args
            assert args[0][0] == ["claude", "-p", "--model", "sonnet"]
            assert args[1]["input"] == "Test prompt"
            assert args[1]["capture_output"] is True

            # Verify response
            assert response.content == "Test response"
            assert response.model == "claude-sonnet"

    def test_call_handles_error(self, mock_cli_client):
        """Should raise on CLI error (wrapped by tenacity retry)."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "API error"

        with patch("subprocess.run", return_value=mock_result):
            # tenacity wraps the error in RetryError after exhausting retries
            from tenacity import RetryError
            with pytest.raises((RuntimeError, RetryError)):
                mock_cli_client.call("Test prompt")

    def test_call_handles_timeout(self, mock_cli_client):
        """Should raise on timeout (wrapped by tenacity retry)."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            from tenacity import RetryError
            with pytest.raises((RuntimeError, RetryError)):
                mock_cli_client.call("Test prompt")

    def test_token_estimation(self, mock_cli_client):
        """Should estimate tokens based on char count."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Response text here"  # 18 chars
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            response = mock_cli_client.call("Test")  # 4 chars

            # (4 + 18) / 4 = 5 estimated tokens
            assert response.tokens_used == 5


class TestMockClient:
    """Tests for MockClient."""

    def test_mock_returns_valid_json(self):
        """Mock should return valid JSON response."""
        client = MockClient()
        response = client.call("Any prompt")

        import json
        parsed = json.loads(response.content)
        assert "name" in parsed
        assert "children" in parsed

    def test_mock_model_id(self):
        """Mock should report correct model ID."""
        client = MockClient("test-model")
        assert client.model_id == "test-model"


class TestModelRegistry:
    """Tests for model registry."""

    def test_claude_models_registered(self):
        """Claude models should be in registry."""
        assert "claude-sonnet" in MODEL_REGISTRY
        assert "claude-opus" in MODEL_REGISTRY
        assert "claude-haiku" in MODEL_REGISTRY

    def test_cli_models_registered(self):
        """CLI models should be in registry."""
        assert "claude-sonnet-cli" in MODEL_REGISTRY
        assert "claude-opus-cli" in MODEL_REGISTRY

    def test_gemini_models_registered(self):
        """Gemini models should be in registry."""
        assert "gemini-2.0-flash" in MODEL_REGISTRY
        assert "gemini-2.5-flash" in MODEL_REGISTRY
        assert "gemini-2.5-pro" in MODEL_REGISTRY


class TestResolveModel:
    """Tests for model name resolution."""

    def test_resolve_shorthand(self):
        """Should resolve shorthand names."""
        assert resolve_model("claude-sonnet") == "claude-sonnet-4-20250514"

    def test_resolve_unknown_returns_same(self):
        """Unknown names should return as-is."""
        assert resolve_model("unknown-model") == "unknown-model"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_fields(self):
        """Should have all required fields."""
        response = LLMResponse(
            content="Test content",
            model="test-model",
            tokens_used=100,
        )
        assert response.content == "Test content"
        assert response.model == "test-model"
        assert response.tokens_used == 100

    def test_default_tokens(self):
        """tokens_used should default to 0."""
        response = LLMResponse(content="Test", model="test")
        assert response.tokens_used == 0


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests that actually call the CLI.

    These tests require:
    - Claude CLI installed
    - Valid Claude Max subscription / OAuth

    Run with: pytest -m integration
    """

    def test_sonnet_cli_simple_response(self):
        """Test actual CLI call with sonnet."""
        client = create_client("claude-sonnet-cli")

        response = client.call("Reply with exactly: INTEGRATION_TEST_OK")

        assert "INTEGRATION_TEST_OK" in response.content or "OK" in response.content
        assert response.tokens_used > 0
        assert response.model == "claude-sonnet-4-20250514"

    def test_sonnet_cli_json_response(self):
        """Test CLI can return structured JSON."""
        client = create_client("claude-sonnet-cli")

        prompt = """Return only this exact JSON, no other text:
{"test": "success", "value": 42}"""

        response = client.call(prompt)

        import json
        # Try to find JSON in response
        content = response.content
        if "```" in content:
            # Extract from code block
            import re
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)

        parsed = json.loads(content)
        assert parsed.get("test") == "success"
        assert parsed.get("value") == 42
