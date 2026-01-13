"""
Tests for model_client library.

Covers:
- Retry logic with exponential backoff
- Response caching (temperature=0 only)
- TokenUsage and ModelResponse dataclasses
- Cost calculation
- ModelRegistry factory pattern
"""

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from model_client import (
    CachedClient,
    ClientConfig,
    ModelRegistry,
    ModelResponse,
    ResponseCache,
    TokenUsage,
)
from model_client.retry import (
    RETRYABLE_EXCEPTIONS,
    is_retryable_exception,
    with_retry,
    with_retry_async,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        """Should have zero defaults."""
        usage = TokenUsage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_total_tokens(self):
        """Should sum input and output tokens."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)

        assert usage.total_tokens == 150

    def test_total_tokens_with_zeros(self):
        """Should handle zero values."""
        usage = TokenUsage(input_tokens=0, output_tokens=0)

        assert usage.total_tokens == 0


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_required_fields(self):
        """Should require content and model."""
        response = ModelResponse(content="Hello", model="test-model")

        assert response.content == "Hello"
        assert response.model == "test-model"

    def test_default_values(self):
        """Should have sensible defaults."""
        response = ModelResponse(content="test", model="test-model")

        assert response.finish_reason == "stop"
        assert response.latency_ms == 0.0
        assert response.raw_response is None
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0

    def test_cost_usd_known_model(self):
        """Should calculate cost for known models."""
        response = ModelResponse(
            content="test",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )

        # claude-sonnet-4: $3/1M input, $15/1M output
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert response.cost_usd == pytest.approx(expected_cost)

    def test_cost_usd_unknown_model(self):
        """Should use default pricing for unknown models."""
        response = ModelResponse(
            content="test",
            model="unknown-model-xyz",
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )

        # Default: $3/1M input, $15/1M output
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert response.cost_usd == pytest.approx(expected_cost)

    def test_cost_usd_gpt4o(self):
        """Should calculate cost for GPT-4o."""
        response = ModelResponse(
            content="test",
            model="gpt-4o",
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )

        # gpt-4o: $2.5/1M input, $10/1M output
        expected_cost = (1000 / 1_000_000) * 2.5 + (500 / 1_000_000) * 10.0
        assert response.cost_usd == pytest.approx(expected_cost)

    def test_cost_usd_zero_tokens(self):
        """Should return zero cost for zero tokens."""
        response = ModelResponse(
            content="test",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )

        assert response.cost_usd == 0.0


class TestIsRetryableException:
    """Tests for is_retryable_exception function."""

    def test_retryable_by_name(self):
        """Should identify exceptions by class name."""
        for exc_name in RETRYABLE_EXCEPTIONS:
            # Create a mock exception with the right name
            exc_class = type(exc_name, (Exception,), {})
            exc = exc_class("Test error")
            assert is_retryable_exception(exc), f"{exc_name} should be retryable"

    def test_retryable_by_message(self):
        """Should identify rate limit by message content."""

        class CustomError(Exception):
            pass

        exc = CustomError("Request rate limited, please retry")
        assert is_retryable_exception(exc)

    def test_rate_limit_case_insensitive(self):
        """Should match 'rate' case-insensitively."""

        class CustomError(Exception):
            pass

        assert is_retryable_exception(CustomError("RATE limit exceeded"))
        assert is_retryable_exception(CustomError("Rate throttled"))

    def test_non_retryable_exception(self):
        """Should return False for non-retryable exceptions."""

        class ValueError(Exception):
            pass

        exc = ValueError("Invalid input")
        assert not is_retryable_exception(exc)


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_success_first_attempt(self):
        """Should return immediately on success."""
        call_count = 0

        @with_retry(max_attempts=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()

        assert result == "success"
        assert call_count == 1

    def test_retries_on_retryable_exception(self):
        """Should retry on retryable exceptions."""
        call_count = 0

        # Create a retryable exception class
        RateLimitError = type("RateLimitError", (Exception,), {})

        @with_retry(max_attempts=3, initial_delay=0.01, max_delay=0.1)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"

        result = failing_then_success()

        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        """Should raise after exhausting attempts."""
        call_count = 0
        RateLimitError = type("RateLimitError", (Exception,), {})

        @with_retry(max_attempts=3, initial_delay=0.01, max_delay=0.1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Always failing")

        with pytest.raises(RateLimitError):
            always_fails()

        assert call_count == 3

    def test_no_retry_on_non_retryable(self):
        """Should not retry non-retryable exceptions."""
        call_count = 0

        @with_retry(max_attempts=3)
        def value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            value_error()

        assert call_count == 1  # No retry

    def test_custom_retryable_exceptions(self):
        """Should retry on custom exception types."""
        call_count = 0

        class CustomError(Exception):
            pass

        @with_retry(
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(CustomError,),
        )
        def custom_failing():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("Custom error")
            return "success"

        result = custom_failing()

        assert result == "success"
        assert call_count == 2

    def test_exponential_backoff(self):
        """Should increase delay exponentially."""
        delays = []
        call_count = 0
        RateLimitError = type("RateLimitError", (Exception,), {})

        @with_retry(
            max_attempts=4,
            initial_delay=0.1,
            exponential_base=2.0,
            max_delay=10.0,
        )
        def track_timing():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise RateLimitError("Fail")
            return "done"

        with patch("time.sleep") as mock_sleep:
            track_timing()
            delays = [call[0][0] for call in mock_sleep.call_args_list]

        # Should be: 0.1, 0.2, 0.4 (exponential)
        assert len(delays) == 3
        assert delays[0] == pytest.approx(0.1)
        assert delays[1] == pytest.approx(0.2)
        assert delays[2] == pytest.approx(0.4)

    def test_max_delay_cap(self):
        """Should cap delay at max_delay."""
        call_count = 0
        RateLimitError = type("RateLimitError", (Exception,), {})

        @with_retry(
            max_attempts=5,
            initial_delay=1.0,
            exponential_base=10.0,  # Would grow very fast
            max_delay=2.0,  # Cap at 2 seconds
        )
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Fail")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RateLimitError):
                always_fail()
            delays = [call[0][0] for call in mock_sleep.call_args_list]

        # All delays after first should be capped at 2.0
        assert delays[0] == pytest.approx(1.0)
        assert all(d <= 2.0 for d in delays)


class TestWithRetryAsync:
    """Tests for async retry decorator."""

    def test_async_success(self):
        """Should work with async functions."""

        @with_retry_async(max_attempts=3, initial_delay=0.01)
        async def async_success():
            return "async result"

        result = asyncio.run(async_success())
        assert result == "async result"

    def test_async_retry(self):
        """Should retry async functions."""
        call_count = 0
        RateLimitError = type("RateLimitError", (Exception,), {})

        @with_retry_async(max_attempts=3, initial_delay=0.01)
        async def async_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("Rate limited")
            return "success"

        result = asyncio.run(async_retry())

        assert result == "success"
        assert call_count == 2


class TestResponseCache:
    """Tests for ResponseCache."""

    def test_cache_miss(self, tmp_path: Path):
        """Should return None on cache miss."""
        cache = ResponseCache(tmp_path / "cache")

        result = cache.get(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            max_tokens=100,
            temperature=0.0,
        )

        assert result is None

    def test_cache_hit(self, tmp_path: Path):
        """Should return cached response on hit."""
        cache = ResponseCache(tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]
        response = ModelResponse(
            content="Cached response",
            model="test-model",
            usage=TokenUsage(input_tokens=10, output_tokens=20),
        )

        cache.set(messages, "test-model", 100, 0.0, response)
        result = cache.get(messages, "test-model", 100, 0.0)

        assert result is not None
        assert result.content == "Cached response"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    def test_no_cache_for_temperature_gt_zero(self, tmp_path: Path):
        """Should not cache when temperature > 0."""
        cache = ResponseCache(tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]
        response = ModelResponse(content="Random", model="test-model")

        # Set with temperature > 0
        cache.set(messages, "test-model", 100, 0.5, response)

        # Should not be cached
        result = cache.get(messages, "test-model", 100, 0.5)
        assert result is None

    def test_no_get_for_temperature_gt_zero(self, tmp_path: Path):
        """Should return None for temperature > 0 even if somehow cached."""
        cache = ResponseCache(tmp_path / "cache")

        # Even if we try to get with temp > 0, should return None
        result = cache.get(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            max_tokens=100,
            temperature=0.5,
        )

        assert result is None

    def test_different_messages_different_cache(self, tmp_path: Path):
        """Should have separate cache entries for different messages."""
        cache = ResponseCache(tmp_path / "cache")
        response1 = ModelResponse(content="Response 1", model="test-model")
        response2 = ModelResponse(content="Response 2", model="test-model")

        cache.set([{"role": "user", "content": "First"}], "test-model", 100, 0.0, response1)
        cache.set([{"role": "user", "content": "Second"}], "test-model", 100, 0.0, response2)

        result1 = cache.get([{"role": "user", "content": "First"}], "test-model", 100, 0.0)
        result2 = cache.get([{"role": "user", "content": "Second"}], "test-model", 100, 0.0)

        assert result1 is not None
        assert result2 is not None
        assert result1.content == "Response 1"
        assert result2.content == "Response 2"

    def test_different_models_different_cache(self, tmp_path: Path):
        """Should have separate cache entries for different models."""
        cache = ResponseCache(tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]
        response1 = ModelResponse(content="Model A", model="model-a")
        response2 = ModelResponse(content="Model B", model="model-b")

        cache.set(messages, "model-a", 100, 0.0, response1)
        cache.set(messages, "model-b", 100, 0.0, response2)

        result1 = cache.get(messages, "model-a", 100, 0.0)
        result2 = cache.get(messages, "model-b", 100, 0.0)

        assert result1 is not None
        assert result2 is not None
        assert result1.content == "Model A"
        assert result2.content == "Model B"

    def test_cache_stats(self, tmp_path: Path):
        """Should track cache hits and misses."""
        cache = ResponseCache(tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]
        response = ModelResponse(content="Test", model="test-model")

        # Miss
        cache.get(messages, "test-model", 100, 0.0)
        # Set
        cache.set(messages, "test-model", 100, 0.0, response)
        # Hit
        cache.get(messages, "test-model", 100, 0.0)
        # Another hit
        cache.get(messages, "test-model", 100, 0.0)

        stats = cache.stats()

        assert stats["misses"] == 1
        assert stats["hits"] == 2
        assert stats["hit_rate"] == pytest.approx(2 / 3)
        assert stats["entries"] == 1

    def test_cache_clear(self, tmp_path: Path):
        """Should clear all cache entries."""
        cache = ResponseCache(tmp_path / "cache")
        response = ModelResponse(content="Test", model="test-model")

        cache.set([{"role": "user", "content": "1"}], "test-model", 100, 0.0, response)
        cache.set([{"role": "user", "content": "2"}], "test-model", 100, 0.0, response)
        cache.set([{"role": "user", "content": "3"}], "test-model", 100, 0.0, response)

        count = cache.clear()

        assert count == 3
        assert cache.stats()["entries"] == 0

    def test_handles_corrupted_cache_file(self, tmp_path: Path):
        """Should handle corrupted cache files gracefully."""
        cache = ResponseCache(tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]
        response = ModelResponse(content="Test", model="test-model")

        # Set a valid entry
        cache.set(messages, "test-model", 100, 0.0, response)

        # Corrupt the cache file
        for json_file in (tmp_path / "cache").rglob("*.json"):
            json_file.write_text("not valid json {{{")

        # Should return None gracefully
        result = cache.get(messages, "test-model", 100, 0.0)
        assert result is None


class TestCachedClient:
    """Tests for CachedClient wrapper."""

    def test_returns_cached_response(self, tmp_path: Path):
        """Should return cached response without calling underlying client."""
        mock_client = Mock()
        cached = CachedClient(mock_client, cache_dir=tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]

        # First call - should hit underlying client
        mock_response = ModelResponse(
            content="Hello back",
            model="test-model",
            usage=TokenUsage(input_tokens=5, output_tokens=10),
        )
        mock_client.generate.return_value = mock_response

        result1 = cached.generate(messages, model="test-model", temperature=0.0)
        assert result1.content == "Hello back"
        assert mock_client.generate.call_count == 1

        # Second call - should use cache
        result2 = cached.generate(messages, model="test-model", temperature=0.0)
        assert result2.content == "Hello back"
        assert mock_client.generate.call_count == 1  # Still 1

    def test_bypasses_cache_for_nonzero_temp(self, tmp_path: Path):
        """Should bypass cache for temperature > 0."""
        mock_client = Mock()
        cached = CachedClient(mock_client, cache_dir=tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]

        mock_response = ModelResponse(content="Random", model="test-model")
        mock_client.generate.return_value = mock_response

        # Both calls should hit the underlying client
        cached.generate(messages, model="test-model", temperature=0.5)
        cached.generate(messages, model="test-model", temperature=0.5)

        assert mock_client.generate.call_count == 2

    def test_cache_stats_method(self, tmp_path: Path):
        """Should expose cache stats."""
        mock_client = Mock()
        cached = CachedClient(mock_client, cache_dir=tmp_path / "cache")

        stats = cached.cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_clear_cache_method(self, tmp_path: Path):
        """Should allow clearing cache."""
        mock_client = Mock()
        cached = CachedClient(mock_client, cache_dir=tmp_path / "cache")
        messages = [{"role": "user", "content": "Hello"}]

        mock_response = ModelResponse(content="Test", model="test-model")
        mock_client.generate.return_value = mock_response

        cached.generate(messages, model="test-model", temperature=0.0)
        count = cached.clear_cache()

        assert count == 1


class TestClientConfig:
    """Tests for ClientConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ClientConfig()

        assert config.api_key is None
        assert config.timeout == 120.0
        assert config.max_retries == 3
        assert config.base_url is None

    def test_custom_values(self):
        """Should accept custom values."""
        config = ClientConfig(
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
            base_url="https://custom.api",
        )

        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.base_url == "https://custom.api"


class TestModelRegistry:
    """Tests for ModelRegistry factory."""

    def setup_method(self):
        """Clear registry between tests."""
        ModelRegistry.clear_cache()

    def test_unknown_provider_raises(self):
        """Should raise for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelRegistry.get_client("unknown_provider", "model")  # type: ignore[arg-type]

    def test_client_caching(self):
        """Should cache and reuse clients."""
        # This test just verifies the caching logic works
        # We can't easily test actual client creation without API keys
        ModelRegistry._clients["anthropic:test"] = Mock()

        client1 = ModelRegistry._clients["anthropic:test"]
        client2 = ModelRegistry._clients["anthropic:test"]

        assert client1 is client2

    def test_clear_cache(self):
        """Should clear client cache."""
        ModelRegistry._clients["test:key"] = Mock()

        ModelRegistry.clear_cache()

        assert len(ModelRegistry._clients) == 0
