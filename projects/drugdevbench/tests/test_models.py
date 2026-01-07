"""Tests for model wrapper and caching."""

import pytest
import tempfile
from pathlib import Path

from drugdevbench.models import (
    EvaluatorConfig,
    SUPPORTED_MODELS,
    estimate_benchmark_cost,
)
from drugdevbench.models.cache import ResponseCache


class TestSupportedModels:
    """Test supported models configuration."""

    def test_claude_models_defined(self):
        """Claude models should be defined."""
        assert "claude-sonnet" in SUPPORTED_MODELS
        assert "claude-haiku" in SUPPORTED_MODELS

    def test_gemini_models_defined(self):
        """Gemini models should be defined."""
        assert "gemini-pro" in SUPPORTED_MODELS
        assert "gemini-flash" in SUPPORTED_MODELS

    def test_gpt_models_defined(self):
        """GPT models should be defined."""
        assert "gpt-4o" in SUPPORTED_MODELS
        assert "gpt-4o-mini" in SUPPORTED_MODELS

    def test_all_models_have_required_fields(self):
        """All models should have required configuration fields."""
        for model_name, config in SUPPORTED_MODELS.items():
            assert "litellm_key" in config
            assert "cost" in config
            assert "vision" in config
            assert config["vision"] is True  # All should support vision

    def test_cost_tiers(self):
        """Models should have appropriate cost tiers."""
        assert SUPPORTED_MODELS["claude-haiku"]["cost"].tier == "$"
        assert SUPPORTED_MODELS["claude-sonnet"]["cost"].tier == "$$"
        assert SUPPORTED_MODELS["gemini-flash"]["cost"].tier == "$"


class TestEvaluatorConfig:
    """Test evaluator configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = EvaluatorConfig()
        assert config.default_model == "claude-haiku"
        assert config.use_cache is True
        assert config.max_tokens == 1024
        assert config.temperature == 0.0

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = EvaluatorConfig(
            default_model="claude-sonnet",
            use_cache=False,
            max_tokens=2048,
        )
        assert config.default_model == "claude-sonnet"
        assert config.use_cache is False
        assert config.max_tokens == 2048


class TestResponseCache:
    """Test response caching."""

    def test_cache_set_and_get(self):
        """Should be able to set and get cached responses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))

            key = "test_key_123"
            response = {
                "response_text": "The IC50 is 2.5 nM",
                "prompt_tokens": 1500,
                "completion_tokens": 50,
            }

            cache.set(key, response)
            retrieved = cache.get(key)

            assert retrieved is not None
            assert retrieved["response_text"] == response["response_text"]
            assert retrieved["prompt_tokens"] == 1500

    def test_cache_miss(self):
        """Should return None for missing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))
            result = cache.get("nonexistent_key")
            assert result is None

    def test_cache_delete(self):
        """Should be able to delete cached entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))

            key = "delete_test_key"
            cache.set(key, {"test": "data"})
            assert cache.get(key) is not None

            deleted = cache.delete(key)
            assert deleted is True
            assert cache.get(key) is None

    def test_cache_delete_nonexistent(self):
        """Deleting nonexistent key should return False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))
            deleted = cache.delete("nonexistent")
            assert deleted is False

    def test_cache_clear(self):
        """Should be able to clear all cached entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))

            # Add some entries
            for i in range(5):
                cache.set(f"key_{i}", {"data": i})

            count = cache.clear()
            assert count == 5

            # Verify all cleared
            for i in range(5):
                assert cache.get(f"key_{i}") is None

    def test_cache_stats(self):
        """Should return cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))

            # Add some entries
            for i in range(3):
                cache.set(f"stats_key_{i}", {"data": "x" * 100})

            stats = cache.stats()
            assert stats["num_entries"] == 3
            assert stats["total_size_bytes"] > 0

    def test_cache_list_keys(self):
        """Should list cached keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))

            # Add entries
            cache.set("alpha", {"data": 1})
            cache.set("beta", {"data": 2})

            keys = cache.list_keys()
            assert len(keys) == 2
            assert "alpha" in keys
            assert "beta" in keys


class TestEstimateBenchmarkCost:
    """Test cost estimation."""

    def test_basic_estimate(self):
        """Should estimate costs for given parameters."""
        estimate = estimate_benchmark_cost(
            n_figures=10,
            n_questions_per_figure=4,
            n_conditions=5,
            models=["claude-haiku"],
        )

        assert "total_requests" in estimate
        assert "costs_by_model" in estimate
        assert "total_cost_usd" in estimate

        # 10 figures * 4 questions * 5 conditions * 1 model = 200 requests
        assert estimate["total_requests"] == 200

    def test_multiple_models(self):
        """Should estimate costs for multiple models."""
        estimate = estimate_benchmark_cost(
            n_figures=10,
            n_questions_per_figure=4,
            n_conditions=5,
            models=["claude-haiku", "gemini-flash"],
        )

        # 10 * 4 * 5 * 2 = 400 requests
        assert estimate["total_requests"] == 400

        # Both models should be in breakdown
        assert "claude-haiku" in estimate["costs_by_model"]
        assert "gemini-flash" in estimate["costs_by_model"]

    def test_cheap_models_are_cheaper(self):
        """Cheap models should have lower estimated costs."""
        cheap_estimate = estimate_benchmark_cost(
            n_figures=50,
            n_questions_per_figure=4,
            n_conditions=5,
            models=["claude-haiku"],
        )

        expensive_estimate = estimate_benchmark_cost(
            n_figures=50,
            n_questions_per_figure=4,
            n_conditions=5,
            models=["claude-sonnet"],
        )

        assert cheap_estimate["total_cost_usd"] < expensive_estimate["total_cost_usd"]

    def test_unknown_model_ignored(self):
        """Unknown models should be ignored in estimation."""
        estimate = estimate_benchmark_cost(
            n_figures=10,
            n_questions_per_figure=4,
            n_conditions=5,
            models=["unknown-model"],
        )

        assert estimate["total_cost_usd"] == 0.0
