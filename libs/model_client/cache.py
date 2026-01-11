"""
Optional response caching for LLM API calls.

Use --use-cache flag to enable. Only caches deterministic calls (temperature=0).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .protocols import ModelResponse, TokenUsage

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = ".llm_cache"


class ResponseCache:
    """
    Disk-based cache for LLM responses.

    Only caches deterministic calls (temperature=0).
    Uses content-addressed storage with SHA256 hashes.
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Generate cache key from request parameters."""
        # Include all parameters that affect output
        key_data = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Include other kwargs that might affect output
            "system": kwargs.get("system"),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        # Use 2-level directory structure to avoid too many files in one dir
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> ModelResponse | None:
        """
        Get cached response if available.

        Returns None if not cached or temperature > 0.
        """
        # Never cache non-deterministic calls
        if temperature > 0:
            return None

        key = self._make_key(messages, model, max_tokens, temperature, **kwargs)
        path = self._cache_path(key)

        if not path.exists():
            self._misses += 1
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            self._hits += 1
            logger.info(f"Cache HIT: {key[:12]}... (model={model})")

            return ModelResponse(
                content=data["content"],
                model=data["model"],
                usage=TokenUsage(
                    input_tokens=data["usage"]["input_tokens"],
                    output_tokens=data["usage"]["output_tokens"],
                ),
                finish_reason=data.get("finish_reason", "stop"),
                latency_ms=0.0,  # Cached, no latency
                raw_response=None,  # Don't cache raw response
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache read error for {key[:12]}...: {e}")
            self._misses += 1
            return None

    def set(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        response: ModelResponse,
        **kwargs: Any,
    ) -> None:
        """
        Cache a response.

        Only caches if temperature=0.
        """
        if temperature > 0:
            return

        key = self._make_key(messages, model, max_tokens, temperature, **kwargs)
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "content": response.content,
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "finish_reason": response.finish_reason,
        }

        with open(path, "w") as f:
            json.dump(data, f)

        logger.debug(f"Cache SET: {key[:12]}... (model={model})")

    def clear(self) -> int:
        """Clear all cached responses. Returns number of entries cleared."""
        import shutil
        count = sum(1 for _ in self.cache_dir.rglob("*.json"))
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
            "entries": sum(1 for _ in self.cache_dir.rglob("*.json")) if self.cache_dir.exists() else 0,
        }


class CachedClient:
    """
    Wrapper that adds caching to any model client.

    Usage:
        client = AnthropicClient(config)
        cached = CachedClient(client, cache_dir=".llm_cache")
        response = cached.generate(messages, model="claude-sonnet-4-20250514")
    """

    def __init__(self, client: Any, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.client = client
        self.cache = ResponseCache(cache_dir)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate response, using cache if available."""
        # Check cache first
        cached = self.cache.get(messages, model, max_tokens, temperature, **kwargs)
        if cached is not None:
            return cached

        # Call underlying client
        response = self.client.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Cache the response
        self.cache.set(messages, model, max_tokens, temperature, response, **kwargs)

        return response

    def cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return self.cache.stats()

    def clear_cache(self) -> int:
        """Clear the cache. Returns number of entries cleared."""
        return self.cache.clear()


# Global cache instance for convenience
_global_cache: ResponseCache | None = None


def get_global_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> ResponseCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache(cache_dir)
    return _global_cache
