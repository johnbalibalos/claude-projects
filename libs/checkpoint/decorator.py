"""
Checkpoint decorator for caching individual function results.
"""

import json
import hashlib
import functools
from pathlib import Path
from typing import Any, Callable, Optional


def _make_key(*args, **kwargs) -> str:
    """Create a hash key from function arguments."""
    # Serialize args and kwargs to create unique key
    key_data = json.dumps({
        'args': [str(a) for a in args],
        'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
    }, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def checkpoint(
    cache_name: str,
    cache_dir: Optional[Path] = None,
    key_fn: Optional[Callable[..., str]] = None
):
    """
    Decorator to cache function results to disk.

    Args:
        cache_name: Name for this cache (used in filename)
        cache_dir: Directory for cache files (default: ./checkpoints)
        key_fn: Optional function to generate cache key from args

    Example:
        @checkpoint("api_calls")
        def call_api(prompt: str) -> str:
            return expensive_api_call(prompt)

        # First call - hits API
        result1 = call_api("hello")

        # Second call with same args - returns cached
        result2 = call_api("hello")
    """
    cache_path = Path(cache_dir or "./checkpoints")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{cache_name}_cache.json"

    # Load existing cache
    cache: dict[str, Any] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = _make_key(*args, **kwargs)

            # Check cache
            if key in cache:
                return cache[key]

            # Call function
            result = func(*args, **kwargs)

            # Save to cache
            cache[key] = result
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, default=str)

            return result

        # Add utility methods
        wrapper.cache = cache
        wrapper.clear_cache = lambda: cache.clear() or cache_file.unlink(missing_ok=True)

        return wrapper

    return decorator


def async_checkpoint(
    cache_name: str,
    cache_dir: Optional[Path] = None,
    key_fn: Optional[Callable[..., str]] = None
):
    """Async version of checkpoint decorator."""
    cache_path = Path(cache_dir or "./checkpoints")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{cache_name}_cache.json"

    cache: dict[str, Any] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = _make_key(*args, **kwargs)

            if key in cache:
                return cache[key]

            result = await func(*args, **kwargs)

            cache[key] = result
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, default=str)

            return result

        wrapper.cache = cache
        wrapper.clear_cache = lambda: cache.clear() or cache_file.unlink(missing_ok=True)

        return wrapper

    return decorator
