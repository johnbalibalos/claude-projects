"""Response caching for DrugDevBench API calls."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResponseCache:
    """Simple file-based cache for API responses.

    Caches responses by a hash key (model + prompt + image hash).
    Each cached response is stored as a JSON file.
    """

    def __init__(self, cache_dir: Path | str = "data/cache/responses"):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cached responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            key: Cache key (hash string)

        Returns:
            Path to the cache file
        """
        # Use first 2 chars as subdirectory for better file distribution
        subdir = key[:2]
        (self.cache_dir / subdir).mkdir(exist_ok=True)
        return self.cache_dir / subdir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Get a cached response.

        Args:
            key: Cache key

        Returns:
            Cached response dictionary or None if not found
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, response: dict[str, Any]) -> None:
        """Store a response in the cache.

        Args:
            key: Cache key
            response: Response dictionary to cache
        """
        cache_path = self._get_cache_path(key)
        data = {
            "key": key,
            "cached_at": datetime.now().isoformat(),
            "response": response,
        }

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def delete(self, key: str) -> bool:
        """Delete a cached response.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.rglob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        cache_files = list(self.cache_dir.rglob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "num_entries": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def list_keys(self, limit: int = 100) -> list[str]:
        """List cached keys.

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of cache keys
        """
        keys = []
        for cache_file in self.cache_dir.rglob("*.json"):
            if len(keys) >= limit:
                break
            keys.append(cache_file.stem)
        return keys
