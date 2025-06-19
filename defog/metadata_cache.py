"""
Metadata caching for database schemas to avoid repeated queries.
"""

import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib


class MetadataCache:
    """
    Cache database metadata to avoid repeated schema queries.

    Supports both in-memory and file-based caching.
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 3600):
        """
        Initialize metadata cache.

        Args:
            cache_dir: Directory for file-based cache. If None, uses ~/.defog/cache
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.ttl = ttl
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

        # Set up file cache directory
        if cache_dir is None:
            self.cache_dir = Path.home() / ".defog" / "cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        api_key: Optional[str],
        db_type: str,
        dev: bool = False,
        db_creds: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a unique cache key."""
        if api_key is not None:
            # Hash the API key for security
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        else:
            # When no API key, use db_creds hash for cache key
            if db_creds:
                creds_str = json.dumps(db_creds, sort_keys=True)
                key_hash = hashlib.sha256(creds_str.encode()).hexdigest()[:16]
            else:
                # Fallback to a generic key if no creds available
                key_hash = "no_api_key"

        return f"{key_hash}_{db_type}_{'dev' if dev else 'prod'}"

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        api_key: Optional[str],
        db_type: str,
        dev: bool = False,
        db_creds: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """
        Get cached metadata if available and not expired.

        Args:
            api_key: Defog API key (can be None)
            db_type: Database type
            dev: Whether this is for dev environment
            db_creds: Database credentials (used for cache key when api_key is None)

        Returns:
            Cached metadata or None if not found/expired
        """
        cache_key = self._get_cache_key(api_key, db_type, dev, db_creds)

        # Check memory cache first
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if time.time() - cached["timestamp"] < self.ttl:
                return cached["metadata"]
            else:
                # Expired, remove from memory
                del self._memory_cache[cache_key]

        # Check file cache
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)

                if time.time() - cached["timestamp"] < self.ttl:
                    # Load into memory cache for faster access
                    self._memory_cache[cache_key] = cached
                    return cached["metadata"]
                else:
                    # Expired, remove file
                    cache_file.unlink()
            except Exception:
                # Corrupted cache file, remove it
                cache_file.unlink()

        return None

    def set(
        self,
        api_key: Optional[str],
        db_type: str,
        metadata: Dict[str, List[Dict[str, str]]],
        dev: bool = False,
        db_creds: Optional[Dict[str, Any]] = None,
    ):
        """
        Cache metadata.

        Args:
            api_key: Defog API key (can be None)
            db_type: Database type
            metadata: Table metadata to cache
            dev: Whether this is for dev environment
            db_creds: Database credentials (used for cache key when api_key is None)
        """
        cache_key = self._get_cache_key(api_key, db_type, dev, db_creds)

        cached_data = {
            "metadata": metadata,
            "timestamp": time.time(),
            "db_type": db_type,
            "dev": dev,
        }

        # Store in memory
        self._memory_cache[cache_key] = cached_data

        # Store in file
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, "w") as f:
                json.dump(cached_data, f, indent=2)
        except Exception as e:
            # Log error but don't fail
            print(f"Warning: Failed to write metadata cache: {e}")

    def invalidate(
        self,
        api_key: Optional[str],
        db_type: str,
        dev: bool = False,
        db_creds: Optional[Dict[str, Any]] = None,
    ):
        """
        Invalidate cached metadata.

        Args:
            api_key: Defog API key (can be None)
            db_type: Database type
            dev: Whether this is for dev environment
            db_creds: Database credentials (used for cache key when api_key is None)
        """
        cache_key = self._get_cache_key(api_key, db_type, dev, db_creds)

        # Remove from memory cache
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

        # Remove file
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            cache_file.unlink()

    def clear_all(self):
        """Clear all cached metadata."""
        # Clear memory cache
        self._memory_cache.clear()

        # Clear all cache files
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


# Global cache instance
_global_cache = None


def get_global_cache() -> MetadataCache:
    """Get or create the global metadata cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MetadataCache()
    return _global_cache
