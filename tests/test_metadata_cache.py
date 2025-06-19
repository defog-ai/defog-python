import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from defog.metadata_cache import MetadataCache, get_global_cache


class TestMetadataCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = MetadataCache(cache_dir=self.temp_dir, ttl=60)

        self.sample_metadata = {
            "users": [
                {
                    "column_name": "id",
                    "data_type": "integer",
                    "column_description": "User ID",
                },
                {
                    "column_name": "name",
                    "data_type": "varchar",
                    "column_description": "User name",
                },
            ],
            "orders": [
                {
                    "column_name": "id",
                    "data_type": "integer",
                    "column_description": "Order ID",
                },
                {
                    "column_name": "user_id",
                    "data_type": "integer",
                    "column_description": "FK to users",
                },
            ],
        }

    def test_init_default_cache_dir(self):
        cache = MetadataCache()
        expected_dir = Path.home() / ".defog" / "cache"
        self.assertEqual(cache.cache_dir, expected_dir)

    def test_init_custom_cache_dir(self):
        custom_dir = "/tmp/custom_cache"
        cache = MetadataCache(cache_dir=custom_dir)
        self.assertEqual(cache.cache_dir, Path(custom_dir))

    def test_get_cache_key(self):
        cache_key = self.cache._get_cache_key("test_api_key", "postgres")
        self.assertIsInstance(cache_key, str)
        self.assertIn("postgres", cache_key)
        self.assertIn("prod", cache_key)

        dev_key = self.cache._get_cache_key("test_api_key", "postgres", dev=True)
        self.assertIn("dev", dev_key)
        self.assertNotEqual(cache_key, dev_key)

    def test_get_cache_key_with_none_api_key(self):
        db_creds = {"host": "localhost", "user": "test", "password": "secret"}
        cache_key = self.cache._get_cache_key(None, "postgres", db_creds=db_creds)
        self.assertIsInstance(cache_key, str)
        self.assertIn("postgres", cache_key)
        self.assertIn("prod", cache_key)

        # Same creds should produce same key
        cache_key2 = self.cache._get_cache_key(None, "postgres", db_creds=db_creds)
        self.assertEqual(cache_key, cache_key2)

        # Different creds should produce different key
        different_creds = {"host": "different", "user": "test", "password": "secret"}
        cache_key3 = self.cache._get_cache_key(
            None, "postgres", db_creds=different_creds
        )
        self.assertNotEqual(cache_key, cache_key3)

    def test_get_cache_file(self):
        cache_key = "test_key"
        cache_file = self.cache._get_cache_file(cache_key)
        expected_path = Path(self.temp_dir) / f"{cache_key}.json"
        self.assertEqual(cache_file, expected_path)

    def test_set_and_get_memory_cache(self):
        api_key = "test_key"
        db_type = "postgres"

        # Set metadata
        self.cache.set(api_key, db_type, self.sample_metadata)

        # Get metadata
        result = self.cache.get(api_key, db_type)
        self.assertEqual(result, self.sample_metadata)

    def test_set_and_get_file_cache(self):
        api_key = "test_key"
        db_type = "mysql"

        # Set metadata
        self.cache.set(api_key, db_type, self.sample_metadata)

        # Clear memory cache to force file read
        self.cache._memory_cache.clear()

        # Get metadata (should read from file)
        result = self.cache.get(api_key, db_type)
        self.assertEqual(result, self.sample_metadata)

    def test_set_and_get_with_none_api_key(self):
        db_creds = {"host": "localhost", "user": "test", "password": "secret"}
        db_type = "postgres"

        # Set metadata with None API key
        self.cache.set(None, db_type, self.sample_metadata, db_creds=db_creds)

        # Get metadata
        result = self.cache.get(None, db_type, db_creds=db_creds)
        self.assertEqual(result, self.sample_metadata)

        # Clear memory cache and test file cache
        self.cache._memory_cache.clear()
        result = self.cache.get(None, db_type, db_creds=db_creds)
        self.assertEqual(result, self.sample_metadata)

    def test_get_nonexistent_cache(self):
        result = self.cache.get("nonexistent_key", "postgres")
        self.assertIsNone(result)

    def test_cache_expiration_memory(self):
        api_key = "test_key"
        db_type = "postgres"

        # Create cache with very short TTL
        short_cache = MetadataCache(cache_dir=self.temp_dir, ttl=1)
        short_cache.set(api_key, db_type, self.sample_metadata)

        # Should get data immediately
        result = short_cache.get(api_key, db_type)
        self.assertEqual(result, self.sample_metadata)

        # Wait for expiration
        time.sleep(2)

        # Should return None after expiration
        result = short_cache.get(api_key, db_type)
        self.assertIsNone(result)

    def test_cache_expiration_file(self):
        api_key = "test_key"
        db_type = "postgres"

        # Create cache with very short TTL
        short_cache = MetadataCache(cache_dir=self.temp_dir, ttl=1)
        short_cache.set(api_key, db_type, self.sample_metadata)

        # Clear memory cache
        short_cache._memory_cache.clear()

        # Wait for expiration
        time.sleep(2)

        # Should return None and remove file
        result = short_cache.get(api_key, db_type)
        self.assertIsNone(result)

        # File should be removed
        cache_key = short_cache._get_cache_key(api_key, db_type)
        cache_file = short_cache._get_cache_file(cache_key)
        self.assertFalse(cache_file.exists())

    def test_invalidate(self):
        api_key = "test_key"
        db_type = "postgres"

        # Set metadata
        self.cache.set(api_key, db_type, self.sample_metadata)

        # Verify it's cached
        result = self.cache.get(api_key, db_type)
        self.assertEqual(result, self.sample_metadata)

        # Invalidate
        self.cache.invalidate(api_key, db_type)

        # Should return None after invalidation
        result = self.cache.get(api_key, db_type)
        self.assertIsNone(result)

    def test_clear_all(self):
        # Set multiple cache entries
        self.cache.set("key1", "postgres", self.sample_metadata)
        self.cache.set("key2", "mysql", self.sample_metadata)

        # Verify they're cached
        self.assertIsNotNone(self.cache.get("key1", "postgres"))
        self.assertIsNotNone(self.cache.get("key2", "mysql"))

        # Clear all
        self.cache.clear_all()

        # Should all return None
        self.assertIsNone(self.cache.get("key1", "postgres"))
        self.assertIsNone(self.cache.get("key2", "mysql"))

    def test_invalidate_with_none_api_key(self):
        db_creds = {"host": "localhost", "user": "test", "password": "secret"}
        db_type = "postgres"

        # Set metadata with None API key
        self.cache.set(None, db_type, self.sample_metadata, db_creds=db_creds)

        # Verify it's cached
        result = self.cache.get(None, db_type, db_creds=db_creds)
        self.assertEqual(result, self.sample_metadata)

        # Invalidate
        self.cache.invalidate(None, db_type, db_creds=db_creds)

        # Should return None after invalidation
        result = self.cache.get(None, db_type, db_creds=db_creds)
        self.assertIsNone(result)

    def test_dev_vs_prod_cache(self):
        api_key = "test_key"
        db_type = "postgres"

        prod_metadata = {"table1": [{"col": "prod"}]}
        dev_metadata = {"table1": [{"col": "dev"}]}

        # Set different metadata for prod and dev
        self.cache.set(api_key, db_type, prod_metadata, dev=False)
        self.cache.set(api_key, db_type, dev_metadata, dev=True)

        # Should get different results
        prod_result = self.cache.get(api_key, db_type, dev=False)
        dev_result = self.cache.get(api_key, db_type, dev=True)

        self.assertEqual(prod_result, prod_metadata)
        self.assertEqual(dev_result, dev_metadata)

    def test_corrupted_cache_file_handling(self):
        api_key = "test_key"
        db_type = "postgres"

        # Create a corrupted cache file
        cache_key = self.cache._get_cache_key(api_key, db_type)
        cache_file = self.cache._get_cache_file(cache_key)

        # Write invalid JSON
        with open(cache_file, "w") as f:
            f.write("invalid json content")

        # Should handle corruption gracefully and return None
        result = self.cache.get(api_key, db_type)
        self.assertIsNone(result)

        # Corrupted file should be removed
        self.assertFalse(cache_file.exists())

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_file_write_error_handling(self, mock_open):
        api_key = "test_key"
        db_type = "postgres"

        # Should not raise exception even if file write fails
        with patch("builtins.print") as mock_print:
            self.cache.set(api_key, db_type, self.sample_metadata)
            mock_print.assert_called()
            self.assertIn(
                "Warning: Failed to write metadata cache", str(mock_print.call_args)
            )

    def test_memory_cache_loaded_from_file(self):
        api_key = "test_key"
        db_type = "postgres"

        # Set metadata (creates file)
        self.cache.set(api_key, db_type, self.sample_metadata)

        # Clear memory cache
        self.cache._memory_cache.clear()

        # Get should load from file and populate memory cache
        result = self.cache.get(api_key, db_type)
        self.assertEqual(result, self.sample_metadata)

        # Memory cache should now be populated
        cache_key = self.cache._get_cache_key(api_key, db_type)
        self.assertIn(cache_key, self.cache._memory_cache)

    def test_get_global_cache(self):
        # Test singleton behavior
        cache1 = get_global_cache()
        cache2 = get_global_cache()

        self.assertIs(cache1, cache2)
        self.assertIsInstance(cache1, MetadataCache)

    def test_cached_data_structure(self):
        api_key = "test_key"
        db_type = "postgres"

        self.cache.set(api_key, db_type, self.sample_metadata, dev=True)

        # Check internal cache structure
        cache_key = self.cache._get_cache_key(api_key, db_type, dev=True)
        cached_data = self.cache._memory_cache[cache_key]

        self.assertEqual(cached_data["metadata"], self.sample_metadata)
        self.assertEqual(cached_data["db_type"], db_type)
        self.assertEqual(cached_data["dev"], True)
        self.assertIsInstance(cached_data["timestamp"], float)


if __name__ == "__main__":
    unittest.main()
