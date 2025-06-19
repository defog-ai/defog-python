import unittest
import pytest
from unittest.mock import Mock, patch
from defog.local_metadata_extractor import (
    extract_metadata_from_db,
    extract_metadata_from_db_async,
)


class TestLocalMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.sample_db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass",
        }

        self.sample_schema_result = {
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

    @patch("defog.Defog")
    def test_extract_metadata_from_db_success(self, mock_defog_class):
        # Setup mock
        mock_defog = Mock()
        mock_defog.generate_db_schema.return_value = self.sample_schema_result
        mock_defog_class.return_value = mock_defog

        # Test
        result = extract_metadata_from_db(
            db_type="postgres",
            db_creds=self.sample_db_creds,
            tables=["users", "orders"],
            api_key="test_key",
        )

        # Verify
        self.assertEqual(result, self.sample_schema_result)
        mock_defog_class.assert_called_once_with(
            api_key="test_key", db_type="postgres", db_creds=self.sample_db_creds
        )
        mock_defog.generate_db_schema.assert_called_once_with(
            tables=["users", "orders"], scan=False, upload=False, return_format="json"
        )

    @patch("defog.Defog")
    def test_extract_metadata_from_db_no_tables(self, mock_defog_class):
        mock_defog = Mock()
        mock_defog.generate_db_schema.return_value = self.sample_schema_result
        mock_defog_class.return_value = mock_defog

        result = extract_metadata_from_db(
            db_type="mysql", db_creds=self.sample_db_creds
        )

        self.assertEqual(result, self.sample_schema_result)
        mock_defog.generate_db_schema.assert_called_once_with(
            tables=[], scan=False, upload=False, return_format="json"
        )

    @patch("defog.Defog")
    def test_extract_metadata_from_db_with_cache(self, mock_defog_class):
        mock_defog = Mock()
        mock_defog.generate_db_schema.return_value = self.sample_schema_result
        mock_defog_class.return_value = mock_defog

        mock_cache = Mock()

        result = extract_metadata_from_db(
            db_type="postgres",
            db_creds=self.sample_db_creds,
            cache=mock_cache,
            api_key="test_key",
        )

        self.assertEqual(result, self.sample_schema_result)
        mock_cache.set.assert_called_once_with(
            "test_key",
            "postgres",
            self.sample_schema_result,
            dev=False,
            db_creds=self.sample_db_creds,
        )

    @patch("defog.Defog")
    def test_extract_metadata_from_db_no_api_key(self, mock_defog_class):
        mock_defog = Mock()
        mock_defog.generate_db_schema.return_value = self.sample_schema_result
        mock_defog_class.return_value = mock_defog

        extract_metadata_from_db(db_type="postgres", db_creds=self.sample_db_creds)

        mock_defog_class.assert_called_once_with(
            api_key=None, db_type="postgres", db_creds=self.sample_db_creds
        )

    @patch("defog.Defog")
    def test_extract_metadata_from_db_error(self, mock_defog_class):
        mock_defog = Mock()
        mock_defog.generate_db_schema.side_effect = Exception(
            "Database connection failed"
        )
        mock_defog_class.return_value = mock_defog

        with self.assertRaises(RuntimeError) as context:
            extract_metadata_from_db(db_type="postgres", db_creds=self.sample_db_creds)

        self.assertIn(
            "Failed to extract metadata from postgres database", str(context.exception)
        )
        self.assertIn("Database connection failed", str(context.exception))

    @pytest.mark.asyncio
    @patch("defog.AsyncDefog")
    async def test_extract_metadata_from_db_async_success(self, mock_async_defog_class):
        # Setup mock
        mock_async_defog = Mock()
        mock_async_defog.generate_db_schema = Mock(
            return_value=self.sample_schema_result
        )
        mock_async_defog_class.return_value = mock_async_defog

        # Test
        result = await extract_metadata_from_db_async(
            db_type="postgres",
            db_creds=self.sample_db_creds,
            tables=["users"],
            api_key="test_key",
        )

        # Verify
        self.assertEqual(result, self.sample_schema_result)
        mock_async_defog_class.assert_called_once_with(
            api_key="test_key", db_type="postgres", db_creds=self.sample_db_creds
        )

    @pytest.mark.asyncio
    @patch("defog.AsyncDefog")
    async def test_extract_metadata_from_db_async_with_cache(
        self, mock_async_defog_class
    ):
        mock_async_defog = Mock()
        mock_async_defog.generate_db_schema = Mock(
            return_value=self.sample_schema_result
        )
        mock_async_defog_class.return_value = mock_async_defog

        mock_cache = Mock()

        result = await extract_metadata_from_db_async(
            db_type="bigquery",
            db_creds={"project_id": "test"},
            cache=mock_cache,
            api_key="test_key",
        )

        self.assertEqual(result, self.sample_schema_result)
        mock_cache.set.assert_called_once_with(
            "test_key",
            "bigquery",
            self.sample_schema_result,
            dev=False,
            db_creds={"project_id": "test"},
        )

    @pytest.mark.asyncio
    @patch("defog.AsyncDefog")
    async def test_extract_metadata_from_db_async_error(self, mock_async_defog_class):
        mock_async_defog = Mock()
        mock_async_defog.generate_db_schema = Mock(side_effect=Exception("Async error"))
        mock_async_defog_class.return_value = mock_async_defog

        with self.assertRaises(RuntimeError) as context:
            await extract_metadata_from_db_async(
                db_type="mysql", db_creds=self.sample_db_creds
            )

        self.assertIn(
            "Failed to extract metadata from mysql database", str(context.exception)
        )
        self.assertIn("Async error", str(context.exception))

    @pytest.mark.asyncio
    @patch("defog.AsyncDefog")
    async def test_extract_metadata_from_db_async_no_api_key(
        self, mock_async_defog_class
    ):
        mock_async_defog = Mock()
        mock_async_defog.generate_db_schema = Mock(
            return_value=self.sample_schema_result
        )
        mock_async_defog_class.return_value = mock_async_defog

        await extract_metadata_from_db_async(
            db_type="postgres", db_creds=self.sample_db_creds
        )

        mock_async_defog_class.assert_called_once_with(
            api_key=None, db_type="postgres", db_creds=self.sample_db_creds
        )


if __name__ == "__main__":
    unittest.main()
