import unittest
import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from defog.llm.sql import sql_answer_tool, identify_relevant_tables_tool
from defog.llm.llm_providers import LLMProvider
from defog.llm.config import LLMConfig


class TestSQLAnswerTool(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        """Set up test fixtures for all tests."""
        self.sample_db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass"
        }
        
        self.sample_metadata_small = {
            "users": [
                {"column_name": "id", "data_type": "integer", "column_description": "User ID"},
                {"column_name": "name", "data_type": "varchar", "column_description": "User name"},
                {"column_name": "email", "data_type": "varchar", "column_description": "Email address"},
            ],
            "orders": [
                {"column_name": "id", "data_type": "integer", "column_description": "Order ID"},
                {"column_name": "user_id", "data_type": "integer", "column_description": "Foreign key to users"},
                {"column_name": "amount", "data_type": "decimal", "column_description": "Order amount"},
            ]
        }
        
        # Large database metadata for testing filtering
        self.sample_metadata_large = {}
        # Create 6 tables with 50 columns each (300 total columns)
        for table_num in range(6):
            table_name = f"table_{table_num}"
            columns = []
            for col_num in range(50):
                columns.append({
                    "column_name": f"col_{col_num}",
                    "data_type": "varchar",
                    "column_description": f"Column {col_num} in {table_name}"
                })
            self.sample_metadata_large[table_name] = columns
        
        # Add more tables to exceed the thresholds (>1000 columns AND >5 tables)
        for table_num in range(6, 12):
            table_name = f"extra_table_{table_num}"
            columns = []
            for col_num in range(130):  # More columns to exceed 1000 total
                columns.append({
                    "column_name": f"extra_col_{col_num}",
                    "data_type": "varchar",
                    "column_description": f"Extra column {col_num}"
                })
            self.sample_metadata_large[table_name] = columns
        
        self.sample_question = "How many orders were placed by each user?"
        
        self.mock_sql_result = {
            "query_generated": "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name",
            "ran_successfully": True,
            "error_message": None,
            "reason_for_query": "This query joins users and orders tables to count orders per user"
        }
        
        self.mock_query_results = {
            "ran_successfully": True,
            "error_message": None,
            "data": [
                {"name": "John Doe", "count": 5},
                {"name": "Jane Smith", "count": 3},
                {"name": "Bob Johnson", "count": 7}
            ]
        }

    def _validate_success_response(self, result):
        """Validate the structure of a successful response."""
        self.assertIsInstance(result, dict)
        self.assertEqual(result["success"], True)
        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["query"])
        self.assertIsNotNone(result["results"])
        self.assertIsInstance(result["results"], list)
        self.assertIn("query_reason", result)
        self.assertIn("row_count", result)
        self.assertIn("tables_used", result)
        self.assertIn("columns_analyzed", result)
        self.assertIn("filtered_tables", result)
        self.assertIsInstance(result["row_count"], int)
        self.assertIsInstance(result["tables_used"], int)
        self.assertIsInstance(result["columns_analyzed"], int)
        self.assertIsInstance(result["filtered_tables"], bool)

    def _validate_error_response(self, result):
        """Validate the structure of an error response."""
        self.assertIsInstance(result, dict)
        self.assertEqual(result["success"], False)
        self.assertIsNotNone(result["error"])
        self.assertIsInstance(result["error"], str)

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.generate_sql_query_local')
    @patch('defog.llm.sql.execute_query')
    async def test_sql_answer_tool_small_database_success(self, mock_execute, mock_generate, mock_extract):
        """Test successful SQL answer with small database (no filtering)."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata_small
        mock_generate.return_value = self.mock_sql_result
        mock_execute.return_value = self.mock_query_results
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_success_response(result)
        self.assertEqual(result["filtered_tables"], False)  # Small DB, no filtering
        self.assertEqual(result["tables_used"], 2)  # users and orders
        self.assertEqual(result["columns_analyzed"], 6)  # 3 + 3 columns
        self.assertIn("SELECT", result["query"].upper())
        self.assertEqual(len(result["results"]), 3)
        
        # Verify mocks were called correctly
        mock_extract.assert_called_once_with("postgres", self.sample_db_creds)
        mock_generate.assert_called_once()
        mock_execute.assert_called_once()

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.identify_relevant_tables_tool')
    @patch('defog.llm.sql.generate_sql_query_local')
    @patch('defog.llm.sql.execute_query')
    async def test_sql_answer_tool_large_database_with_filtering(self, mock_execute, mock_generate, mock_filter, mock_extract):
        """Test SQL answer with large database that triggers table filtering."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata_large
        
        # Mock filtering result
        filtered_metadata = {
            "users": self.sample_metadata_small["users"],
            "orders": self.sample_metadata_small["orders"]
        }
        mock_filter.return_value = {
            "success": True,
            "filtered_metadata": filtered_metadata,
            "relevant_tables": [
                {"table_name": "users", "relevance_score": 0.9, "reason": "Contains user data"},
                {"table_name": "orders", "relevance_score": 0.8, "reason": "Contains order data"}
            ]
        }
        
        mock_generate.return_value = self.mock_sql_result
        mock_execute.return_value = self.mock_query_results
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_success_response(result)
        self.assertEqual(result["filtered_tables"], True)  # Large DB, filtering applied
        self.assertEqual(result["tables_used"], 2)  # Filtered to users and orders
        
        # Verify filtering was called
        mock_filter.assert_called_once()
        
        # Verify generate_sql was called with filtered metadata
        generate_call_args = mock_generate.call_args
        self.assertEqual(generate_call_args[1]["table_metadata"], filtered_metadata)

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.identify_relevant_tables_tool')
    @patch('defog.llm.sql.generate_sql_query_local')
    @patch('defog.llm.sql.execute_query')
    async def test_sql_answer_tool_filtering_fails_fallback(self, mock_execute, mock_generate, mock_filter, mock_extract):
        """Test that when filtering fails, it falls back to using all tables."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata_large
        
        # Mock filtering failure
        mock_filter.return_value = {
            "success": False,
            "error": "Filtering failed for some reason",
            "filtered_metadata": {}
        }
        
        mock_generate.return_value = self.mock_sql_result
        mock_execute.return_value = self.mock_query_results
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_success_response(result)
        
        # Should still succeed using all tables
        self.assertEqual(result["filtered_tables"], True)  # Filtering was attempted
        
        # Verify generate_sql was called with original (large) metadata
        generate_call_args = mock_generate.call_args
        self.assertEqual(generate_call_args[1]["table_metadata"], self.sample_metadata_large)

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.generate_sql_query_local')
    async def test_sql_answer_tool_sql_generation_fails(self, mock_generate, mock_extract):
        """Test error handling when SQL generation fails."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata_small
        mock_generate.return_value = {
            "ran_successfully": False,
            "error_message": "Failed to generate SQL",
            "query_generated": None
        }
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_error_response(result)
        self.assertIn("SQL generation failed", result["error"])
        self.assertIsNone(result["query"])
        self.assertIsNone(result["results"])

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.generate_sql_query_local')
    @patch('defog.llm.sql.execute_query')
    async def test_sql_answer_tool_query_execution_fails(self, mock_execute, mock_generate, mock_extract):
        """Test error handling when query execution fails."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata_small
        mock_generate.return_value = self.mock_sql_result
        mock_execute.return_value = {
            "ran_successfully": False,
            "error_message": "Query execution failed: syntax error",
            "data": None
        }
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_error_response(result)
        self.assertIn("Query execution failed", result["error"])
        self.assertIsNotNone(result["query"])  # Query was generated
        self.assertIsNone(result["results"])

    @patch('defog.llm.sql.extract_metadata_from_db')
    async def test_sql_answer_tool_metadata_extraction_fails(self, mock_extract):
        """Test error handling when metadata extraction fails."""
        # Setup mock to raise exception
        mock_extract.side_effect = RuntimeError("Database connection failed")
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_error_response(result)
        self.assertIn("Database connection failed", result["error"])

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.generate_sql_query_local')
    @patch('defog.llm.sql.execute_query')
    async def test_sql_answer_tool_with_optional_params(self, mock_execute, mock_generate, mock_extract):
        """Test SQL answer tool with optional parameters."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata_small
        mock_generate.return_value = self.mock_sql_result
        mock_execute.return_value = self.mock_query_results
        
        glossary = "User: A person who uses the system"
        hard_filters = "WHERE created_at > '2023-01-01'"
        previous_context = [{"role": "user", "content": "Previous question"}]
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC,
            glossary=glossary,
            hard_filters=hard_filters,
            previous_context=previous_context,
            temperature=0.1
        )
        
        self._validate_success_response(result)
        
        # Verify generate_sql was called with optional parameters
        generate_call_args = mock_generate.call_args
        self.assertEqual(generate_call_args[1]["glossary"], glossary)
        self.assertEqual(generate_call_args[1]["hard_filters"], hard_filters)
        self.assertEqual(generate_call_args[1]["previous_context"], previous_context)
        self.assertEqual(generate_call_args[1]["temperature"], 0.1)

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.generate_sql_query_local')
    @patch('defog.llm.sql.execute_query')
    async def test_sql_answer_tool_empty_results(self, mock_execute, mock_generate, mock_extract):
        """Test handling of empty query results."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata_small
        mock_generate.return_value = self.mock_sql_result
        mock_execute.return_value = {
            "ran_successfully": True,
            "error_message": None,
            "data": []  # Empty results
        }
        
        result = await sql_answer_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_success_response(result)
        self.assertEqual(result["row_count"], 0)
        self.assertEqual(len(result["results"]), 0)

    def test_database_size_thresholds(self):
        """Test the logic for determining when to apply table filtering."""
        # Small database (< 1000 columns, < 5 tables) - no filtering
        small_columns = sum(len(columns) for columns in self.sample_metadata_small.values())
        small_tables = len(self.sample_metadata_small)
        should_filter_small = small_columns > 1000 and small_tables > 5
        self.assertFalse(should_filter_small)
        
        # Large database (> 1000 columns, > 5 tables) - should filter
        large_columns = sum(len(columns) for columns in self.sample_metadata_large.values())
        large_tables = len(self.sample_metadata_large)
        should_filter_large = large_columns > 1000 and large_tables > 5
        self.assertTrue(should_filter_large)


class TestIdentifyRelevantTablesTool(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        """Set up test fixtures for table relevance tests."""
        self.sample_db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass"
        }
        
        self.sample_metadata = {
            "users": [
                {"column_name": "id", "data_type": "integer", "column_description": "User ID"},
                {"column_name": "name", "data_type": "varchar", "column_description": "User name"},
            ],
            "orders": [
                {"column_name": "id", "data_type": "integer", "column_description": "Order ID"},
                {"column_name": "user_id", "data_type": "integer", "column_description": "Foreign key to users"},
            ],
            "products": [
                {"column_name": "id", "data_type": "integer", "column_description": "Product ID"},
                {"column_name": "name", "data_type": "varchar", "column_description": "Product name"},
            ],
            "irrelevant_table": [
                {"column_name": "id", "data_type": "integer", "column_description": "Some ID"},
                {"column_name": "data", "data_type": "text", "column_description": "Random data"},
            ]
        }
        
        self.sample_question = "How many orders were placed by each user?"

    def _validate_relevance_response(self, result):
        """Validate the structure of a table relevance response."""
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("relevant_tables", result)
        self.assertIn("filtered_metadata", result)
        self.assertIn("total_tables_analyzed", result)
        self.assertIn("tables_selected", result)

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.chat_async')
    async def test_identify_relevant_tables_success(self, mock_chat, mock_extract):
        """Test successful table relevance identification."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata
        
        # Mock LLM response with valid JSON
        mock_llm_response = Mock()
        mock_llm_response.content = '''
        {
            "relevant_tables": [
                {
                    "table_name": "users",
                    "relevance_score": 0.9,
                    "reason": "Contains user information needed for the query"
                },
                {
                    "table_name": "orders",
                    "relevance_score": 0.8,
                    "reason": "Contains order data to count orders per user"
                }
            ]
        }
        '''
        mock_chat.return_value = mock_llm_response
        
        result = await identify_relevant_tables_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC,
            max_tables=5
        )
        
        self._validate_relevance_response(result)
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["relevant_tables"]), 2)
        self.assertEqual(len(result["filtered_metadata"]), 2)
        self.assertEqual(result["total_tables_analyzed"], 4)
        self.assertEqual(result["tables_selected"], 2)
        
        # Check that the right tables were selected
        table_names = [table["table_name"] for table in result["relevant_tables"]]
        self.assertIn("users", table_names)
        self.assertIn("orders", table_names)
        
        # Verify filtered metadata contains the right tables
        self.assertIn("users", result["filtered_metadata"])
        self.assertIn("orders", result["filtered_metadata"])
        self.assertNotIn("irrelevant_table", result["filtered_metadata"])

    @patch('defog.llm.sql.extract_metadata_from_db')
    @patch('defog.llm.sql.chat_async')
    async def test_identify_relevant_tables_invalid_json(self, mock_chat, mock_extract):
        """Test handling of invalid JSON response from LLM."""
        # Setup mocks
        mock_extract.return_value = self.sample_metadata
        
        # Mock LLM response with invalid JSON
        mock_llm_response = Mock()
        mock_llm_response.content = "This is not valid JSON"
        mock_chat.return_value = mock_llm_response
        
        result = await identify_relevant_tables_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_relevance_response(result)
        self.assertFalse(result["success"])
        self.assertIn("Failed to parse LLM response as JSON", result["error"])
        self.assertEqual(len(result["relevant_tables"]), 0)
        self.assertEqual(len(result["filtered_metadata"]), 0)

    @patch('defog.llm.sql.extract_metadata_from_db')
    async def test_identify_relevant_tables_extraction_fails(self, mock_extract):
        """Test error handling when metadata extraction fails."""
        # Setup mock to raise exception
        mock_extract.side_effect = RuntimeError("Database connection failed")
        
        result = await identify_relevant_tables_tool(
            question=self.sample_question,
            db_type="postgres",
            db_creds=self.sample_db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC
        )
        
        self._validate_relevance_response(result)
        self.assertFalse(result["success"])
        self.assertIn("Database connection failed", result["error"])


if __name__ == "__main__":
    unittest.main()