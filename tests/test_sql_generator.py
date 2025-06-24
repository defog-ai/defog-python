import unittest
import pytest
from unittest.mock import Mock, patch
from defog.llm.sql_generator import (
    format_schema_for_prompt,
    build_sql_generation_prompt,
    generate_sql_query_local,
    generate_sql_query_local_sync,
)


class TestSQLGenerator(unittest.TestCase):
    def setUp(self):
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
                {
                    "column_name": "email",
                    "data_type": "varchar",
                    "column_description": "Email address",
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
                    "column_description": "Foreign key to users",
                },
                {
                    "column_name": "amount",
                    "data_type": "decimal",
                    "column_description": "Order amount",
                },
                {
                    "column_name": "created_at",
                    "data_type": "timestamp",
                    "column_description": "Order creation time",
                },
            ],
        }

    def test_format_schema_for_prompt(self):
        result = format_schema_for_prompt(self.sample_metadata)

        # Check for DDL CREATE TABLE statements
        self.assertIn("CREATE TABLE users (", result)
        self.assertIn("CREATE TABLE orders (", result)

        # Check for column definitions with inline comments
        self.assertIn("  id integer, -- User ID", result)
        self.assertIn("  name varchar, -- User name", result)
        self.assertIn(
            "  email varchar -- Email address", result
        )  # No comma on last column
        self.assertIn("  user_id integer, -- Foreign key to users", result)
        self.assertIn("  amount decimal, -- Order amount", result)
        self.assertIn("  created_at timestamp -- Order creation time", result)

        # Check for closing parenthesis
        self.assertIn(");", result)

    def test_format_schema_for_prompt_no_description(self):
        metadata = {
            "simple_table": [
                {"column_name": "col1", "data_type": "int"},
                {
                    "column_name": "col2",
                    "data_type": "varchar",
                    "column_description": "",
                },
            ]
        }
        result = format_schema_for_prompt(metadata)

        # Check DDL format without comments for columns with no description
        self.assertIn("CREATE TABLE simple_table (", result)
        self.assertIn("  col1 int,", result)
        self.assertIn("  col2 varchar", result)  # No comma on last column
        self.assertNotIn("-- ", result)  # No comments when no description

    def test_format_schema_for_prompt_with_table_description(self):
        # Test new format with table descriptions
        metadata = {
            "products": {
                "table_description": "Table storing product information",
                "columns": [
                    {
                        "column_name": "product_id",
                        "data_type": "INTEGER",
                        "column_description": "Unique ID for each product",
                    },
                    {
                        "column_name": "name",
                        "data_type": "VARCHAR(50)",
                        "column_description": "Name of the product",
                    },
                    {
                        "column_name": "price",
                        "data_type": "DECIMAL(10,2)",
                    },  # No description
                ],
            }
        }
        result = format_schema_for_prompt(metadata)

        # Check for table comment
        self.assertIn("CREATE TABLE products (", result)
        self.assertIn("  -- Table storing product information", result)

        # Check for column definitions with inline comments
        self.assertIn("  product_id INTEGER, -- Unique ID for each product", result)
        self.assertIn("  name VARCHAR(50), -- Name of the product", result)
        self.assertIn("  price DECIMAL(10,2)", result)  # No comment for this column

        # Check closing
        self.assertIn(");", result)

    def test_build_sql_generation_prompt_basic(self):
        question = "How many users are there?"
        messages = build_sql_generation_prompt(
            question=question, table_metadata=self.sample_metadata, db_type="postgres"
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], question)

        system_content = messages[0]["content"]
        self.assertIn("postgres", system_content)
        self.assertIn("CREATE TABLE users", system_content)
        self.assertIn("CREATE TABLE orders", system_content)

    def test_build_sql_generation_prompt_with_glossary(self):
        question = "How many active users?"
        glossary = "Active users: users who logged in within the last 30 days"

        messages = build_sql_generation_prompt(
            question=question,
            table_metadata=self.sample_metadata,
            db_type="mysql",
            glossary=glossary,
        )

        system_content = messages[0]["content"]
        self.assertIn("mysql", system_content)
        self.assertIn("Business Glossary:", system_content)
        self.assertIn(glossary, system_content)

    def test_build_sql_generation_prompt_with_hard_filters(self):
        question = "Show all orders"
        hard_filters = "WHERE deleted_at IS NULL"

        messages = build_sql_generation_prompt(
            question=question,
            table_metadata=self.sample_metadata,
            db_type="postgres",
            hard_filters=hard_filters,
        )

        system_content = messages[0]["content"]
        self.assertIn("Always apply these filters:", system_content)
        self.assertIn(hard_filters, system_content)

    def test_build_sql_generation_prompt_with_context(self):
        question = "What about orders from yesterday?"
        previous_context = [
            {"role": "user", "content": "Show me all users"},
            {"role": "assistant", "content": "SELECT * FROM users"},
        ]

        messages = build_sql_generation_prompt(
            question=question,
            table_metadata=self.sample_metadata,
            db_type="postgres",
            previous_context=previous_context,
        )

        self.assertEqual(len(messages), 4)  # system + 2 context + current question
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Show me all users")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "SELECT * FROM users")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], question)

    @pytest.mark.asyncio
    @patch("defog.llm.sql_generator.chat_async")
    async def test_generate_sql_query_local_success(self, mock_chat):
        # Mock the first chat call (SQL generation)
        mock_response = Mock()
        mock_response.content = "SELECT COUNT(*) FROM users"

        # Mock the second chat call (reason generation)
        mock_reason_response = Mock()
        mock_reason_response.content = (
            "This query counts the total number of users in the database."
        )

        mock_chat.side_effect = [mock_response, mock_reason_response]

        result = await generate_sql_query_local(
            question="How many users are there?",
            table_metadata=self.sample_metadata,
            db_type="postgres",
        )

        self.assertTrue(result["ran_successfully"])
        self.assertEqual(result["query_generated"], "SELECT COUNT(*) FROM users")
        self.assertEqual(result["query_db"], "postgres")
        self.assertIn("counts the total number", result["reason_for_query"])
        self.assertIsNone(result["error_message"])

        # Verify context was updated
        self.assertEqual(len(result["previous_context"]), 2)
        self.assertEqual(result["previous_context"][0]["role"], "user")
        self.assertEqual(result["previous_context"][1]["role"], "assistant")

    @pytest.mark.asyncio
    @patch("defog.llm.sql_generator.chat_async")
    async def test_generate_sql_query_local_with_markdown_cleanup(self, mock_chat):
        # Mock response with markdown formatting
        mock_response = Mock()
        mock_response.content = "```sql\nSELECT * FROM orders\n```"

        mock_reason_response = Mock()
        mock_reason_response.content = "Shows all orders"

        mock_chat.side_effect = [mock_response, mock_reason_response]

        result = await generate_sql_query_local(
            question="Show all orders",
            table_metadata=self.sample_metadata,
            db_type="postgres",
        )

        self.assertEqual(result["query_generated"], "SELECT * FROM orders")

    @pytest.mark.asyncio
    @patch("defog.llm.sql_generator.chat_async")
    async def test_generate_sql_query_local_with_existing_context(self, mock_chat):
        mock_response = Mock()
        mock_response.content = "SELECT * FROM users WHERE id = 1"

        mock_reason_response = Mock()
        mock_reason_response.content = "Gets user with ID 1"

        mock_chat.side_effect = [mock_response, mock_reason_response]

        existing_context = [
            {"role": "user", "content": "Show all users"},
            {"role": "assistant", "content": "SELECT * FROM users"},
        ]

        result = await generate_sql_query_local(
            question="Show user with ID 1",
            table_metadata=self.sample_metadata,
            db_type="postgres",
            previous_context=existing_context,
        )

        # Context should include existing + new
        self.assertEqual(len(result["previous_context"]), 4)

    @pytest.mark.asyncio
    @patch("defog.llm.sql_generator.chat_async")
    async def test_generate_sql_query_local_error_handling(self, mock_chat):
        mock_chat.side_effect = Exception("API Error")

        result = await generate_sql_query_local(
            question="How many users?",
            table_metadata=self.sample_metadata,
            db_type="postgres",
        )

        self.assertFalse(result["ran_successfully"])
        self.assertIsNone(result["query_generated"])
        self.assertEqual(result["error_message"], "API Error")
        self.assertEqual(result["query_db"], "postgres")

    @patch("asyncio.run")
    def test_generate_sql_query_local_sync(self, mock_run):
        mock_result = {
            "query_generated": "SELECT * FROM users",
            "ran_successfully": True,
        }
        mock_run.return_value = mock_result

        result = generate_sql_query_local_sync(
            question="Show users",
            table_metadata=self.sample_metadata,
            db_type="postgres",
        )

        self.assertEqual(result, mock_result)
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
