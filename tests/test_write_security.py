"""Security tests to verify SQL injection fixes."""

import pytest
from unittest.mock import Mock
from defog.schema_documenter import SchemaDocumenter, _validate_sql_identifier


class TestSQLInjectionPrevention:
    """Test SQL injection prevention measures."""

    def test_validate_sql_identifier_valid_names(self):
        """Test that valid identifiers pass validation."""
        valid_identifiers = [
            "users",
            "user_table",
            "schema1.table1",
            "Table123",
            "schema.Table_Name_123",
            "customer_updates",
            "dropshipments",
            "drop_update",
            '"My Table"',
            "my_table",
            "my_drop_table",
            "my_table_123",
            "my_drop_table_123",
            "my_table_123",
            '"my drop table"',
        ]

        for identifier in valid_identifiers:
            result = _validate_sql_identifier(identifier)
            assert result == identifier

    def test_validate_sql_identifier_invalid_names(self):
        """Test that invalid identifiers are rejected."""
        invalid_identifiers = [
            "users; DROP TABLE important_data; --",
            "users' OR '1'='1",
            'users" UNION SELECT * FROM passwords --',
            "users--comment",
            "users/*comment*/",
            "users + 1",
            "users * 2",
            "users / 1",
            "users \\ path",
            "users with spaces",
            "users\ntab",
            "users\ttab",
            "SELECT * FROM users",
            "DROP TABLE users",
            "INSERT INTO users",
            "UPDATE users SET",
            "DELETE FROM users",
            "ALTER TABLE users",
            "CREATE TABLE evil",
            "EXEC sp_evil",
            "UNION SELECT",
        ]

        for identifier in invalid_identifiers:
            with pytest.raises(
                ValueError, match="Invalid SQL identifier|contains invalid pattern"
            ):
                _validate_sql_identifier(identifier)

    def test_validate_sql_identifier_length_limit(self):
        """Test that overly long identifiers are rejected."""
        # Test identifier at the limit (255 bytes)
        max_length_identifier = "a" * 255
        result = _validate_sql_identifier(max_length_identifier)
        assert result == max_length_identifier

        # Test identifier over the limit
        too_long_identifier = "a" * 256
        with pytest.raises(ValueError, match="SQL identifier too long"):
            _validate_sql_identifier(too_long_identifier)

        # Test quoted identifier over the limit
        too_long_quoted = '"' + "a" * 254 + '"'
        with pytest.raises(ValueError, match="SQL identifier too long"):
            _validate_sql_identifier(too_long_quoted)

        # Test UTF-8 multi-byte characters
        # Each emoji is 4 bytes, so 64 emojis = 256 bytes
        emoji_identifier = "🎉" * 64
        with pytest.raises(ValueError, match="SQL identifier too long"):
            _validate_sql_identifier(emoji_identifier)

    def test_validate_sql_identifier_zero_width_chars(self):
        """Test that zero-width and invisible characters are rejected."""
        # Zero-width space
        with pytest.raises(ValueError, match="invisible/zero-width characters"):
            _validate_sql_identifier("users\u200btable")

        # Zero-width non-joiner
        with pytest.raises(ValueError, match="invisible/zero-width characters"):
            _validate_sql_identifier("users\u200ctable")

        # Zero-width joiner
        with pytest.raises(ValueError, match="invisible/zero-width characters"):
            _validate_sql_identifier("users\u200dtable")

        # Zero-width no-break space (BOM)
        with pytest.raises(ValueError, match="invisible/zero-width characters"):
            _validate_sql_identifier("\ufeffusers")

        # Word joiner
        with pytest.raises(ValueError, match="invisible/zero-width characters"):
            _validate_sql_identifier("users\u2060table")

        # Also test in quoted identifiers
        with pytest.raises(ValueError, match="invisible/zero-width characters"):
            _validate_sql_identifier('"users\u200btable"')

    def test_postgres_get_sample_data_injection_prevention(self):
        """Test that _get_sample_data prevents SQL injection for Postgres."""
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})

        # Mock cursor
        mock_cursor = Mock()

        # Test with malicious table name
        malicious_table = "users; DROP TABLE important_data; --"
        columns_info = [{"name": "id"}, {"name": "email"}]

        result = documenter._get_sample_data(mock_cursor, malicious_table, columns_info)

        # Should return empty data due to validation error
        assert result == {"id": [], "email": []}

        # Cursor execute should not have been called with malicious SQL
        mock_cursor.execute.assert_not_called()

    def test_duckdb_get_sample_data_injection_prevention(self):
        """Test that _get_sample_data prevents SQL injection for DuckDB."""
        documenter = SchemaDocumenter("duckdb", {"database": ":memory:"})

        # Mock cursor
        mock_cursor = Mock()

        # Test with malicious table name
        malicious_table = "users; DROP TABLE important_data; --"
        columns_info = [{"name": "id"}, {"name": "email"}]

        result = documenter._get_sample_data(mock_cursor, malicious_table, columns_info)

        # Should return empty data due to validation error
        assert result == {"id": [], "email": []}

        # Cursor execute should not have been called with malicious SQL
        mock_cursor.execute.assert_not_called()

    def test_duckdb_get_columns_injection_prevention(self):
        """Test that _get_duckdb_columns prevents SQL injection."""
        documenter = SchemaDocumenter("duckdb", {"database": ":memory:"})

        # Mock cursor
        mock_cursor = Mock()

        # Test with malicious table name
        malicious_table = "users; DROP TABLE important_data; --"

        result = documenter._get_duckdb_columns(mock_cursor, malicious_table)

        # Should return empty list due to validation error
        assert result == []

        # Cursor execute should not have been called with malicious SQL
        mock_cursor.execute.assert_not_called()

    def test_get_row_count_injection_prevention(self):
        """Test that _get_row_count prevents SQL injection."""
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})

        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)

        # Test with malicious table name
        malicious_table = "users; DROP TABLE important_data; --"

        result = documenter._get_row_count(mock_conn, malicious_table)

        # Should return 0 due to validation error
        assert result == 0

    def test_apply_table_comment_injection_prevention(self):
        """Test that _apply_table_comment prevents SQL injection."""
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})

        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)

        # Test with malicious table name
        malicious_table = "users; DROP TABLE important_data; --"
        description = "Safe description"

        result = documenter._apply_table_comment(
            mock_conn, malicious_table, description
        )

        # Should return False due to validation error
        assert result is False

        # Cursor execute should not have been called with malicious SQL
        mock_cursor.execute.assert_not_called()

    def test_apply_column_comment_injection_prevention(self):
        """Test that _apply_column_comment prevents SQL injection."""
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})

        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)

        # Test with malicious table and column names
        malicious_table = "users; DROP TABLE important_data; --"
        malicious_column = "id; DROP TABLE passwords; --"
        description = "Safe description"

        result = documenter._apply_column_comment(
            mock_conn, malicious_table, malicious_column, description
        )

        # Should return False due to validation error
        assert result is False

        # Cursor execute should not have been called with malicious SQL
        mock_cursor.execute.assert_not_called()

    def test_safe_identifiers_work_correctly(self):
        """Test that safe identifiers work correctly after validation."""
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})

        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)

        # Test with safe table name
        safe_table = "users"

        result = documenter._get_row_count(mock_conn, safe_table)

        # Should work correctly
        assert result == 42

        # Cursor execute should have been called with properly quoted identifier
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0][0]
        assert 'SELECT COUNT(*) FROM "users"' == call_args

    def test_schema_table_format_validation(self):
        """Test that schema.table format is handled safely."""
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})

        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (100,)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)

        # Test with safe schema.table format
        safe_table = "public.users"

        result = documenter._get_row_count(mock_conn, safe_table)

        # Should work correctly
        assert result == 100

        # Cursor execute should have been called with properly quoted identifiers
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0][0]
        assert 'SELECT COUNT(*) FROM "public"."users"' == call_args


class TestParameterizedQueries:
    """Test that parameterized queries are used where possible."""

    def test_sample_data_uses_parameterized_limit(self):
        """Test that sample data query uses parameterized limit."""
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})

        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [(1, "user1@example.com")]

        columns_info = [{"name": "id"}, {"name": "email"}]

        documenter._get_sample_data(mock_cursor, "users", columns_info)

        # Should have called execute with parameterized query
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]

        # Query should contain parameter placeholder
        assert "LIMIT %s" in query
        # Parameters should contain the sample size
        assert params == (documenter.config.sample_size,)

    def test_duckdb_columns_uses_parameterized_query(self):
        """Test that DuckDB column query uses parameters."""
        documenter = SchemaDocumenter("duckdb", {"database": ":memory:"})

        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("id", "INTEGER", "NO", None)]

        documenter._get_duckdb_columns(mock_cursor, "users")

        # Should have called execute with parameterized query
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]

        # Query should contain parameter placeholders
        assert "table_name = ?" in query
        assert "table_schema = ?" in query
        # Parameters should contain the table name and schema
        assert params == ("users", "main")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
