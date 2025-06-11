"""
Tests for the schema documentation functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, AsyncMock, MagicMock
import json

from defog.schema_documenter import (
    SchemaDocumenter,
    DocumentationConfig,
    TableDocumentation,
    ColumnAnalysis,
    document_and_apply_schema,
)


class TestDocumentationConfig:
    """Test DocumentationConfig dataclass."""
    
    def test_default_config(self):
        config = DocumentationConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.1
        assert config.sample_size == 100
        assert config.max_categorical_values == 20
        assert config.include_data_patterns is True
        assert config.min_confidence_score == 0.7


class TestSchemaDocumenter:
    """Test SchemaDocumenter class."""
    
    def test_init_unsupported_database(self):
        """Test that unsupported databases raise ValueError."""
        with pytest.raises(ValueError, match="Schema documentation not supported for mysql"):
            SchemaDocumenter("mysql", {})
    
    def test_init_supported_databases(self):
        """Test that supported databases initialize correctly."""
        # Postgres
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})
        assert documenter.db_type == "postgres"
        
        # DuckDB
        documenter = SchemaDocumenter("duckdb", {"database": ":memory:"})
        assert documenter.db_type == "duckdb"
    
    def test_is_categorical_column(self):
        """Test categorical column detection logic."""
        documenter = SchemaDocumenter("postgres", {})
        
        # String type with low unique ratio
        stats = {"unique_count": 5, "total_count": 100}
        assert documenter._is_categorical_column("varchar", stats, []) is True
        
        # String type with high unique ratio
        stats = {"unique_count": 80, "total_count": 100}
        assert documenter._is_categorical_column("varchar", stats, []) is False
        
        # Boolean type
        stats = {"unique_count": 2, "total_count": 100}
        assert documenter._is_categorical_column("boolean", stats, []) is True
        
        # Enum type
        assert documenter._is_categorical_column("enum('a','b')", stats, []) is True
    
    def test_detect_data_patterns(self):
        """Test data pattern detection."""
        documenter = SchemaDocumenter("postgres", {})
        
        # Email pattern
        email_values = ["user@example.com", "test@gmail.com"]
        patterns = documenter._detect_data_patterns("varchar", email_values)
        assert "email_addresses" in patterns
        
        # URL pattern
        url_values = ["https://example.com", "http://test.org"]
        patterns = documenter._detect_data_patterns("varchar", url_values)
        assert "urls" in patterns
        
        # UUID pattern
        uuid_values = ["550e8400-e29b-41d4-a716-446655440000"]
        patterns = documenter._detect_data_patterns("varchar", uuid_values)
        assert "uuids" in patterns
        
        # No patterns
        normal_values = ["alice", "bob", "charlie"]
        patterns = documenter._detect_data_patterns("varchar", normal_values)
        assert len(patterns) == 0
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid LLM JSON response."""
        documenter = SchemaDocumenter("postgres", {})
        
        response_content = json.dumps({
            "table_description": "User information table",
            "column_descriptions": {
                "id": "Primary key identifier",
                "name": "User's full name"
            },
            "confidence_score": 0.9
        })
        
        table_desc, column_descs, confidence = documenter._parse_llm_response(response_content)
        
        assert table_desc == "User information table"
        assert column_descs["id"] == "Primary key identifier"
        assert column_descs["name"] == "User's full name"
        assert confidence == 0.9
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid LLM response."""
        documenter = SchemaDocumenter("postgres", {})
        
        response_content = "This is not valid JSON"
        
        table_desc, column_descs, confidence = documenter._parse_llm_response(response_content)
        
        assert table_desc == "AI-generated description (parsing error)"
        assert isinstance(column_descs, dict)
        assert confidence == 0.5
    
    def test_generate_fallback_documentation(self):
        """Test fallback documentation generation."""
        documenter = SchemaDocumenter("postgres", {})
        
        # Mock column analysis
        col_analysis = ColumnAnalysis(
            column_name="status",
            data_type="varchar",
            sample_values=["active", "inactive"],
            null_count=0,
            total_count=100,
            unique_count=2,
            is_categorical=True,
            categorical_values=["active", "inactive"],
            data_patterns=[]
        )
        
        table_analysis = {
            "table_name": "users",
            "total_rows": 100,
            "columns": [col_analysis]
        }
        
        table_desc, column_descs, confidence = documenter._generate_fallback_documentation(table_analysis)
        
        assert "100 rows" in table_desc
        assert "1 columns" in table_desc
        assert "status" in column_descs
        assert "varchar column" in column_descs["status"]
        assert confidence == 0.4
    
    def test_prepare_analysis_for_llm(self):
        """Test LLM analysis preparation."""
        documenter = SchemaDocumenter("postgres", {})
        
        # Mock column analysis
        col_analysis = ColumnAnalysis(
            column_name="email",
            data_type="varchar",
            sample_values=["user@example.com", "test@gmail.com"],
            null_count=5,
            total_count=100,
            unique_count=95,
            is_categorical=False,
            data_patterns=["email_addresses"]
        )
        
        table_analysis = {
            "table_name": "users",
            "total_rows": 100,
            "columns": [col_analysis],
            "sample_data": [{"email": "user@example.com", "id": 1}] * 10  # More than 5 rows
        }
        
        analysis = documenter._prepare_analysis_for_llm(table_analysis)
        
        assert analysis["table_name"] == "users"
        assert analysis["total_rows"] == 100
        assert len(analysis["sample_data"]) == 5  # Should be limited to 5
        assert len(analysis["columns"]) == 1
        
        col_summary = analysis["columns"][0]
        assert col_summary["name"] == "email"
        assert col_summary["type"] == "varchar"
        assert col_summary["patterns"] == ["email_addresses"]
        assert "sample_values" in col_summary


@pytest.mark.asyncio
class TestSchemaDocumenterAsync:
    """Test async functionality of SchemaDocumenter."""
    
    @patch('defog.schema_documenter.psycopg2')
    async def test_postgres_connection(self, mock_psycopg2):
        """Test Postgres connection handling."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        
        documenter = SchemaDocumenter("postgres", {"host": "localhost"})
        conn, cur = await documenter._get_connection()
        
        assert conn == mock_conn
        assert cur == mock_cur
        mock_psycopg2.connect.assert_called_once_with(host="localhost")
    
    @patch('defog.schema_documenter.duckdb')
    async def test_duckdb_connection(self, mock_duckdb):
        """Test DuckDB connection handling."""
        # Mock connection
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn
        
        documenter = SchemaDocumenter("duckdb", {"database": ":memory:"})
        conn, cur = await documenter._get_connection()
        
        assert conn == mock_conn
        assert cur == mock_conn  # DuckDB uses same object for both
        mock_duckdb.connect.assert_called_once_with(":memory:")
    
    @patch('defog.schema_documenter.chat_async')
    async def test_generate_documentation_llm_success(self, mock_chat_async):
        """Test successful LLM documentation generation."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "table_description": "User data table",
            "column_descriptions": {"id": "User identifier"},
            "confidence_score": 0.95
        })
        mock_chat_async.return_value = mock_response
        
        documenter = SchemaDocumenter("postgres", {})
        
        table_analysis = {
            "table_name": "users",
            "total_rows": 100,
            "columns": [],
            "sample_data": []
        }
        
        table_desc, column_descs, confidence = await documenter._generate_documentation_llm(table_analysis)
        
        assert table_desc == "User data table"
        assert column_descs["id"] == "User identifier"
        assert confidence == 0.95
        mock_chat_async.assert_called_once()
    
    @patch('defog.schema_documenter.chat_async')
    async def test_generate_documentation_llm_failure(self, mock_chat_async):
        """Test LLM documentation generation failure."""
        # Mock LLM failure
        mock_chat_async.side_effect = Exception("LLM service unavailable")
        
        documenter = SchemaDocumenter("postgres", {})
        
        table_analysis = {
            "table_name": "users",
            "total_rows": 100,
            "columns": [],
            "sample_data": []
        }
        
        table_desc, column_descs, confidence = await documenter._generate_documentation_llm(table_analysis)
        
        # Should fall back to basic documentation
        assert "100 rows" in table_desc
        assert confidence == 0.4


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('defog.schema_documenter.SchemaDocumenter')
    @pytest.mark.asyncio
    async def test_document_and_apply_schema_dry_run(self, mock_documenter_class):
        """Test document_and_apply_schema with dry_run=True."""
        # Mock documenter instance
        mock_documenter = AsyncMock()
        mock_documenter_class.return_value = mock_documenter
        
        mock_documentation = {
            "users": TableDocumentation(
                table_name="users",
                table_description="User table",
                column_descriptions={"id": "User ID"},
                confidence_score=0.9,
                analysis_metadata={}
            )
        }
        mock_documenter.document_schema.return_value = mock_documentation
        
        result = await document_and_apply_schema(
            db_type="postgres",
            db_creds={"host": "localhost"},
            dry_run=True
        )
        
        assert "documentation" in result
        assert "application_results" in result
        assert "summary" in result
        assert result["summary"]["tables_analyzed"] == 1
        assert result["summary"]["tables_applied"] == 0
        assert result["summary"]["dry_run"] is True
        
        # Should not call apply_documentation for dry run
        mock_documenter.apply_documentation.assert_not_called()
    
    @patch('defog.schema_documenter.SchemaDocumenter')
    @pytest.mark.asyncio
    async def test_document_and_apply_schema_apply(self, mock_documenter_class):
        """Test document_and_apply_schema with actual application."""
        # Mock documenter instance
        mock_documenter = AsyncMock()
        mock_documenter_class.return_value = mock_documenter
        
        mock_documentation = {
            "users": TableDocumentation(
                table_name="users",
                table_description="User table",
                column_descriptions={"id": "User ID"},
                confidence_score=0.9,
                analysis_metadata={}
            )
        }
        mock_documenter.document_schema.return_value = mock_documentation
        mock_documenter.apply_documentation.return_value = {"users": True}
        
        result = await document_and_apply_schema(
            db_type="postgres",
            db_creds={"host": "localhost"},
            dry_run=False
        )
        
        assert result["summary"]["tables_analyzed"] == 1
        assert result["summary"]["tables_applied"] == 1
        assert result["summary"]["dry_run"] is False
        
        # Should call apply_documentation
        mock_documenter.apply_documentation.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])