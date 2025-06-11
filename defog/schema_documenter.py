"""
LLM-powered database schema self-documentation.

This module provides functionality to use LLMs to automatically generate
table and column descriptions by analyzing database data, then store these
as database comments to improve SQL query generation quality.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import asyncio
import time

from .llm.utils import chat_async
from .llm.providers.base import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Configuration for schema documentation generation."""
    
    # LLM settings
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1
    
    # Data sampling settings
    sample_size: int = 100
    max_categorical_values: int = 20
    
    # Processing settings
    max_concurrent_requests: int = 3
    request_delay: float = 1.0
    
    # Documentation settings
    include_data_patterns: bool = True
    include_sample_values: bool = True
    min_confidence_score: float = 0.7
    
    # Database settings
    update_existing_comments: bool = True
    backup_existing_comments: bool = True


@dataclass
class TableDocumentation:
    """Documentation result for a table."""
    
    table_name: str
    table_description: str
    column_descriptions: Dict[str, str]
    confidence_score: float
    analysis_metadata: Dict[str, Any]


@dataclass
class ColumnAnalysis:
    """Analysis result for a column."""
    
    column_name: str
    data_type: str
    sample_values: List[Any]
    null_count: int
    total_count: int
    unique_count: int
    is_categorical: bool
    categorical_values: Optional[List[str]] = None
    data_patterns: Optional[List[str]] = None
    statistics: Optional[Dict[str, Any]] = None


class SchemaDocumenter:
    """LLM-powered database schema documenter."""
    
    def __init__(
        self,
        db_type: str,
        db_creds: Dict[str, Any],
        config: Optional[DocumentationConfig] = None,
    ):
        self.db_type = db_type.lower()
        self.db_creds = db_creds
        self.config = config or DocumentationConfig()
        
        # Validate database type support
        if self.db_type not in ["postgres", "duckdb"]:
            raise ValueError(f"Schema documentation not supported for {db_type}. Currently supports: postgres, duckdb")
    
    async def document_schema(
        self,
        tables: Optional[List[str]] = None,
        schemas: Optional[List[str]] = None,
    ) -> Dict[str, TableDocumentation]:
        """
        Generate LLM-powered documentation for database schema.
        
        Args:
            tables: List of specific tables to document. If None, documents all tables.
            schemas: List of schemas to include (Postgres only). If None, uses default.
        
        Returns:
            Dictionary mapping table names to their documentation.
        """
        logger.info(f"Starting schema documentation for {self.db_type} database")
        
        # Get connection
        conn, cur = await self._get_connection()
        
        try:
            # Get table list if not specified
            if not tables:
                tables = await self._get_table_list(cur, schemas)
            
            logger.info(f"Documenting {len(tables)} tables: {tables}")
            
            # Analyze and document tables
            results = {}
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            
            async def document_table_with_semaphore(table_name: str):
                async with semaphore:
                    if self.config.request_delay > 0:
                        await asyncio.sleep(self.config.request_delay)
                    return await self._document_single_table(cur, table_name)
            
            # Process tables concurrently
            tasks = [document_table_with_semaphore(table) for table in tables]
            documentation_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for i, result in enumerate(documentation_results):
                table_name = tables[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to document table {table_name}: {result}")
                else:
                    results[table_name] = result
            
            logger.info(f"Successfully documented {len(results)} out of {len(tables)} tables")
            return results
            
        finally:
            await self._close_connection(conn, cur)
    
    async def apply_documentation(
        self,
        documentation: Dict[str, TableDocumentation],
        dry_run: bool = False,
    ) -> Dict[str, bool]:
        """
        Apply generated documentation as database comments.
        
        Args:
            documentation: Dictionary of table documentation to apply.
            dry_run: If True, only validates but doesn't apply changes.
        
        Returns:
            Dictionary mapping table names to success status.
        """
        logger.info(f"{'Validating' if dry_run else 'Applying'} documentation for {len(documentation)} tables")
        
        conn, cur = await self._get_connection()
        
        try:
            results = {}
            
            for table_name, doc in documentation.items():
                try:
                    if doc.confidence_score < self.config.min_confidence_score:
                        logger.warning(f"Skipping {table_name} due to low confidence score: {doc.confidence_score}")
                        results[table_name] = False
                        continue
                    
                    if not dry_run:
                        await self._apply_table_documentation(cur, table_name, doc)
                    
                    results[table_name] = True
                    logger.info(f"{'Validated' if dry_run else 'Applied'} documentation for table {table_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to {'validate' if dry_run else 'apply'} documentation for {table_name}: {e}")
                    results[table_name] = False
            
            if not dry_run:
                await self._commit_changes(conn)
            
            return results
            
        finally:
            await self._close_connection(conn, cur)
    
    async def _get_connection(self) -> Tuple[Any, Any]:
        """Get database connection and cursor."""
        if self.db_type == "postgres":
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(**self.db_creds)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            return conn, cur
            
        elif self.db_type == "duckdb":
            import duckdb
            
            database_path = self.db_creds.get("database", ":memory:")
            conn = duckdb.connect(database_path)
            return conn, conn  # DuckDB connection serves as both conn and cursor
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    async def _close_connection(self, conn: Any, cur: Any) -> None:
        """Close database connection."""
        try:
            if self.db_type == "postgres":
                if cur:
                    cur.close()
                if conn:
                    conn.close()
            elif self.db_type == "duckdb":
                if conn:
                    conn.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    async def _commit_changes(self, conn: Any) -> None:
        """Commit database changes."""
        try:
            if self.db_type == "postgres":
                conn.commit()
            # DuckDB auto-commits by default
        except Exception as e:
            logger.error(f"Error committing changes: {e}")
            raise
    
    async def _get_table_list(self, cur: Any, schemas: Optional[List[str]] = None) -> List[str]:
        """Get list of tables to document."""
        if self.db_type == "postgres":
            if not schemas:
                schemas = ["public"]
            
            tables = []
            for schema in schemas:
                query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
                cur.execute(query, (schema,))
                schema_tables = [row["table_name"] for row in cur.fetchall()]
                
                if schema == "public":
                    tables.extend(schema_tables)
                else:
                    tables.extend([f"{schema}.{table}" for table in schema_tables])
            
            return tables
            
        elif self.db_type == "duckdb":
            query = """
            SELECT table_schema || '.' || table_name as full_name, table_name
            FROM information_schema.tables 
            WHERE table_type = 'BASE TABLE'
            AND table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_name
            """
            result = cur.execute(query).fetchall()
            
            # Return both qualified and unqualified names for main schema
            tables = []
            for row in result:
                full_name, table_name = row
                if full_name.startswith("main."):
                    tables.append(table_name)  # Add unqualified name for main schema
                tables.append(full_name)  # Add qualified name
            
            return list(set(tables))  # Remove duplicates
        
        return []
    
    async def _document_single_table(self, cur: Any, table_name: str) -> TableDocumentation:
        """Generate documentation for a single table."""
        logger.debug(f"Analyzing table: {table_name}")
        
        # Analyze table structure and data
        table_analysis = await self._analyze_table(cur, table_name)
        
        # Generate documentation using LLM
        table_doc, column_docs, confidence = await self._generate_documentation_llm(table_analysis)
        
        return TableDocumentation(
            table_name=table_name,
            table_description=table_doc,
            column_descriptions=column_docs,
            confidence_score=confidence,
            analysis_metadata={
                "columns_analyzed": len(table_analysis["columns"]),
                "total_rows": table_analysis["total_rows"],
                "analysis_timestamp": time.time(),
            }
        )
    
    async def _analyze_table(self, cur: Any, table_name: str) -> Dict[str, Any]:
        """Analyze table structure and sample data."""
        # Get basic table info
        columns_info = await self._get_table_columns(cur, table_name)
        total_rows = await self._get_row_count(cur, table_name)
        
        # Analyze each column
        column_analyses = []
        for col_info in columns_info:
            col_analysis = await self._analyze_column(cur, table_name, col_info)
            column_analyses.append(col_analysis)
        
        return {
            "table_name": table_name,
            "total_rows": total_rows,
            "columns": column_analyses,
            "sample_data": await self._get_sample_data(cur, table_name),
        }
    
    async def _get_table_columns(self, cur: Any, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        if self.db_type == "postgres":
            # Handle schema.table format
            if "." in table_name:
                schema, table = table_name.split(".", 1)
            else:
                schema, table = "public", table_name
            
            query = """
            SELECT column_name, data_type, is_nullable,
                   col_description(
                       FORMAT('%s.%s', table_schema, table_name)::regclass::oid, 
                       ordinal_position
                   ) AS existing_comment
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """
            cur.execute(query, (schema, table))
            return [dict(row) for row in cur.fetchall()]
            
        elif self.db_type == "duckdb":
            # Handle schema.table format
            if "." in table_name:
                schema, table = table_name.split(".", 1)
                query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_schema = ? AND table_name = ?
                ORDER BY ordinal_position
                """
                params = (schema, table)
            else:
                query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = ? AND table_schema = 'main'
                ORDER BY ordinal_position
                """
                params = (table_name,)
            
            result = cur.execute(query, params).fetchall()
            return [{"column_name": row[0], "data_type": row[1], "is_nullable": row[2], "existing_comment": None} for row in result]
        
        return []
    
    async def _get_row_count(self, cur: Any, table_name: str) -> int:
        """Get total row count for a table."""
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            if self.db_type == "postgres":
                cur.execute(query)
                return cur.fetchone()[0]
            else:
                return cur.execute(query).fetchone()[0]
        except Exception as e:
            logger.warning(f"Could not get row count for {table_name}: {e}")
            return 0
    
    async def _analyze_column(self, cur: Any, table_name: str, col_info: Dict[str, Any]) -> ColumnAnalysis:
        """Analyze a single column."""
        column_name = col_info["column_name"]
        data_type = col_info["data_type"]
        
        # Get basic statistics
        stats = await self._get_column_statistics(cur, table_name, column_name)
        
        # Get sample values
        sample_values = await self._get_sample_values(cur, table_name, column_name)
        
        # Determine if categorical
        is_categorical = self._is_categorical_column(data_type, stats, sample_values)
        categorical_values = None
        
        if is_categorical and stats["unique_count"] <= self.config.max_categorical_values:
            categorical_values = await self._get_categorical_values(cur, table_name, column_name)
        
        # Detect data patterns
        data_patterns = self._detect_data_patterns(data_type, sample_values)
        
        return ColumnAnalysis(
            column_name=column_name,
            data_type=data_type,
            sample_values=sample_values,
            null_count=stats["null_count"],
            total_count=stats["total_count"],
            unique_count=stats["unique_count"],
            is_categorical=is_categorical,
            categorical_values=categorical_values,
            data_patterns=data_patterns,
            statistics=stats,
        )
    
    async def _get_column_statistics(self, cur: Any, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get basic statistics for a column."""
        try:
            query = f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT({column_name}) as non_null_count,
                COUNT(*) - COUNT({column_name}) as null_count,
                COUNT(DISTINCT {column_name}) as unique_count
            FROM {table_name}
            """
            
            if self.db_type == "postgres":
                cur.execute(query)
                result = dict(cur.fetchone())
            else:
                result = dict(zip(["total_count", "non_null_count", "null_count", "unique_count"], 
                                cur.execute(query).fetchone()))
            
            return result
            
        except Exception as e:
            logger.warning(f"Could not get statistics for {table_name}.{column_name}: {e}")
            return {"total_count": 0, "non_null_count": 0, "null_count": 0, "unique_count": 0}
    
    async def _get_sample_values(self, cur: Any, table_name: str, column_name: str) -> List[Any]:
        """Get sample values from a column."""
        try:
            query = f"""
            SELECT DISTINCT {column_name} 
            FROM {table_name} 
            WHERE {column_name} IS NOT NULL 
            LIMIT {self.config.sample_size}
            """
            
            if self.db_type == "postgres":
                cur.execute(query)
                return [row[0] for row in cur.fetchall()]
            else:
                return [row[0] for row in cur.execute(query).fetchall()]
                
        except Exception as e:
            logger.warning(f"Could not get sample values for {table_name}.{column_name}: {e}")
            return []
    
    async def _get_sample_data(self, cur: Any, table_name: str) -> List[Dict[str, Any]]:
        """Get sample rows from the table."""
        try:
            limit = min(self.config.sample_size // 10, 20)  # Fewer rows for full table samples
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            
            if self.db_type == "postgres":
                cur.execute(query)
                return [dict(row) for row in cur.fetchall()]
            else:
                columns = [desc[0] for desc in cur.description] if hasattr(cur, 'description') and cur.description else []
                rows = cur.execute(query).fetchall()
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.warning(f"Could not get sample data for {table_name}: {e}")
            return []
    
    def _is_categorical_column(self, data_type: str, stats: Dict[str, Any], sample_values: List[Any]) -> bool:
        """Determine if a column is categorical."""
        # String types with limited unique values
        if data_type.lower() in ["text", "varchar", "character varying", "char", "string"]:
            unique_ratio = stats["unique_count"] / max(stats["total_count"], 1)
            return unique_ratio < 0.1 and stats["unique_count"] <= self.config.max_categorical_values
        
        # Boolean type
        if data_type.lower() in ["boolean", "bool"]:
            return True
        
        # Enum types
        if "enum" in data_type.lower():
            return True
        
        return False
    
    async def _get_categorical_values(self, cur: Any, table_name: str, column_name: str) -> List[str]:
        """Get all distinct values for a categorical column."""
        try:
            query = f"""
            SELECT {column_name}, COUNT(*) as count
            FROM {table_name} 
            WHERE {column_name} IS NOT NULL
            GROUP BY {column_name}
            ORDER BY count DESC
            LIMIT {self.config.max_categorical_values}
            """
            
            if self.db_type == "postgres":
                cur.execute(query)
                return [str(row[0]) for row in cur.fetchall()]
            else:
                return [str(row[0]) for row in cur.execute(query).fetchall()]
                
        except Exception as e:
            logger.warning(f"Could not get categorical values for {table_name}.{column_name}: {e}")
            return []
    
    def _detect_data_patterns(self, data_type: str, sample_values: List[Any]) -> List[str]:
        """Detect common patterns in the data."""
        if not sample_values or not self.config.include_data_patterns:
            return []
        
        patterns = []
        
        # Convert to strings for pattern analysis
        str_values = [str(v) for v in sample_values if v is not None]
        
        if not str_values:
            return patterns
        
        # Check for common patterns
        import re
        
        # Email pattern
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if any(email_pattern.match(v) for v in str_values[:10]):
            patterns.append("email_addresses")
        
        # URL pattern
        url_pattern = re.compile(r'^https?://')
        if any(url_pattern.match(v) for v in str_values[:10]):
            patterns.append("urls")
        
        # Phone number pattern (basic)
        phone_pattern = re.compile(r'^\+?[\d\s\-\(\)]{10,}$')
        if any(phone_pattern.match(v) for v in str_values[:10]):
            patterns.append("phone_numbers")
        
        # Date pattern (ISO format)
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}')
        if any(date_pattern.match(v) for v in str_values[:10]):
            patterns.append("dates")
        
        # UUID pattern
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
        if any(uuid_pattern.match(v) for v in str_values[:10]):
            patterns.append("uuids")
        
        # Numeric ID pattern
        if data_type.lower() in ["integer", "bigint", "int", "int4", "int8"]:
            # Check if values are sequential or look like IDs
            numeric_values = []
            for v in sample_values[:20]:
                try:
                    numeric_values.append(int(v))
                except (ValueError, TypeError):
                    continue
            
            if numeric_values:
                if len(set(numeric_values)) == len(numeric_values):  # All unique
                    patterns.append("unique_identifiers")
                elif all(v > 0 for v in numeric_values):  # All positive
                    patterns.append("positive_integers")
        
        return patterns
    
    async def _generate_documentation_llm(self, table_analysis: Dict[str, Any]) -> Tuple[str, Dict[str, str], float]:
        """Generate documentation using LLM."""
        # Prepare analysis for LLM
        analysis_summary = self._prepare_analysis_for_llm(table_analysis)
        
        # Create prompt for LLM
        prompt = self._create_documentation_prompt(analysis_summary)
        
        # Call LLM
        try:
            response = await chat_async(
                provider=self.config.provider,
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
            )
            
            # Parse response
            return self._parse_llm_response(response.content)
            
        except Exception as e:
            logger.error(f"LLM documentation generation failed: {e}")
            # Fallback to basic descriptions
            return self._generate_fallback_documentation(table_analysis)
    
    def _prepare_analysis_for_llm(self, table_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare table analysis data for LLM consumption."""
        # Limit sample data size to avoid token limits
        sample_data = table_analysis["sample_data"][:5]  # Only first 5 rows
        
        # Prepare column summaries
        column_summaries = []
        for col in table_analysis["columns"]:
            col_summary = {
                "name": col.column_name,
                "type": col.data_type,
                "null_rate": col.null_count / max(col.total_count, 1),
                "unique_rate": col.unique_count / max(col.total_count, 1),
                "is_categorical": col.is_categorical,
            }
            
            if col.categorical_values:
                col_summary["categories"] = col.categorical_values[:10]  # Limit categories
            
            if col.data_patterns:
                col_summary["patterns"] = col.data_patterns
            
            if col.sample_values and self.config.include_sample_values:
                col_summary["sample_values"] = [str(v)[:50] for v in col.sample_values[:5]]  # Truncate long values
            
            column_summaries.append(col_summary)
        
        return {
            "table_name": table_analysis["table_name"],
            "total_rows": table_analysis["total_rows"],
            "columns": column_summaries,
            "sample_data": sample_data,
        }
    
    def _create_documentation_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create prompt for LLM documentation generation."""
        return f"""You are a database documentation expert. Analyze the following table and generate clear, concise descriptions.

Table: {analysis['table_name']}
Total Rows: {analysis['total_rows']}

Column Analysis:
{json.dumps(analysis['columns'], indent=2)}

Sample Data (first few rows):
{json.dumps(analysis['sample_data'], indent=2)}

Please provide:
1. A brief, clear description of what this table represents and its purpose
2. For each column, provide a concise description explaining what the column contains and its business meaning

Requirements:
- Keep descriptions factual and based on the data patterns you observe
- Use clear, professional language suitable for developers and analysts
- Focus on business meaning rather than technical implementation details
- For categorical columns, mention the possible values if they're meaningful
- Highlight any important data patterns you notice

Respond in this exact JSON format:
{{
    "table_description": "Description of the table's purpose and contents",
    "column_descriptions": {{
        "column_name_1": "Description of what this column contains",
        "column_name_2": "Description of what this column contains"
    }},
    "confidence_score": 0.95
}}

The confidence_score should be between 0.0 and 1.0, reflecting how confident you are in your analysis based on the available data."""
    
    def _parse_llm_response(self, response_content: str) -> Tuple[str, Dict[str, str], float]:
        """Parse LLM response into structured documentation."""
        try:
            # Try to parse as JSON
            response_data = json.loads(response_content)
            
            table_desc = response_data.get("table_description", "")
            column_descs = response_data.get("column_descriptions", {})
            confidence = float(response_data.get("confidence_score", 0.7))
            
            return table_desc, column_descs, confidence
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Could not parse LLM response as JSON: {e}")
            
            # Try to extract information from text response
            lines = response_content.split('\n')
            table_desc = "AI-generated description (parsing error)"
            column_descs = {}
            confidence = 0.5
            
            # Basic fallback parsing
            current_section = None
            for line in lines:
                line = line.strip()
                if "table" in line.lower() and "description" in line.lower():
                    current_section = "table"
                elif "column" in line.lower() and "description" in line.lower():
                    current_section = "columns"
                elif current_section == "table" and line and not line.startswith('-'):
                    table_desc = line
                elif current_section == "columns" and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        col_name = parts[0].strip().strip('"\'`')
                        col_desc = parts[1].strip()
                        column_descs[col_name] = col_desc
            
            return table_desc, column_descs, confidence
    
    def _generate_fallback_documentation(self, table_analysis: Dict[str, Any]) -> Tuple[str, Dict[str, str], float]:
        """Generate basic fallback documentation when LLM fails."""
        table_name = table_analysis["table_name"]
        table_desc = f"Table containing {table_analysis['total_rows']} rows with {len(table_analysis['columns'])} columns."
        
        column_descs = {}
        for col in table_analysis["columns"]:
            desc_parts = [f"{col.data_type} column"]
            
            if col.is_categorical and col.categorical_values:
                desc_parts.append(f"with values: {', '.join(col.categorical_values[:5])}")
            
            if col.data_patterns:
                desc_parts.append(f"({', '.join(col.data_patterns)})")
            
            column_descs[col.column_name] = ". ".join(desc_parts)
        
        return table_desc, column_descs, 0.4  # Low confidence for fallback
    
    async def _apply_table_documentation(self, cur: Any, table_name: str, doc: TableDocumentation) -> None:
        """Apply documentation as database comments."""
        # Backup existing comments if requested
        if self.config.backup_existing_comments:
            await self._backup_existing_comments(cur, table_name)
        
        # Apply table comment
        if doc.table_description:
            await self._set_table_comment(cur, table_name, doc.table_description)
        
        # Apply column comments
        for column_name, description in doc.column_descriptions.items():
            if description:
                await self._set_column_comment(cur, table_name, column_name, description)
    
    async def _backup_existing_comments(self, cur: Any, table_name: str) -> None:
        """Backup existing comments before overwriting."""
        # This could be implemented to store existing comments in a backup table
        # For now, just log that we're about to overwrite
        logger.debug(f"Backing up existing comments for {table_name} (not implemented)")
    
    async def _set_table_comment(self, cur: Any, table_name: str, description: str) -> None:
        """Set table comment."""
        if self.db_type == "postgres":
            query = f"COMMENT ON TABLE {table_name} IS %s"
            cur.execute(query, (description,))
        elif self.db_type == "duckdb":
            # DuckDB supports COMMENT ON TABLE
            query = f"COMMENT ON TABLE {table_name} IS ?"
            cur.execute(query, (description,))
    
    async def _set_column_comment(self, cur: Any, table_name: str, column_name: str, description: str) -> None:
        """Set column comment."""
        if self.db_type == "postgres":
            query = f"COMMENT ON COLUMN {table_name}.{column_name} IS %s"
            cur.execute(query, (description,))
        elif self.db_type == "duckdb":
            # DuckDB supports COMMENT ON COLUMN
            query = f"COMMENT ON COLUMN {table_name}.{column_name} IS ?"
            cur.execute(query, (description,))


# Convenience functions for integration with existing code

async def document_and_apply_schema(
    db_type: str,
    db_creds: Dict[str, Any],
    tables: Optional[List[str]] = None,
    schemas: Optional[List[str]] = None,
    config: Optional[DocumentationConfig] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to document and optionally apply schema documentation.
    
    Args:
        db_type: Database type (postgres, duckdb)
        db_creds: Database connection credentials
        tables: List of tables to document (None for all)
        schemas: List of schemas to include (Postgres only)
        config: Documentation configuration
        dry_run: If True, only generates documentation without applying
    
    Returns:
        Dictionary with documentation results and application status.
    """
    documenter = SchemaDocumenter(db_type, db_creds, config)
    
    # Generate documentation
    documentation = await documenter.document_schema(tables, schemas)
    
    # Apply documentation if not dry run
    application_results = {}
    if not dry_run and documentation:
        application_results = await documenter.apply_documentation(documentation)
    
    return {
        "documentation": documentation,
        "application_results": application_results,
        "summary": {
            "tables_analyzed": len(documentation),
            "tables_applied": sum(application_results.values()) if application_results else 0,
            "dry_run": dry_run,
        }
    }