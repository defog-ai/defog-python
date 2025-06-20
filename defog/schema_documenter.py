"""
LLM-powered database schema documentation system.

This module provides functionality to automatically generate table and column
descriptions using LLMs by analyzing database structure and sample data,
then storing these descriptions as database comments.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
from contextlib import contextmanager

# Import LLM providers
from defog.llm.utils import chat_async

from pydantic import BaseModel


class ColumnDocumentation(BaseModel):
    """Pydantic model for column documentation."""

    description: str
    confidence: float


class SchemaDocumentationResponse(BaseModel):
    """Pydantic model for LLM response containing schema documentation."""

    table_description: str
    table_confidence: float
    columns: Dict[str, ColumnDocumentation]


# Constants for data analysis
MAX_CATEGORICAL_VALUES = 20
CATEGORICAL_THRESHOLD = 0.5
SAMPLE_VALUES_LIMIT = 3
MAX_CATEGORY_EXAMPLES = 10
EMAIL_PATTERN_THRESHOLD = 0.8
URL_PATTERN_THRESHOLD = 0.8
UUID_PATTERN_THRESHOLD = 0.8
PHONE_PATTERN_THRESHOLD = 0.8


def _validate_sql_identifier(identifier: str) -> str:
    """
    Validate SQL identifiers (table names, column names) to prevent injection.

    Args:
        identifier: The SQL identifier to validate

    Returns:
        Validated identifier safe for use in SQL queries

    Raises:
        ValueError: If identifier contains invalid characters
    """
    # Length check (255 bytes is reasonable for most databases)
    if len(identifier.encode("utf-8")) > 255:
        raise ValueError(
            f"SQL identifier too long: {len(identifier.encode('utf-8'))} bytes (max 255)"
        )

    # Reject zero-width and invisible characters
    invisible_chars = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
    for char in invisible_chars:
        if char in identifier:
            raise ValueError("SQL identifier contains invisible/zero-width characters")

    # Handle quoted identifiers (double quotes)
    if identifier.startswith('"') and identifier.endswith('"'):
        # For quoted identifiers, we're more permissive but still check for truly dangerous patterns
        inner_content = identifier[1:-1]
        # Check for dangerous SQL injection patterns within quotes
        # We allow SQL keywords as table/column names when quoted, but not injection patterns
        dangerous_patterns = [
            r"[;\-\-]",  # Semicolon and SQL comments
            r"/\*.*\*/",  # Block comments
            r"'.*'",  # Single quotes (potential injection)
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, inner_content, re.IGNORECASE):
                raise ValueError(
                    f"SQL identifier contains invalid pattern: {identifier}"
                )

        return identifier

    # For unquoted identifiers, strict validation - only allow alphanumeric, underscore, and dot
    if not re.match(r"^[a-zA-Z0-9_.]+$", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")

    # Additional validation for dangerous patterns
    dangerous_patterns = [
        r"\b(drop|delete|insert|update|alter|create|exec|union|select)\b",
        r"[;\-\+\*\/\\]",
        r"\s+",  # No whitespace
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, identifier, re.IGNORECASE):
            raise ValueError(f"SQL identifier contains invalid pattern: {identifier}")

    return identifier


@dataclass
class DocumentationConfig:
    """Configuration for schema documentation generation."""

    provider: str = "anthropic"  # anthropic, openai, gemini
    model: str = "claude-sonnet-4-20250514"
    sample_size: int = 100
    min_confidence_score: float = 0.7
    include_data_patterns: bool = True
    dry_run: bool = False
    concurrent_requests: int = 3
    rate_limit_delay: float = 1.0


@dataclass
class TableDocumentation:
    """Documentation for a database table."""

    table_name: str
    table_description: Optional[str] = None
    table_confidence: float = 0.0
    column_descriptions: Dict[str, str] = None
    column_confidences: Dict[str, float] = None
    existing_table_comment: Optional[str] = None
    existing_column_comments: Dict[str, str] = None

    def __post_init__(self):
        if self.column_descriptions is None:
            self.column_descriptions = {}
        if self.column_confidences is None:
            self.column_confidences = {}
        if self.existing_column_comments is None:
            self.existing_column_comments = {}


class SchemaDocumenter:
    """Main class for LLM-powered database schema documentation."""

    def __init__(
        self, db_type: str, db_creds: Dict, config: Optional[DocumentationConfig] = None
    ):
        self.db_type = db_type.lower()
        self.db_creds = db_creds
        self.config = config or DocumentationConfig()
        self.logger = logging.getLogger(__name__)

        # Validate database type
        if self.db_type not in ["postgres", "duckdb"]:
            raise ValueError(
                f"Database type '{db_type}' not supported. Use 'postgres' or 'duckdb'."
            )

    @contextmanager
    def _get_cursor(self, conn):
        """Context manager for database cursor."""
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _get_connection(self):
        """Get database connection based on db_type."""
        if self.db_type == "postgres":
            try:
                import psycopg2

                return psycopg2.connect(**self.db_creds)
            except ImportError:
                raise ImportError(
                    "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
                )
        elif self.db_type == "duckdb":
            try:
                import duckdb

                database_path = self.db_creds.get("database", ":memory:")
                return duckdb.connect(database_path)
            except ImportError:
                raise ImportError(
                    "duckdb not installed. Please install it with `pip install duckdb`."
                )

    def _get_existing_comments(
        self, conn, table_name: str
    ) -> Tuple[Optional[str], Dict[str, str]]:
        """Retrieve existing table and column comments."""
        if self.db_type == "postgres":
            return self._get_postgres_comments(conn, table_name)
        elif self.db_type == "duckdb":
            return self._get_duckdb_comments(conn, table_name)

    def _get_postgres_comments(
        self, conn, table_name: str
    ) -> Tuple[Optional[str], Dict[str, str]]:
        """Get existing comments for Postgres table and columns."""
        cursor = conn.cursor()

        # Parse schema and table name
        if "." in table_name:
            schema, table = table_name.split(".", 1)
        else:
            schema, table = "public", table_name

        # Get table comment
        cursor.execute(
            """
            SELECT obj_description(c.oid, 'pg_class') as table_comment
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = %s
        """,
            (table, schema),
        )

        result = cursor.fetchone()
        table_comment = result[0] if result and result[0] else None

        # Get column comments
        cursor.execute(
            """
            SELECT 
                column_name,
                col_description(
                    FORMAT('%%s.%%s', table_schema, table_name)::regclass::oid, 
                    ordinal_position
                ) AS column_comment
            FROM information_schema.columns 
            WHERE table_name = %s AND table_schema = %s
            AND col_description(
                FORMAT('%%s.%%s', table_schema, table_name)::regclass::oid, 
                ordinal_position
            ) IS NOT NULL
        """,
            (table, schema),
        )

        column_comments = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()

        return table_comment, column_comments

    def _get_duckdb_comments(
        self, conn, table_name: str
    ) -> Tuple[Optional[str], Dict[str, str]]:
        """Get existing comments for DuckDB table and columns."""
        # DuckDB comment support is limited, return empty for now
        # This can be enhanced when DuckDB adds better comment support
        return None, {}

    def _analyze_table_structure(self, conn, table_name: str) -> Dict[str, Any]:
        """Analyze table structure and sample data."""
        cursor = conn.cursor()

        # Get column information
        if self.db_type == "postgres":
            columns_info = self._get_postgres_columns(cursor, table_name)
        elif self.db_type == "duckdb":
            columns_info = self._get_duckdb_columns(cursor, table_name)

        # Get sample data
        sample_data = self._get_sample_data(cursor, table_name, columns_info)

        # Analyze data patterns
        data_patterns = {}
        if self.config.include_data_patterns:
            data_patterns = self._analyze_data_patterns(sample_data, columns_info)

        cursor.close()

        return {
            "columns": columns_info,
            "sample_data": sample_data,
            "data_patterns": data_patterns,
            "row_count": self._get_row_count(conn, table_name),
        }

    def _get_postgres_columns(self, cursor, table_name: str) -> List[Dict]:
        """Get column information for Postgres."""
        if "." in table_name:
            schema, table = table_name.split(".", 1)
        else:
            schema, table = "public", table_name

        cursor.execute(
            """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns 
            WHERE table_name = %s AND table_schema = %s
            ORDER BY ordinal_position
        """,
            (table, schema),
        )

        return [
            {
                "name": row[0],
                "type": row[1],
                "nullable": row[2] == "YES",
                "default": row[3],
                "max_length": row[4],
                "precision": row[5],
                "scale": row[6],
            }
            for row in cursor.fetchall()
        ]

    def _get_duckdb_columns(self, cursor, table_name: str) -> List[Dict]:
        """Get column information for DuckDB using parameterized queries."""
        try:
            validated_table = _validate_sql_identifier(table_name)

            if "." in validated_table:
                schema_name, table_only = validated_table.split(".", 1)
                query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_schema = ? AND table_name = ?
                    ORDER BY ordinal_position
                """
                cursor.execute(query, (schema_name, table_only))
            else:
                query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = ? AND table_schema = ?
                    ORDER BY ordinal_position
                """
                cursor.execute(query, (validated_table, "main"))

            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3],
                    "max_length": None,
                    "precision": None,
                    "scale": None,
                }
                for row in cursor.fetchall()
            ]
        except ValueError as e:
            self.logger.error(f"Invalid table name {table_name}: {e}")
            return []

    def _get_sample_data(
        self, cursor, table_name: str, columns_info: List[Dict]
    ) -> Dict[str, List]:
        """Get sample data for analysis using parameterized queries."""
        column_names = [col["name"] for col in columns_info]

        try:
            # Validate identifiers strictly
            validated_table = _validate_sql_identifier(table_name)
            validated_columns = [
                _validate_sql_identifier(name) for name in column_names
            ]

            # Build query with proper identifier quoting (not parameterization as identifiers can't be parameterized)
            if self.db_type == "postgres":
                # Parse schema.table if present
                if "." in validated_table:
                    schema, table = validated_table.split(".", 1)
                    quoted_table = f'"{schema}"."{table}"'
                else:
                    quoted_table = f'"{validated_table}"'

                quoted_columns = [f'"{col}"' for col in validated_columns]
                column_list = ", ".join(quoted_columns)

                # Use parameterized limit
                query = f"SELECT {column_list} FROM {quoted_table} LIMIT %s"
                cursor.execute(query, (self.config.sample_size,))

            elif self.db_type == "duckdb":
                # DuckDB uses different parameter syntax
                if "." in validated_table:
                    schema, table = validated_table.split(".", 1)
                    quoted_table = f'"{schema}"."{table}"'
                else:
                    quoted_table = f'"{validated_table}"'

                quoted_columns = [f'"{col}"' for col in validated_columns]
                column_list = ", ".join(quoted_columns)

                # Use parameterized limit with DuckDB syntax
                query = f"SELECT {column_list} FROM {quoted_table} LIMIT ?"
                cursor.execute(query, (self.config.sample_size,))

            rows = cursor.fetchall()

            # Organize data by column
            sample_data = {col_name: [] for col_name in column_names}
            for row in rows:
                for i, col_name in enumerate(column_names):
                    if i < len(row) and row[i] is not None:
                        sample_data[col_name].append(row[i])

            return sample_data
        except ValueError as e:
            self.logger.error(f"Invalid identifier in table {table_name}: {e}")
            return {col_name: [] for col_name in column_names}
        except Exception as e:
            self.logger.warning(f"Could not retrieve sample data for {table_name}: {e}")
            return {col_name: [] for col_name in column_names}

    def _get_row_count(self, conn, table_name: str) -> int:
        """Get approximate row count for the table using safe identifier validation."""
        try:
            validated_table = _validate_sql_identifier(table_name)

            with self._get_cursor(conn) as cursor:
                # Build properly quoted query (identifiers can't be parameterized)
                if self.db_type == "postgres":
                    if "." in validated_table:
                        schema, table = validated_table.split(".", 1)
                        quoted_table = f'"{schema}"."{table}"'
                    else:
                        quoted_table = f'"{validated_table}"'
                elif self.db_type == "duckdb":
                    if "." in validated_table:
                        schema, table = validated_table.split(".", 1)
                        quoted_table = f'"{schema}"."{table}"'
                    else:
                        quoted_table = f'"{validated_table}"'
                else:
                    quoted_table = f'"{validated_table}"'

                cursor.execute(f"SELECT COUNT(*) FROM {quoted_table}")
                result = cursor.fetchone()
                return result[0] if result else 0
        except ValueError as e:
            self.logger.error(f"Invalid table identifier: {table_name} - {e}")
            return 0
        except Exception as e:
            self.logger.warning(f"Could not get row count for {table_name}: {e}")
            return 0

    def _analyze_data_patterns(
        self, sample_data: Dict[str, List], columns_info: List[Dict]
    ) -> Dict[str, Dict]:
        """Analyze patterns in the sample data."""
        patterns = {}

        for col_info in columns_info:
            col_name = col_info["name"]
            col_data = sample_data.get(col_name, [])

            if not col_data:
                continue

            col_patterns = {}

            # String patterns
            if col_info["type"].lower() in [
                "text",
                "varchar",
                "character varying",
                "string",
            ]:
                col_patterns.update(self._analyze_string_patterns(col_data))

            # Numeric patterns
            elif col_info["type"].lower() in [
                "integer",
                "bigint",
                "decimal",
                "numeric",
                "real",
                "double",
            ]:
                col_patterns.update(self._analyze_numeric_patterns(col_data))

            # Date/time patterns
            elif (
                "date" in col_info["type"].lower() or "time" in col_info["type"].lower()
            ):
                col_patterns.update(self._analyze_datetime_patterns(col_data))

            # General patterns
            col_patterns.update(self._analyze_general_patterns(col_data))

            if col_patterns:
                patterns[col_name] = col_patterns

        return patterns

    def _analyze_string_patterns(self, data: List) -> Dict:
        """Analyze patterns in string data."""
        patterns = {}

        if not data:
            return patterns

        # Email pattern
        email_count = sum(1 for item in data if self._is_email(str(item)))
        if email_count / len(data) > EMAIL_PATTERN_THRESHOLD:
            patterns["email"] = True

        # URL pattern
        url_count = sum(1 for item in data if self._is_url(str(item)))
        if url_count / len(data) > URL_PATTERN_THRESHOLD:
            patterns["url"] = True

        # UUID pattern
        uuid_count = sum(1 for item in data if self._is_uuid(str(item)))
        if uuid_count / len(data) > UUID_PATTERN_THRESHOLD:
            patterns["uuid"] = True

        # Phone number pattern
        phone_count = sum(1 for item in data if self._is_phone(str(item)))
        if phone_count / len(data) > PHONE_PATTERN_THRESHOLD:
            patterns["phone"] = True

        # Categories (limited unique values)
        unique_values = set(str(item) for item in data)
        if (
            len(unique_values) <= MAX_CATEGORICAL_VALUES
            and len(unique_values) < len(data) * CATEGORICAL_THRESHOLD
        ):
            patterns["categorical"] = True
            patterns["categories"] = list(unique_values)[:MAX_CATEGORY_EXAMPLES]

        return patterns

    def _analyze_numeric_patterns(self, data: List) -> Dict:
        """Analyze patterns in numeric data."""
        patterns = {}

        if not data:
            return patterns

        # Check if values look like IDs (sequential, unique)
        try:
            numeric_data = [float(item) for item in data if item is not None]
            if len(set(numeric_data)) == len(numeric_data):  # All unique
                patterns["unique"] = True
                if all(x == int(x) for x in numeric_data):  # All integers
                    patterns["id_like"] = True
        except (ValueError, TypeError):
            pass

        return patterns

    def _analyze_datetime_patterns(self, data: List) -> Dict:
        """Analyze patterns in datetime data."""
        return {"datetime": True}

    def _analyze_general_patterns(self, data: List) -> Dict:
        """Analyze general patterns in data."""
        patterns = {}

        if not data:
            return patterns

        # Null percentage
        null_count = sum(1 for item in data if item is None or str(item).strip() == "")
        if null_count > 0:
            patterns["null_percentage"] = round(null_count / len(data) * 100, 1)

        return patterns

    # Pattern matching utility functions
    def _is_email(self, text: str) -> bool:
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(email_pattern, text.strip()))

    def _is_url(self, text: str) -> bool:
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        return bool(re.match(url_pattern, text.strip()))

    def _is_uuid(self, text: str) -> bool:
        uuid_pattern = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        return bool(re.match(uuid_pattern, text.strip()))

    def _is_phone(self, text: str) -> bool:
        phone_pattern = r"^[\+]?[1-9][\d\s\-\(\)]{7,15}$"
        return bool(re.match(phone_pattern, text.strip()))

    async def _generate_description(
        self, table_name: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate LLM-based descriptions for table and columns."""
        prompt = self._build_documentation_prompt(table_name, analysis)

        try:
            # Make async call to LLM with structured response format
            response = await chat_async(
                messages=[{"role": "user", "content": prompt}],
                provider=self.config.provider,
                model=self.config.model,
                temperature=0.1,
                response_format=SchemaDocumentationResponse,
            )

            # Convert the structured response to dictionary format
            result = {
                "table_description": response.content.table_description,
                "table_confidence": response.content.table_confidence,
                "columns": {},
            }

            for col_name, col_doc in response.content.columns.items():
                result["columns"][col_name] = {
                    "description": col_doc.description,
                    "confidence": col_doc.confidence,
                }

            return result

        except Exception as e:
            self.logger.error(f"Error generating description for {table_name}: {e}")
            return {"error": str(e)}

    def _build_documentation_prompt(
        self, table_name: str, analysis: Dict[str, Any]
    ) -> str:
        """Build the prompt for LLM to generate documentation."""
        columns = analysis["columns"]
        sample_data = analysis.get("sample_data", {})
        patterns = analysis.get("data_patterns", {})
        row_count = analysis.get("row_count", 0)

        prompt = f"""You are a database documentation expert. Analyze the following table and generate clear, concise descriptions.

TABLE: {table_name}
ROW COUNT: {row_count:,}

COLUMNS:
"""

        for col in columns:
            col_name = col["name"]
            col_type = col["type"]
            nullable = "NULL" if col["nullable"] else "NOT NULL"

            prompt += f"\n- {col_name} ({col_type}, {nullable})"

            # Add pattern information
            if col_name in patterns:
                col_patterns = patterns[col_name]
                if "email" in col_patterns:
                    prompt += " [EMAIL ADDRESSES]"
                elif "url" in col_patterns:
                    prompt += " [URLs]"
                elif "uuid" in col_patterns:
                    prompt += " [UUIDs]"
                elif "phone" in col_patterns:
                    prompt += " [PHONE NUMBERS]"
                elif "id_like" in col_patterns:
                    prompt += " [ID/SEQUENCE]"
                elif "categorical" in col_patterns:
                    categories = col_patterns.get("categories", [])
                    prompt += (
                        f" [CATEGORICAL: {', '.join(map(str, categories[:5]))}...]"
                    )

            # Add sample values
            if col_name in sample_data and sample_data[col_name]:
                samples = sample_data[col_name][:SAMPLE_VALUES_LIMIT]
                sample_str = ", ".join(f"'{s}'" for s in samples if s is not None)
                if sample_str:
                    prompt += f" [SAMPLES: {sample_str}]"

        prompt += """

Please provide concise documentation for this table and its columns.

Guidelines:
- Keep descriptions concise but informative
- Focus on business purpose, not technical details
- Use confidence scores between 0.5-1.0 based on how clear the purpose is
- Avoid generic descriptions like "stores data" or "contains values"
- For ID columns, mention what entity they identify
- For categorical columns, explain what the categories represent
- For timestamp columns, explain when the event occurred
- Provide a clear table description (1-2 sentences) explaining what this table stores and its purpose
- For each column, provide a clear description of what it stores and a confidence score
"""

        return prompt

    async def document_schema(
        self, tables: Optional[List[str]] = None
    ) -> Dict[str, TableDocumentation]:
        """Main method to document database schema."""
        conn = self._get_connection()

        try:
            # Get tables to document
            if not tables:
                tables = self._get_all_tables(conn)

            # Document each table concurrently
            semaphore = asyncio.Semaphore(self.config.concurrent_requests)
            tasks = []

            for table_name in tables:
                task = self._document_single_table(semaphore, conn, table_name)
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            documentation = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to document table {tables[i]}: {result}")
                else:
                    documentation[tables[i]] = result

            return documentation

        finally:
            conn.close()

    async def _document_single_table(
        self, semaphore: asyncio.Semaphore, conn, table_name: str
    ) -> TableDocumentation:
        """Document a single table with rate limiting."""
        async with semaphore:
            # Rate limiting
            if self.config.rate_limit_delay > 0:
                await asyncio.sleep(self.config.rate_limit_delay)

            # Get existing comments first
            existing_table_comment, existing_column_comments = (
                self._get_existing_comments(conn, table_name)
            )

            # Analyze table structure
            analysis = self._analyze_table_structure(conn, table_name)

            # Generate new descriptions using LLM
            llm_result = await self._generate_description(table_name, analysis)

            # Create documentation object
            doc = TableDocumentation(
                table_name=table_name,
                existing_table_comment=existing_table_comment,
                existing_column_comments=existing_column_comments,
            )

            # Only use LLM-generated descriptions if no existing comments
            if not existing_table_comment and "table_description" in llm_result:
                doc.table_description = llm_result["table_description"]
                doc.table_confidence = llm_result.get("table_confidence", 0.0)

            if "columns" in llm_result:
                for col_name, col_info in llm_result["columns"].items():
                    # Only use LLM description if no existing comment
                    if col_name not in existing_column_comments:
                        doc.column_descriptions[col_name] = col_info.get(
                            "description", ""
                        )
                        doc.column_confidences[col_name] = col_info.get(
                            "confidence", 0.0
                        )

            return doc

    def _get_all_tables(self, conn) -> List[str]:
        """Get all tables in the database."""
        cursor = conn.cursor()

        if self.db_type == "postgres":
            cursor.execute(
                """
                SELECT table_schema || '.' || table_name as full_table_name
                FROM information_schema.tables 
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            """
            )
        elif self.db_type == "duckdb":
            cursor.execute(
                """
                SELECT table_schema || '.' || table_name as full_table_name
                FROM information_schema.tables 
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN ('information_schema', 'pg_catalog')
            """
            )

        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()

        return tables

    async def apply_documentation(
        self, documentation: Dict[str, TableDocumentation]
    ) -> Dict[str, Dict[str, bool]]:
        """Apply the generated documentation to the database."""
        if self.config.dry_run:
            self.logger.info("DRY RUN: Would apply the following documentation:")
            self._log_documentation_preview(documentation)
            return {}

        conn = self._get_connection()
        results = {}

        try:
            for table_name, doc in documentation.items():
                table_results = {"table": False, "columns": {}}

                # Apply table comment
                if (
                    doc.table_description
                    and doc.table_confidence >= self.config.min_confidence_score
                ):
                    success = self._apply_table_comment(
                        conn, table_name, doc.table_description
                    )
                    table_results["table"] = success

                # Apply column comments
                for col_name, description in doc.column_descriptions.items():
                    confidence = doc.column_confidences.get(col_name, 0.0)
                    if confidence >= self.config.min_confidence_score:
                        success = self._apply_column_comment(
                            conn, table_name, col_name, description
                        )
                        table_results["columns"][col_name] = success

                results[table_name] = table_results

            conn.commit()

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error applying documentation: {e}")
            raise
        finally:
            conn.close()

        return results

    def _apply_table_comment(self, conn, table_name: str, description: str) -> bool:
        """Apply table comment to database using safe identifier validation."""
        try:
            validated_table = _validate_sql_identifier(table_name)

            with self._get_cursor(conn) as cursor:
                if self.db_type == "postgres":
                    # Build properly quoted identifier
                    if "." in validated_table:
                        schema, table = validated_table.split(".", 1)
                        quoted_table = f'"{schema}"."{table}"'
                    else:
                        quoted_table = f'"{validated_table}"'

                    # Use parameterized query for the description value
                    cursor.execute(
                        f"COMMENT ON TABLE {quoted_table} IS %s", (description,)
                    )
                elif self.db_type == "duckdb":
                    # DuckDB comment support is limited, log for now
                    self.logger.info(
                        f"Would set table comment for {table_name}: {description}"
                    )

                return True
        except ValueError as e:
            self.logger.error(f"Invalid table identifier {table_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to set table comment for {table_name}: {e}")
            return False

    def _apply_column_comment(
        self, conn, table_name: str, column_name: str, description: str
    ) -> bool:
        """Apply column comment to database using safe identifier validation."""
        try:
            validated_table = _validate_sql_identifier(table_name)
            validated_column = _validate_sql_identifier(column_name)

            with self._get_cursor(conn) as cursor:
                if self.db_type == "postgres":
                    # Build properly quoted identifiers
                    if "." in validated_table:
                        schema, table = validated_table.split(".", 1)
                        quoted_table = f'"{schema}"."{table}"'
                    else:
                        quoted_table = f'"{validated_table}"'

                    quoted_column = f'"{validated_column}"'

                    # Use parameterized query for the description value
                    cursor.execute(
                        f"COMMENT ON COLUMN {quoted_table}.{quoted_column} IS %s",
                        (description,),
                    )
                elif self.db_type == "duckdb":
                    # DuckDB comment support is limited, log for now
                    self.logger.info(
                        f"Would set column comment for {table_name}.{column_name}: {description}"
                    )

                return True
        except ValueError as e:
            self.logger.error(f"Invalid identifier for {table_name}.{column_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to set column comment for {table_name}.{column_name}: {e}"
            )
            return False

    def _log_documentation_preview(self, documentation: Dict[str, TableDocumentation]):
        """Log documentation preview for dry run."""
        for table_name, doc in documentation.items():
            self.logger.info(f"\nTable: {table_name}")

            if doc.table_description:
                self.logger.info(f"  Table Description: {doc.table_description}")
                self.logger.info(f"  Table Confidence: {doc.table_confidence}")

            if doc.existing_table_comment:
                self.logger.info(
                    f"  Existing Table Comment: {doc.existing_table_comment}"
                )

            for col_name, description in doc.column_descriptions.items():
                confidence = doc.column_confidences.get(col_name, 0.0)
                self.logger.info(
                    f"    Column {col_name}: {description} (confidence: {confidence})"
                )

            for col_name, comment in doc.existing_column_comments.items():
                self.logger.info(f"    Existing comment for {col_name}: {comment}")


# Utility function for integration with existing Defog workflow
async def document_schema_for_defog(
    db_type: str,
    db_creds: Dict,
    tables: Optional[List[str]] = None,
    config: Optional[DocumentationConfig] = None,
) -> Dict[str, TableDocumentation]:
    """
    Utility function to document schema and return documentation.
    Designed for integration with existing Defog generate_db_schema workflow.
    """
    documenter = SchemaDocumenter(db_type, db_creds, config)
    documentation = await documenter.document_schema(tables)

    # Apply documentation to database
    if not config or not config.dry_run:
        await documenter.apply_documentation(documentation)

    return documentation
