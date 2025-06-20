"""
Local SQL generation using LLM providers without external API calls.
"""

from typing import Dict, List, Optional, Any, Union
from defog.llm.utils import chat_async, LLMProvider
from defog.llm.config import LLMConfig


def format_schema_for_prompt(table_metadata: Dict[str, Any]) -> str:
    """
    Format table metadata into a DDL string representation for the LLM prompt.

    Args:
        table_metadata: Dictionary mapping table names to column info.
                       Supports both legacy format (list of columns) and new format (dict with table_description)

    Returns:
        DDL formatted string representation of the schema with comments
    """
    schema_parts = []

    for table_name, table_info in table_metadata.items():
        # Handle both legacy format (list) and new format (dict with table_description)
        if isinstance(table_info, list):
            # Legacy format: table_info is a list of columns
            columns = table_info
            table_description = None
        else:
            # New format: table_info is a dict with table_description and columns
            columns = table_info.get("columns", [])
            table_description = table_info.get("table_description", "")

        # Generate CREATE TABLE statement
        schema_parts.append(f"CREATE TABLE {table_name} (")

        # Add table comment if available
        if table_description:
            schema_parts.append(f"  -- {table_description}")

        # Add columns with inline comments
        for i, col in enumerate(columns):
            col_name = col.get("column_name", "")
            data_type = col.get("data_type", "")
            description = col.get("column_description", "")

            # Add comma for all but the last column
            comma = "," if i < len(columns) - 1 else ""

            # Build column line with optional inline comment
            col_line = f"  {col_name} {data_type}{comma}"
            if description:
                col_line += f" -- {description}"

            schema_parts.append(col_line)

        schema_parts.append(");")

        schema_parts.append("")  # Empty line between tables

    return "\n".join(schema_parts)


def build_sql_generation_prompt(
    question: str,
    table_metadata: Dict[str, List[Dict[str, str]]],
    db_type: str,
    glossary: Optional[str] = None,
    hard_filters: Optional[str] = None,
    previous_context: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build the prompt messages for SQL generation.

    Args:
        question: Natural language question
        table_metadata: Database schema information
        db_type: Type of database (postgres, mysql, etc.)
        glossary: Optional business glossary
        hard_filters: Optional hard filters to apply
        previous_context: Optional previous conversation context

    Returns:
        List of message dictionaries for the LLM
    """
    # Format the schema
    schema_str = format_schema_for_prompt(table_metadata)

    # Build the system prompt
    system_prompt = f"""You are an expert SQL developer. Your task is to convert natural language questions into SQL queries for a {db_type} database.

Database Schema:
{schema_str}

Rules:
1. Generate only valid {db_type} SQL syntax
2. Use only the tables and columns provided in the schema
3. Return ONLY the SQL query without any explanation or markdown formatting
4. Ensure the query is optimized and efficient
5. Handle NULL values appropriately
6. Use appropriate JOINs when querying multiple tables"""

    if glossary:
        system_prompt += f"\n\nBusiness Glossary:\n{glossary}"

    if hard_filters:
        system_prompt += f"\n\nAlways apply these filters:\n{hard_filters}"

    messages = [{"role": "system", "content": system_prompt}]

    # Add previous context if provided
    if previous_context:
        for ctx in previous_context:
            messages.append(ctx)

    # Add the current question
    messages.append({"role": "user", "content": question})

    return messages


async def generate_sql_query_local(
    question: str,
    table_metadata: Dict[str, List[Dict[str, str]]],
    db_type: str,
    provider: Union[LLMProvider, str] = LLMProvider.ANTHROPIC,
    model: str = "claude-sonnet-4-20250514",
    glossary: Optional[str] = None,
    hard_filters: Optional[str] = None,
    previous_context: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """
    Generate SQL query locally using LLM providers.

    Args:
        question: Natural language question
        table_metadata: Database schema information
        db_type: Type of database (postgres, mysql, etc.)
        provider: LLM provider to use
        model: Model name
        glossary: Optional business glossary
        hard_filters: Optional hard filters to apply
        previous_context: Optional previous conversation context
        temperature: LLM temperature setting
        config: Optional LLM configuration

    Returns:
        Dictionary with generated SQL and metadata
    """
    try:
        # Build the prompt
        messages = build_sql_generation_prompt(
            question=question,
            table_metadata=table_metadata,
            db_type=db_type,
            glossary=glossary,
            hard_filters=hard_filters,
            previous_context=previous_context,
        )

        # Generate SQL using the LLM
        response = await chat_async(
            provider=provider,
            model=model,
            messages=messages,
            temperature=temperature,
            config=config,
        )

        # Extract the SQL query
        sql_query = response.content.strip()

        # Clean up the SQL if it contains markdown formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()

        # Update context for future queries
        new_context = previous_context.copy() if previous_context else []
        new_context.append({"role": "user", "content": question})
        new_context.append({"role": "assistant", "content": sql_query})

        return {
            "query_generated": sql_query,
            "ran_successfully": True,
            "error_message": None,
            "query_db": db_type,
            "reason_for_query": "No reason provided",
            "previous_context": new_context,
            "cost_in_cents": response.cost_in_cents,
        }

    except Exception as e:
        return {
            "query_generated": None,
            "ran_successfully": False,
            "error_message": str(e),
            "query_db": db_type,
            "reason_for_query": None,
            "previous_context": previous_context,
        }


def generate_sql_query_local_sync(
    question: str,
    table_metadata: Dict[str, List[Dict[str, str]]],
    db_type: str,
    provider: Union[LLMProvider, str] = LLMProvider.ANTHROPIC,
    model: str = "claude-sonnet-4-20250514",
    glossary: Optional[str] = None,
    hard_filters: Optional[str] = None,
    previous_context: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for generate_sql_query_local.
    """
    import asyncio

    return asyncio.run(
        generate_sql_query_local(
            question=question,
            table_metadata=table_metadata,
            db_type=db_type,
            provider=provider,
            model=model,
            glossary=glossary,
            hard_filters=hard_filters,
            previous_context=previous_context,
            temperature=temperature,
            config=config,
        )
    )
