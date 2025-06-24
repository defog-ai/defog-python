#!/usr/bin/env python3
"""
MCP Server for Defog Python
Provides tools for SQL queries, code interpretation, web search, and YouTube transcription
"""

from typing import List, Dict, Any
import logging

# we use fastmcp 2.0 provider instead of the fastmcp provided by mcp
# this is because this version makes it easier to change multiple variables, like the port
from fastmcp import FastMCP
from defog.llm.youtube import get_youtube_summary
from defog.llm.sql import sql_answer_tool
from defog.llm.pdf_data_extractor import extract_pdf_data as extract_pdf_data_tool
from defog import config
from defog.local_metadata_extractor import extract_metadata_from_db_async
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Defog MCP Server")


def validate_database_credentials() -> Dict[str, Any]:
    """
    Validate database credentials based on DB_TYPE environment variable.
    Returns a dictionary with validation results.
    """
    db_type = config.get("DB_TYPE")

    if not db_type:
        return {"valid": False, "error": "DB_TYPE environment variable not set"}

    # Define required credentials for each database type
    required_creds = {
        "postgres": ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"],
        "mysql": ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"],
        "bigquery": ["DB_KEY_PATH"],
        "snowflake": [
            "DB_USER",
            "DB_PASSWORD",
            "DB_ACCOUNT",
            "DB_WAREHOUSE",
            "DB_NAME",
        ],
        "databricks": ["DB_HOST", "DB_PATH", "DB_TOKEN"],
        "sqlserver": ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"],
        "sqlite": ["DATABASE_PATH"],
        "duckdb": ["DATABASE_PATH"],
    }

    if db_type not in required_creds:
        return {"valid": False, "error": f"Unsupported database type: {db_type}"}

    # Check for missing credentials
    missing_creds = []
    for cred in required_creds[db_type]:
        if not config.get(cred):
            missing_creds.append(cred)

    if missing_creds:
        return {
            "valid": False,
            "error": f"Missing required credentials for {db_type}: {', '.join(missing_creds)}",
        }

    return {
        "valid": True,
        "db_type": db_type,
        "message": f"Successfully validated {db_type} database credentials",
    }


# Only register text_to_sql_tool if database credentials are valid
validation_result = validate_database_credentials()
if validation_result["valid"]:

    @mcp.tool(
        description="Execute a natural language query against a SQL database. Supports multiple database types."
    )
    async def text_to_sql_tool(
        question: str,
    ) -> str:
        """
        Execute a natural language query against a SQL database. This works best for questions involving structured data.

        Args:
            question: Natural language question to query the database

        Returns:
            JSON string containing query results or error message
        """
        db_type = config.get("DB_TYPE")

        # Parse database credentials
        creds = {}
        # get db creds from env, depending on db_type
        if db_type == "postgres":
            creds = {
                "host": config.get("DB_HOST"),
                "port": config.get("DB_PORT"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "mysql":
            creds = {
                "host": config.get("DB_HOST"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "bigquery":
            creds = {
                "json_key_path": config.get("DB_KEY_PATH"),
            }
        elif db_type == "snowflake":
            creds = {
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "account": config.get("DB_ACCOUNT"),
                "warehouse": config.get("DB_WAREHOUSE"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "databricks":
            creds = {
                "server_hostname": config.get("DB_HOST"),
                "http_path": config.get("DB_PATH"),
                "access_token": config.get("DB_TOKEN"),
            }
        elif db_type == "sqlserver":
            creds = {
                "server": config.get("DB_HOST"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "sqlite":
            creds = {
                "database": config.get("DATABASE_PATH"),
            }
        elif db_type == "duckdb":
            creds = {
                "database": config.get("DATABASE_PATH"),
            }
        elif db_type == "redshift":
            creds = {
                "host": config.get("DB_HOST"),
                "port": config.get("DB_PORT"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        # select model/provider based on what API keys are available
        # we prefer to use openai/o4-mini, gemini/gemini-pro-2.5, anthropic/claude-sonnet-4 - in that order
        if config.get("OPENAI_API_KEY"):
            model = "o4-mini"
            provider = "openai"
        elif config.get("GEMINI_API_KEY"):
            model = "gemini-pro-2.5"
            provider = "gemini"
        elif config.get("ANTHROPIC_API_KEY"):
            model = "claude-sonnet-4-20250514"
            provider = "anthropic"
        else:
            raise ValueError("No API keys found")

        # Parse table metadata if provided
        result = await sql_answer_tool(
            question=question,
            db_type=db_type,
            db_creds=creds,
            model=model,
            provider=provider,
        )

        return json.dumps(result, indent=2, default=str)

    @mcp.tool(
        description="List all tables and their schemas in the configured database. Returns table names, column names, and data types."
    )
    async def list_database_schema() -> str:
        """
        List all tables and their schemas in the configured database.

        Returns:
            JSON string containing database schema information
        """
        db_type = config.get("DB_TYPE")

        # Parse database credentials (same as in text_to_sql_tool)
        creds = {}
        if db_type == "postgres":
            creds = {
                "host": config.get("DB_HOST"),
                "port": config.get("DB_PORT"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "mysql":
            creds = {
                "host": config.get("DB_HOST"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "bigquery":
            creds = {
                "json_key_path": config.get("DB_KEY_PATH"),
            }
        elif db_type == "snowflake":
            creds = {
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "account": config.get("DB_ACCOUNT"),
                "warehouse": config.get("DB_WAREHOUSE"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "databricks":
            creds = {
                "server_hostname": config.get("DB_HOST"),
                "http_path": config.get("DB_PATH"),
                "access_token": config.get("DB_TOKEN"),
            }
        elif db_type == "sqlserver":
            creds = {
                "server": config.get("DB_HOST"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        elif db_type == "sqlite":
            creds = {
                "database": config.get("DATABASE_PATH"),
            }
        elif db_type == "duckdb":
            creds = {
                "database": config.get("DATABASE_PATH"),
            }
        elif db_type == "redshift":
            creds = {
                "host": config.get("DB_HOST"),
                "port": config.get("DB_PORT"),
                "user": config.get("DB_USER"),
                "password": config.get("DB_PASSWORD"),
                "database": config.get("DB_NAME"),
            }
        else:
            return json.dumps(
                {"error": f"Unsupported database type: {db_type}"}, indent=2
            )

        try:
            # Extract metadata from database
            schema_result = await extract_metadata_from_db_async(
                db_type=db_type,
                db_creds=creds,
                tables=[],  # Empty list means all tables
            )

            # Format the result for better readability
            formatted_result = {
                "database_type": db_type,
                "tables": {},
            }

            for table_name, table_info in schema_result.items():
                if isinstance(table_info, dict) and "columns" in table_info:
                    # Handle new format with table_description
                    formatted_result["tables"][table_name] = {
                        "description": table_info.get("table_description", ""),
                        "columns": table_info["columns"],
                    }
                else:
                    # Handle old format (list of columns)
                    formatted_result["tables"][table_name] = {
                        "description": "",
                        "columns": table_info,
                    }

            # Add summary statistics
            formatted_result["summary"] = {
                "total_tables": len(formatted_result["tables"]),
                "total_columns": sum(
                    len(table_data["columns"])
                    for table_data in formatted_result["tables"].values()
                ),
            }

            return json.dumps(formatted_result, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error in list_database_schema: {e}")
            return json.dumps({"error": str(e), "status": "error"}, indent=2)


if config.get("GEMINI_API_KEY"):

    @mcp.tool(
        description="Get a detailed transcript/summary of a YouTube video. Returns formatted transcript with timestamps."
    )
    async def youtube_video_summary(
        video_url: str,
        task_description: str,
        system_instructions: List[str] = None,
    ) -> str:
        """
        Get a summary of a YouTube video.

        Args:
            video_url: YouTube video URL
            task_description: Description of the task to accomplish
            system_instructions: System instructions to use for the task

        Returns:
            JSON string containing summary
        """
        try:
            result = await get_youtube_summary(
                video_url=video_url,
                verbose=False,
                task_description=task_description,
                system_instructions=system_instructions,
            )

            return json.dumps({"summary": result}, indent=2)

        except Exception as e:
            logger.error(f"Error in youtube_transcript: {e}")
            return json.dumps({"error": str(e), "status": "error"})
else:
    logger.warning("No API keys found for Gemini, YouTube tool will not be available")


if config.get("ANTHROPIC_API_KEY"):

    @mcp.tool(
        description="Extract structured data from a PDF document. Requires a URL to the PDF."
    )
    async def extract_pdf_data(
        pdf_url: str,
    ) -> str:
        """
        Extract structured data from a PDF document.

        Args:
            pdf_url: URL to the PDF document

        Returns:
            JSON string containing extracted data or error message
        """
        try:
            result = await extract_pdf_data_tool(
                pdf_url=pdf_url,
            )

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error in extract_pdf_data: {e}")
            return json.dumps({"error": str(e), "status": "error"})
else:
    logger.warning(
        "No API keys found for Anthropic, PDF extraction tool will not be available"
    )


def run_server():
    """Run the MCP server."""
    logger.info("Starting Defog Browser")

    # Log database validation status
    if validation_result["valid"]:
        logger.info(f"Database credentials validated: {validation_result['message']}")
    else:
        logger.warning(
            f"Database credentials validation failed: {validation_result['error']}"
        )
        logger.warning("SQL tool will not be available")

    # List all registered tools
    try:
        import asyncio

        tools = asyncio.run(mcp._list_tools())
        tool_names = [tool.name for tool in tools]
        logger.info(f"Registered {len(tool_names)} tools: {', '.join(tool_names)}")
    except Exception as e:
        logger.warning(f"Could not list tools: {e}")

    # disable streamable-http for now, until Claude Desktop supports it
    # mcp.run(transport="streamable-http", port=33364)

    # use normal HTTP (stdio under the hood, AFAIK) for now
    mcp.run()


if __name__ == "__main__":
    run_server()
