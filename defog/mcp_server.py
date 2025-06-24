#!/usr/bin/env python3
"""
MCP Server for Defog Python
Provides tools for SQL queries, code interpretation, web search, and YouTube transcription
"""

import os
from typing import Optional, List
import logging

# we use fastmcp 2.0 provider instead of the fastmcp provided by mcp
# this is because this version makes it easier to change multiple variables, like the port
from fastmcp import FastMCP
from defog.llm.web_search import web_search_tool
from defog.llm.youtube import get_youtube_summary
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.sql import sql_answer_tool
from defog.llm.pdf_data_extractor import extract_pdf_data as extract_pdf_data_tool
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Defog MCP Server")


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
    db_type = os.getenv("DB_TYPE")

    # Parse database credentials
    creds = {}
    # get db creds from env, depending on db_type
    if db_type == "postgres":
        creds = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
    elif db_type == "mysql":
        creds = {
            "host": os.getenv("DB_HOST"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
    elif db_type == "bigquery":
        creds = {
            "json_key_path": os.getenv("DB_KEY_PATH"),
        }
    elif db_type == "snowflake":
        creds = {
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "account": os.getenv("DB_ACCOUNT"),
            "warehouse": os.getenv("DB_WAREHOUSE"),
            "database": os.getenv("DB_NAME"),
        }
    elif db_type == "databricks":
        creds = {
            "server_hostname": os.getenv("DB_HOST"),
            "http_path": os.getenv("DB_PATH"),
            "access_token": os.getenv("DB_TOKEN"),
        }
    elif db_type == "sqlserver":
        creds = {
            "server": os.getenv("DB_HOST"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }
    elif db_type == "sqlite":
        creds = {
            "database": os.getenv("DATABASE_PATH"),
        }
    elif db_type == "duckdb":
        creds = {
            "database": os.getenv("DATABASE_PATH"),
        }

    # Parse table metadata if provided
    result = await sql_answer_tool(
        question=question,
        db_type=db_type,
        db_creds=creds,
        model="o4-mini",
        provider="openai",
    )

    return json.dumps(result, indent=2, default=str)


@mcp.tool(
    description="Execute Python code to analyze data or perform calculations. Returns execution results."
)
async def code_interpreter(question: str, csv_data: Optional[str] = None) -> str:
    """
    Execute Python code to answer questions or analyze data.

    Args:
        question: The question or task to accomplish
        code: Optional Python code to execute (will be generated if not provided)
        csv_data: Optional CSV data as a string to analyze
        model: LLM model to use
        provider: LLM provider (openai, anthropic, gemini, etc.)

    Returns:
        JSON string containing execution results or error message
    """
    try:
        result = await code_interpreter_tool(
            question=question,
            model="gpt-4.1",
            provider="openai",
            csv_string=csv_data or "",
            verbose=False,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error in code_interpreter: {e}")
        return json.dumps({"error": str(e), "status": "error"})


@mcp.tool(
    description="Search the web for information. Returns search results with citations."
)
async def web_search(
    query: str,
) -> str:
    """
    Search the web for information about a topic.

    Args:
        query: Search query or question
        model: LLM model to use
        provider: LLM provider (openai, anthropic, gemini, etc.)
        max_tokens: Maximum tokens for the response

    Returns:
        JSON string containing search results and citations
    """
    try:
        # Perform web search
        result = await web_search_tool(
            question=query,
            model="o4-mini",
            provider="openai",
            verbose=False,
        )

        return json.dumps({"result": result, "status": "success"}, indent=2)

    except Exception as e:
        logger.error(f"Error in web_search: {e}")
        return json.dumps({"error": str(e), "status": "error"})


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


def run_server():
    """Run the MCP server."""
    logger.info("Starting MCP Browser Explorer (Simplified)")
    # List all registered tools
    try:
        import asyncio

        tools = asyncio.run(mcp.list_tools())
        tool_names = [tool.name for tool in tools if tool.name.startswith("browser_")]
        logger.info(
            f"Registered {len(tool_names)} browser tools: {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}"
        )
    except Exception as e:
        logger.warning(f"Could not list tools: {e}")

    mcp.run(transport="streamable-http", port=33364)


if __name__ == "__main__":
    run_server()
