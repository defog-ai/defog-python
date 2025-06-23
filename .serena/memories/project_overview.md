# Defog Python Project Overview

## Purpose
Defog is a comprehensive Python toolkit for AI-powered data operations. It provides:
- Cross-provider LLM operations (unified interface for OpenAI, Anthropic, Gemini, Together AI)
- SQL Agent for converting natural language to SQL with automatic table filtering
- Data extraction from PDFs, images, HTML, text documents
- Advanced AI tools (code interpreter, web search, YouTube transcription, document citations)
- Agent orchestration for hierarchical task delegation and multi-agent coordination
- Memory management for automatic conversation compactification

## Tech Stack
- Python 3.12 (target version)
- Async/await patterns extensively used
- Multiple LLM provider support (OpenAI, Anthropic, Gemini, Together AI, Mistral)
- Database connectors: PostgreSQL, MySQL, Snowflake, BigQuery, Redshift, Databricks, SQL Server, DuckDB
- Key dependencies: httpx, pandas, pydantic, anthropic, openai, google-genai, together, mistralai, tiktoken, beautifulsoup4, mcp

## Project Structure
- `defog/` - Main package directory
  - `defog/llm/` - LLM-related utilities and providers
  - `defog/llm/tools/` - Various AI tools
  - `defog/llm/orchestrator/` - Agent orchestration functionality
- `tests/` - Test files using pytest
- `docs/` - Documentation files
- `examples/` - Example scripts demonstrating usage
- `notebooks/` - Jupyter notebooks