# Defog MCP Server

The Model Context Protocol (MCP) server for Defog Python provides tools for SQL queries, code interpretation, web search, YouTube transcription, and PDF data extraction.

## Overview

This MCP server exposes the following tools:
- **run_query**: Execute natural language queries against SQL databases
- **code_interpreter**: Execute Python code for data analysis and calculations
- **web_search**: Search the web for information with citations
- **youtube_video_summary**: Get transcripts/summaries of YouTube videos
- **extract_pdf_data**: Extract structured data from PDF documents

## Setup

### Prerequisites

Ensure you have the defog package installed with the required dependencies:
```bash
pip install defog
```

### Environment Variables

Set the following environment variables based on your needs:

#### LLM Configuration
- `OPENAI_API_KEY`: Required for using OpenAI models (default provider)
- `ANTHROPIC_API_KEY`: Required if using Anthropic models
- `GEMINI_API_KEY`: Required if using Google Gemini models

#### Database Configuration (for SQL queries)

Set `DB_TYPE` to one of the supported database types:
- `postgres`
- `redshift`
- `mysql`
- `bigquery`
- `snowflake`
- `databricks`
- `sqlserver`
- `sqlite`
- `duckdb`

Then set the appropriate credentials based on your database type:

##### PostgreSQL
```bash
export DB_TYPE="postgres"
export DB_HOST="your-host"
export DB_PORT="5432"
export DB_USER="your-username"
export DB_PASSWORD="your-password"
export DB_NAME="your-database"
```

##### MySQL
```bash
export DB_TYPE="mysql"
export DB_HOST="your-host"
export DB_USER="your-username"
export DB_PASSWORD="your-password"
export DB_NAME="your-database"
```

##### BigQuery
```bash
export DB_TYPE="bigquery"
export DB_KEY_PATH="/path/to/service-account-key.json"
```

##### Snowflake
```bash
export DB_TYPE="snowflake"
export DB_USER="your-username"
export DB_PASSWORD="your-password"
export DB_ACCOUNT="your-account"
export DB_WAREHOUSE="your-warehouse"
export DB_NAME="your-database"
```

##### Databricks
```bash
export DB_TYPE="databricks"
export DB_HOST="your-server-hostname"
export DB_PATH="your-http-path"
export DB_TOKEN="your-access-token"
```

##### SQL Server
```bash
export DB_TYPE="sqlserver"
export DB_HOST="your-server"
export DB_USER="your-username"
export DB_PASSWORD="your-password"
export DB_NAME="your-database"
```

##### SQLite
```bash
export DB_TYPE="sqlite"
export DATABASE_PATH="/path/to/your/database.db"
```

##### DuckDB
```bash
export DB_TYPE="duckdb"
export DATABASE_PATH="/path/to/your/database.duckdb"
```

## Running the Server

### Standalone Mode
```bash
# first, export any database related env variables you need to
# also make sure you have the API keys either exported or already in your env
defog serve
```

The server will start on port 33364 by default.

### Using with Claude Code

1. Run `claude mcp add --transport http defog http://127.0.0.1:33364/mcp/` to add the server.
2. That's it! For text to SQL questions, you may have to specify using the text to SQL tool or ask something like "get this from the database". We will work on making this more seamless

### Using with Claude Desktop

1. Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "defog": {
      "command": "python",
      "args": ["/path/to/defog-python/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "DB_TYPE": "postgres",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_USER": "your-username",
        "DB_PASSWORD": "your-password",
        "DB_NAME": "your-database"
      }
    }
  }
}
```

2. Restart Claude Desktop

## Configuration

The server uses the following defaults:
- **Model**: `o4-mini` (OpenAI)
- **Provider**: OpenAI
- **Port**: 33364

These can be modified in the `mcp_server.py` file if needed.

## Troubleshooting

1. **Authentication errors**: Ensure your API keys are correctly set
2. **Database connection errors**: Verify your database credentials and network connectivity
3. **Port conflicts**: Change the port in the server settings if 33364 is already in use

## Security Notes

- Never commit API keys or database credentials to version control
- Use environment variables or secure credential management systems
- Ensure database users have appropriate permissions (read-only recommended for query tools)