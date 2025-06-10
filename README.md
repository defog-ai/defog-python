# Defog Python

[![tests](https://github.com/defog-ai/defog-python/actions/workflows/main.yml/badge.svg)](https://github.com/defog-ai/defog-python/actions/workflows/main.yml)

# TLDR

This library used to be an SDK for accessing Defog's cloud hosted text-to-SQL service. It has since transformed into a comprehensive toolkit for:

1. **Cross-provider LLM operations** - Unified interface for OpenAI, Anthropic, Gemini, and Together AI
2. **Advanced AI tools** - Code interpreter, web search, YouTube transcription, document citations
3. **Database operations** - Schema generation, query execution, and connection management
4. **Agent orchestration** - Hierarchical task delegation and multi-agent coordination
5. **Memory management** - Automatic conversation compactification for long-context scenarios
6. **Local development server** - FastAPI server for integration and testing

If you are looking for text-to-SQL or deep-research like capabilities, check out [introspect](https://github.com/defog-ai/introspect), our open-source, MIT licensed repo that uses this library as a dependency.

# Installation

```bash
pip install defog
```

# Using this library

## LLM Utilities (`defog.llm`)

The `defog.llm` module provides cross-provider LLM functionality with support for function calling, structured output, and specialized tools.

**Note:** As of the latest version, all LLM functions are async-only. Synchronous methods have been removed to improve performance and consistency.

### Core Chat Functions

```python
from defog.llm.utils import chat_async, chat_async_legacy, LLMResponse
from defog.llm.llm_providers import LLMProvider

# Unified async interface with explicit provider specification
response: LLMResponse = await chat_async(
    provider=LLMProvider.OPENAI,  # or "openai", LLMProvider.ANTHROPIC, etc.
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    max_completion_tokens=1000,
    temperature=0.0
)

print(response.content)  # Response text
print(f"Cost: ${response.cost_in_cents/100:.4f}")

# Alternative: Legacy model-to-provider inference

response = await chat_async_legacy(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Provider-Specific Examples

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

# OpenAI with function calling
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[my_function],  # Optional function calling
    tool_choice="auto"
)

# Anthropic with structured output
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
    response_format=MyPydanticModel  # Structured output
)

# Gemini
response = await chat_async(
    provider=LLMProvider.GEMINI,
    model="gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Code Interpreter Tool

Execute Python code in sandboxed environments across providers:

```python
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.llm_providers import LLMProvider

result = await code_interpreter_tool(
    question="Analyze this CSV data and create a visualization",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string="name,age\nAlice,25\nBob,30",
    instructions="You are a data analyst. Create clear visualizations."
)

print(result["code"])    # Generated Python code
print(result["output"])  # Execution results
```

### Web Search Tool

Search the web for current information:

```python
from defog.llm.web_search import web_search_tool
from defog.llm.llm_providers import LLMProvider

result = await web_search_tool(
    question="What are the latest developments in AI?",
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    max_tokens=2048
)

print(result["search_results"])   # Search results text
print(result["websites_cited"])   # Source citations
```

### Function Calling

Define tools for LLMs to call:

```python
from pydantic import BaseModel
from defog.llm.utils import chat_async

class WeatherInput(BaseModel):
    location: str
    units: str = "celsius"

def get_weather(input: WeatherInput) -> str:
    """Get current weather for a location"""
    return f"Weather in {input.location}: 22°{input.units[0].upper()}, sunny"

response = await chat_async(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
    tool_choice="auto"
)
```

### Memory Compactification

Automatically manage long conversations by intelligently summarizing older messages while preserving context:

```python
from defog.llm import chat_async_with_memory, create_memory_manager, MemoryConfig

# Create a memory manager with custom settings
memory_manager = create_memory_manager(
    token_threshold=50000,      # Compactify when reaching 50k tokens
    preserve_last_n_messages=10, # Keep last 10 messages intact
    summary_max_tokens=2000,    # Max tokens for summary
    enabled=True
)

# System messages are automatically preserved across compactifications
response1 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "Tell me about Python"}
    ],
    memory_manager=memory_manager
)

# Continue the conversation - memory is automatically managed
response2 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "What about its use in data science?"}],
    memory_manager=memory_manager
)

# The system message is preserved even after compactification!
# Check current conversation state:
print(f"Total messages: {len(memory_manager.get_current_messages())}")
print(f"Compactifications: {memory_manager.get_stats()['compactification_count']}")

# Or use memory configuration without explicit manager
response = await chat_async_with_memory(
    provider="anthropic",
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
    memory_config=MemoryConfig(
        enabled=True,
        token_threshold=100000,  # 100k tokens before compactification
        preserve_last_n_messages=10,
        summary_max_tokens=4000
    )
)
```

Key features:
- **System message preservation**: System messages are always kept intact, never summarized
- **Automatic summarization**: When token count exceeds threshold, older messages are intelligently summarized
- **Context preservation**: Recent messages are kept intact for continuity
- **Provider agnostic**: Works with all supported LLM providers
- **Token counting**: Uses tiktoken for accurate OpenAI token counts, with intelligent fallbacks for other providers
- **Flexible configuration**: Customize thresholds, preservation rules, and summary sizes

How it works:
1. As conversation grows, token count is tracked
2. When threshold is exceeded, older messages (except system messages) are summarized
3. Summary is added as a user message with `[Previous conversation summary]` prefix
4. Recent messages + system messages are preserved for context
5. Process repeats as needed for very long conversations

### YouTube Transcript Tool

Generate detailed, diarized transcripts from YouTube videos:

```python
from defog.llm.youtube import get_youtube_summary

# Get transcript with timestamps and speaker identification
transcript = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    model="gemini-2.5-pro-preview-05-06",
    verbose=True,
    system_instructions=[
        "Provide detailed transcript with timestamps (HH:MM:SS)",
        "Include speaker names if available",
        "Skip filler words and small talk"
    ]
)

print(f"Transcript: {transcript}")
```

### Citations Tool

Generate well-cited answers from document collections:

```python
from defog.llm.citations import citations_tool
from defog.llm.llm_providers import LLMProvider

# Prepare documents
documents = [
    {"document_name": "research_paper.pdf", "document_content": "..."},
    {"document_name": "article.txt", "document_content": "..."}
]

# Get cited answer
result = await citations_tool(
    question="What are the main findings?",
    instructions="Provide detailed analysis with citations",
    documents=documents,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    max_tokens=16000
)
```

### Agent Orchestration

Coordinate multiple AI agents for complex tasks:

```python
from defog.llm.orchestrator import (
    Orchestrator, SubAgentTask, ExecutionMode
)

# Create orchestrator
orchestrator = Orchestrator(
    model="claude-3-5-sonnet",
    provider="anthropic"
)

# Define coordinated tasks
tasks = [
    SubAgentTask(
        agent_id="researcher",
        task_description="Research the topic thoroughly",
        execution_mode=ExecutionMode.PARALLEL
    ),
    SubAgentTask(
        agent_id="writer",
        task_description="Write comprehensive report",
        dependencies=["researcher"]
    )
]

# Execute coordinated workflow
results = await orchestrator.execute_tasks(tasks)
```

### MCP (Model Context Protocol) Support

Connect to MCP servers for extended tool capabilities:

```python
from defog.llm.utils_mcp import initialize_mcp_client

# Initialize with config file
mcp_client = await initialize_mcp_client(
    config="path/to/mcp_config.json",
    model="claude-3-5-sonnet"
)

# Process queries with MCP tools
response, tool_outputs = await mcp_client.mcp_chat(
    "Use the calculator tool to compute 123 * 456"
)
```

## Database Operations

The library provides comprehensive database management capabilities for schema generation and query execution.

### Initialization

```python
from defog import Defog, AsyncDefog

# Synchronous version
defog = Defog(
    api_key="your_api_key",
    db_type="postgres",  # postgres, mysql, bigquery, snowflake, etc.
    db_creds={
        "host": "localhost",
        "port": 5432,
        "database": "mydb",
        "user": "username",
        "password": "password"
    }
)

# Asynchronous version
async_defog = AsyncDefog(
    api_key="your_api_key",
    db_type="postgres",
    db_creds={...}
)
```

### Supported Database Types

- **PostgreSQL** (`postgres`)
- **Amazon Redshift** (`redshift`) 
- **MySQL** (`mysql`)
- **Google BigQuery** (`bigquery`)
- **Snowflake** (`snowflake`)
- **Databricks** (`databricks`)
- **SQL Server** (`sqlserver`)

### Schema Generation

```python
# Generate and upload database schema
schema = defog.generate_db_schema(
    tables=["users", "orders", "products"],
    scan=True,        # Scan table contents for better understanding
    upload=True,      # Upload to Defog service
    return_format="csv"  # csv, json, or markdown
)

print(schema)
```

### Query Execution

```python
# Natural language to SQL query execution
result = defog.run_query(
    question="What are the top 10 customers by total order value?",
    hard_filters="WHERE order_date >= '2024-01-01'",  # Optional SQL filters
    glossary="Customer refers to registered users",   # Domain-specific terms
    retries=3,
    use_golden_queries=True  # Use validated query patterns
)

print(f"SQL: {result['query_generated']}")
print(f"Results: {result['data']}")
```

## Command Line Interface

Defog provides a CLI for easy setup and management:

```bash
# Initialize credentials and database connection
defog init

# Generate schema for specific tables
defog gen table1 table2

# Update schema from CSV file
defog update schema.csv

# Interactive query interface
defog query

# Check API quota
defog quota

# View documentation
defog docs

# Start local development server
defog serve
```

## Development Server

Run a local FastAPI server for integration and testing:

```bash
defog serve
```

Endpoints:
- `POST /generate_query` - Generate SQL from natural language
- `POST /integration/get_tables_db_creds` - Get database table information
- `GET /` - Health check

```python
# Example API usage
import requests

response = requests.post(
    "http://localhost:8000/generate_query",
    json={
        "question": "Show me sales by month",
        "previous_context": []
    }
)

result = response.json()
print(result["query_generated"])
```

# Testing
For developers who want to test or add tests for this client, you can run:
```
pytest tests
```
Note that we will transfer the existing .defog/connection.json file over to /tmp (if at all), and transfer the original file back once the tests are done to avoid messing with the original config.
If submitting a PR, please use the `black` linter to lint your code. You can add it as a git hook to your repo by running the command below:
```bash
echo -e '#!/bin/sh\n#\n# Run linter before commit\nblack $(git rev-parse --show-toplevel)' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```
