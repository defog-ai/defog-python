# Defog Python

[![tests](https://github.com/defog-ai/defog-python/actions/workflows/main.yml/badge.svg)](https://github.com/defog-ai/defog-python/actions/workflows/main.yml)

# TLDR

This library used to be an SDK for accessing Defog's cloud hosted text-to-SQL service. It has since transformed into a general purpose library for:

1. Making convenient, cross-provider LLM calls (including server-hosted tools)
2. Easily extracting information from databases to make them easy to use

If you are looking for text-to-SQL or deep-research like capabilities, check out [introspect](https://github.com/defog-ai/introspect), our open-source, MIT licensed repo that uses this library as a dependency.

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
