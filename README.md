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
