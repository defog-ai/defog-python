# LLM Utilities and Tools

Comprehensive documentation for all LLM-related functionality in the defog library.

## Table of Contents

- [Core Chat Functions](#core-chat-functions)
- [Multimodal Support](#multimodal-support)
- [Function Calling](#function-calling)
- [Structured Output](#structured-output)
- [Memory Management](#memory-management)
- [Code Interpreter](#code-interpreter)
- [Web Search](#web-search)
- [YouTube Transcription](#youtube-transcription)
- [Citations Tool](#citations-tool)
- [MCP Integration](#mcp-integration)
- [Cost Tracking](#cost-tracking)
- [Best Practices](#best-practices)

## Core Chat Functions

The library provides a unified interface for working with multiple LLM providers.

### Basic Usage

```python
from defog.llm.utils import chat_async, chat_async_legacy, LLMResponse
from defog.llm.llm_providers import LLMProvider

# Unified async interface with explicit provider specification
response: LLMResponse = await chat_async(
    provider=LLMProvider.OPENAI,  # or "openai", LLMProvider.ANTHROPIC, etc.
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
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

### Advanced Parameters

```python
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    
    # Core parameters
    max_completion_tokens=2000,
    temperature=0.7,
    
    # Advanced parameters
    reasoning_effort="high",          # For o1 models: low, medium, high
    response_format=MyPydanticModel,  # Structured output
    tools=[...],                      # Function calling
    tool_choice="auto",               # auto, none, required, or specific tool
    
    # Provider-specific options
    top_p=0.9,                        # Nucleus sampling
    frequency_penalty=0.0,            # Reduce repetition
    presence_penalty=0.0,             # Encourage new topics
    
    # Logging and debugging
    verbose=True,                     # Detailed logging
    return_usage=True,                # Include token usage
)
```

### Provider-Specific Examples

```python
# OpenAI
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages,
    tools=[my_function],
    tool_choice="auto"
)

# Anthropic
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=messages,
    response_format=MyPydanticModel
)

# Gemini
response = await chat_async(
    provider=LLMProvider.GEMINI,
    model="gemini-2.0-flash",
    messages=messages
)

# Together AI
response = await chat_async(
    provider=LLMProvider.TOGETHER,
    model="mixtral-8x7b",
    messages=messages
)
```

## Multimodal Support

The library supports image inputs across all major providers with automatic format conversion.

### Image Input Examples

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider
import base64

# Using base64-encoded images
with open("image.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }
]

# Using image URLs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
    }
]

# Works with all providers - automatic format conversion
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,  # or OPENAI, GEMINI
    model="claude-sonnet-4-20250514",
    messages=messages
)
```

### Multiple Images

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two charts"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{chart1_base64}"}
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{chart2_base64}"}
            }
        ]
    }
]
```

## Function Calling

Define tools for LLMs to call with automatic schema generation.

### Basic Function Definition

```python
from pydantic import BaseModel, Field
from defog.llm.utils import chat_async

class WeatherInput(BaseModel):
    location: str = Field(description="City and country")
    units: str = Field(default="celsius", description="Temperature units")

def get_weather(input: WeatherInput) -> str:
    """Get current weather for a location"""
    return f"Weather in {input.location}: 22°{input.units[0].upper()}, sunny"

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
    tool_choice="auto"
)
```

### Advanced Tool Configuration

```python
from typing import Literal, Optional
from defog.llm.utils import create_tool_from_function

class DatabaseQuery(BaseModel):
    query: str = Field(description="SQL query to execute")
    database: Literal["prod", "staging", "dev"] = Field(default="dev")
    timeout: Optional[int] = Field(default=30, description="Query timeout")
    explain: bool = Field(default=False, description="Include query plan")

@create_tool_from_function
async def execute_database_query(params: DatabaseQuery) -> dict:
    """Execute a database query with safety controls"""
    # Implementation here
    return {"results": [...], "execution_time": 0.5}

# Use with advanced options
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages,
    tools=[execute_database_query],
    tool_choice="required",           # Force tool use
    parallel_tool_calls=True,         # Allow parallel execution
    max_tool_iterations=3,            # Limit recursive calls
)
```

## Structured Output

Get validated, structured responses using Pydantic models.

### Basic Structured Output

```python
from pydantic import BaseModel
from typing import List

class Analysis(BaseModel):
    sentiment: str
    key_points: List[str]
    confidence: float

response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Analyze this text..."}],
    response_format=Analysis
)

# Access structured data
analysis = response.parsed  # Type: Analysis
print(f"Sentiment: {analysis.sentiment}")
print(f"Confidence: {analysis.confidence}")
```

### Complex Structured Output

```python
from typing import List, Optional
from datetime import datetime

class Person(BaseModel):
    name: str
    role: str
    email: Optional[str] = None

class MeetingNotes(BaseModel):
    date: datetime
    attendees: List[Person]
    agenda_items: List[str]
    decisions: List[str]
    action_items: List[dict]
    next_meeting: Optional[datetime] = None

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": meeting_transcript}],
    response_format=MeetingNotes
)

notes = response.parsed
print(f"Meeting on {notes.date}")
print(f"Attendees: {', '.join(p.name for p in notes.attendees)}")
```

## Memory Management

Automatically manage long conversations by intelligently summarizing older messages.

### Basic Memory Usage

```python
from defog.llm import chat_async_with_memory, create_memory_manager, MemoryConfig

# Create a memory manager
memory_manager = create_memory_manager(
    token_threshold=50000,      # Compactify when reaching 50k tokens
    preserve_last_n_messages=10, # Keep last 10 messages intact
    summary_max_tokens=2000,    # Max tokens for summary
    enabled=True
)

# System messages are automatically preserved
response1 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "Tell me about Python"}
    ],
    memory_manager=memory_manager
)

# Continue conversation - memory is automatically managed
response2 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "What about its use in data science?"}],
    memory_manager=memory_manager
)

# Check memory stats
stats = memory_manager.get_stats()
print(f"Total messages: {len(memory_manager.get_current_messages())}")
print(f"Compactifications: {stats['compactification_count']}")
```

### Memory Configuration Options

```python
# Use memory configuration without explicit manager
response = await chat_async_with_memory(
    provider="anthropic",
    model="claude-3-5-sonnet",
    messages=messages,
    memory_config=MemoryConfig(
        enabled=True,
        token_threshold=100000,       # 100k tokens before compactification
        preserve_last_n_messages=10,
        summary_max_tokens=4000,
        preserve_system_messages=True, # Always preserve system messages
        preserve_tool_calls=True,      # Keep function calls in memory
        compression_ratio=0.3          # Target 30% compression
    )
)
```

### Advanced Memory Management

```python
from defog.llm.memory import EnhancedMemoryManager

# Enhanced memory with cross-agent support
memory_manager = EnhancedMemoryManager(
    token_threshold=50000,
    preserve_last_n_messages=10,
    summary_max_tokens=2000,
    
    # Advanced options
    enable_cross_agent_memory=True,    # Share memory across agents
    memory_tags=["research", "analysis"], # Tag memories
    preserve_images=False,             # Exclude images from memory
    
    # Custom summarization
    summary_model="gpt-4o-mini",       # Use cheaper model for summaries
    summary_provider="openai",
    summary_instructions="Focus on key decisions and findings"
)

# Agent 1 with memory
response1 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Research topic X"}],
    memory_manager=memory_manager,
    agent_id="researcher"
)

# Agent 2 can access Agent 1's memory
response2 = await chat_async_with_memory(
    provider="anthropic",
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Summarize the research findings"}],
    memory_manager=memory_manager,
    agent_id="summarizer",
    include_cross_agent_memory=True
)
```

## Code Interpreter

Execute Python code in sandboxed environments with AI assistance.

### Basic Code Execution

```python
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.llm_providers import LLMProvider

result = await code_interpreter_tool(
    question="Analyze this CSV data and create a visualization",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string="name,age,score\nAlice,25,95\nBob,30,87\nCarol,28,92",
    instructions="Create a bar chart showing scores by name"
)

print(result["code"])    # Generated Python code
print(result["output"])  # Execution results
# Images are returned as base64 if generated
```

### Advanced Code Interpreter Options

```python
result = await code_interpreter_tool(
    question="Perform statistical analysis",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string=data,
    
    # Advanced options
    allowed_libraries=["pandas", "matplotlib", "seaborn", "numpy", "scipy"],
    memory_limit_mb=512,              # Memory limit
    timeout_seconds=30,               # Execution timeout
    
    # Custom environment setup
    pre_code="""
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    """,
    
    # Output preferences
    return_images=True,               # Return generated images
    image_format="png",               # png, svg, jpg
    figure_dpi=150,                   # Image quality
    
    # Security
    allow_file_access=False,          # Restrict file system access
    allow_network_access=False        # Restrict network access
)

# Access results
if result.get("images"):
    for idx, img_base64 in enumerate(result["images"]):
        with open(f"output_{idx}.png", "wb") as f:
            f.write(base64.b64decode(img_base64))
```

## Web Search

Search the web for current information with AI-powered analysis.

### Basic Web Search

```python
from defog.llm.web_search import web_search_tool
from defog.llm.llm_providers import LLMProvider

result = await web_search_tool(
    question="What are the latest developments in AI?",
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    max_tokens=2048
)

print(result["search_results"])   # Analyzed search results
print(result["websites_cited"])   # Source citations
```

### Advanced Search Configuration

```python
result = await web_search_tool(
    question="Latest quantum computing breakthroughs",
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    
    # Advanced options
    search_depth="comprehensive",      # quick, standard, comprehensive
    max_results=20,                   # Number of search results
    include_snippets=True,            # Include search snippets
    follow_links=True,                # Fetch full page content
    
    # Domain filtering
    preferred_domains=["arxiv.org", "nature.com", "science.org"],
    blocked_domains=["blogspot.com"],
    
    # Content preferences
    prefer_recent=True,               # Prioritize recent content
    date_range="6m",                  # Last 6 months only
    content_type="academic",          # academic, news, general
    
    # Output formatting
    include_timestamps=True,
    group_by_domain=True,
    summarize_per_domain=True
)
```

## YouTube Transcription

Generate detailed summaries and transcripts from YouTube videos.

### Basic Usage

```python
from defog.llm.youtube import get_youtube_summary

# Get basic summary
summary = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    model="gemini-2.5-pro"
)

print(summary)
```

### Detailed Transcription

```python
# Get detailed transcript with timestamps
transcript = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=...",
    model="gemini-2.5-pro",
    verbose=True,
    
    # Detailed instructions
    system_instructions=[
        "Provide detailed transcript with timestamps (HH:MM:SS)",
        "Include speaker names if available",
        "Separate content into logical sections",
        "Include [MUSIC] and [APPLAUSE] markers",
        "Extract any text shown on screen",
        "Note visual demonstrations with [DEMO] tags",
        "Highlight key quotes with quotation marks"
    ],
    
    # Advanced options
    include_auto_chapters=True,       # Use YouTube's chapter markers
    language_preference="en",         # Preferred transcript language
    fallback_to_auto_captions=True,   # Use auto-generated if needed
    max_retries=3                     # Retry on failure
)
```

## Citations Tool

Generate well-cited answers from document collections.

### Basic Citations

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

print(result["response"])         # Answer with citations
print(result["citations_used"])   # List of citations
```

### Advanced Citation Options

```python
result = await citations_tool(
    question="Compare the methodologies",
    instructions="Provide academic-style analysis",
    documents=documents,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    
    # Advanced options
    citation_style="academic",        # academic, inline, footnote
    min_citations_per_claim=2,        # Require multiple sources
    include_confidence_scores=True,   # Rate citation confidence
    extract_quotes=True,              # Include exact quotes
    max_quote_length=200,             # Limit quote size
    
    # Chunk handling
    chunk_size=2000,                  # Tokens per chunk
    chunk_overlap=200,                # Overlap between chunks
    relevance_threshold=0.7           # Minimum relevance score
)

# Access detailed citation information
for citation in result["citation_details"]:
    print(f"Source: {citation['source']}")
    print(f"Quote: {citation['quote']}")
    print(f"Confidence: {citation['confidence']}")
```

## MCP Integration

Connect to Model Context Protocol servers for extended tool capabilities.

### Basic MCP Setup

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

### Advanced MCP Configuration

```python
from defog.llm.utils_mcp import initialize_mcp_client, MCPConfig

# Advanced configuration
mcp_config = MCPConfig(
    config_path="mcp_config.json",
    
    # Connection options
    connection_timeout=30,
    keep_alive=True,
    auto_reconnect=True,
    
    # Tool discovery
    discover_tools_on_init=True,
    tool_refresh_interval=300,        # Refresh every 5 minutes
    
    # Security
    allowed_servers=["server1", "server2"],
    require_authentication=True,
    
    # Performance
    connection_pool_size=5,
    max_concurrent_requests=10
)

mcp_client = await initialize_mcp_client(
    config=mcp_config,
    model="claude-3-5-sonnet",
    provider="anthropic"
)

# Use with automatic tool routing
response, tool_outputs = await mcp_client.mcp_chat(
    "Query the database for user statistics",
    
    # Advanced options
    prefer_local_tools=True,          # Use local tools when available
    tool_selection_strategy="best",   # best, first, all
    include_tool_reasoning=True       # Explain tool selection
)
```

## Cost Tracking

All LLM operations include detailed cost tracking.

### Basic Cost Information

```python
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages
)

print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Cost: ${response.cost_in_cents / 100:.4f}")
```

### Aggregate Cost Tracking

```python
from defog.llm.cost_tracker import CostTracker

# Initialize cost tracker
tracker = CostTracker()

# Track multiple operations
async def process_documents(docs):
    for doc in docs:
        response = await chat_async(...)
        tracker.add_cost(
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_cents=response.cost_in_cents
        )

# Get cost summary
summary = tracker.get_summary()
print(f"Total cost: ${summary['total_cost_cents'] / 100:.2f}")
print(f"By provider: {summary['by_provider']}")
print(f"By model: {summary['by_model']}")
```

## Best Practices

### 1. Provider Selection

- **OpenAI**: Best for function calling, structured output, general purpose
- **Anthropic**: Best for long context, complex reasoning, document analysis
- **Gemini**: Best for cost-effectiveness, multimodal tasks, large documents
- **Together**: Best for open-source models, specific use cases

### 2. Error Handling

```python
from defog.llm.utils import chat_async
from defog.llm.exceptions import LLMError, RateLimitError, ContextLengthError

try:
    response = await chat_async(...)
except RateLimitError:
    # Wait and retry
    await asyncio.sleep(60)
    response = await chat_async(...)
except ContextLengthError:
    # Use memory management or chunk content
    response = await chat_async_with_memory(...)
except LLMError as e:
    print(f"LLM error: {e}")
```

### 3. Performance Optimization

- Use appropriate models for tasks (don't use GPT-4 for simple tasks)
- Enable caching for repeated operations (especially PDFs)
- Use structured output for consistent parsing
- Batch operations when possible
- Configure memory management for long conversations

### 4. Cost Optimization

- Use cheaper models for summarization (e.g., gpt-4o-mini)
- Enable input caching for repeated content
- Set max_tokens appropriately
- Use token counting before sending requests
- Monitor costs with tracking utilities

### 5. Security Considerations

- Validate all tool inputs with Pydantic models
- Restrict code interpreter capabilities appropriately
- Use domain filtering for web searches
- Implement rate limiting for production use
- Sanitize any user-provided content

## See Also

- [Data Extraction](data-extraction.md) - PDF, Image, and HTML extraction
- [Database Operations](database-operations.md) - SQL generation and execution
- [Agent Orchestration](agent-orchestration.md) - Multi-agent coordination
- [API Reference](api-reference.md) - Complete API documentation