# LLM Utilities and Tools

This document covers all LLM-related functionality, including undocumented features and advanced usage patterns.

## Core Chat Functions

### Enhanced Chat Parameters

The `chat_async` function supports additional undocumented parameters:

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    
    # Advanced parameters
    reasoning_effort="high",          # For o1 models: low, medium, high
    response_format=MyPydanticModel,  # Structured output
    tools=[...],                      # Function calling
    tool_choice="auto",               # auto, none, or specific tool
    
    # Provider-specific options
    top_p=0.9,                        # Nucleus sampling
    frequency_penalty=0.0,            # Reduce repetition
    presence_penalty=0.0,             # Encourage new topics
    
    # Logging and debugging
    verbose=True,                     # Detailed logging
    return_usage=True,                # Include token usage
)
```

### Response Format Support

Structured output across providers:

```python
from pydantic import BaseModel
from typing import List

class Analysis(BaseModel):
    sentiment: str
    key_points: List[str]
    confidence: float

# Structured response
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Analyze this text..."}],
    response_format=Analysis  # Automatic JSON mode + validation
)

# Access structured data
analysis = response.parsed  # Type: Analysis
print(f"Sentiment: {analysis.sentiment}")
```

## PDF Analysis Tool

### Advanced PDF Processing

The PDF processor includes undocumented features:

```python
from defog.llm.pdf_processor import analyze_pdf, PDFAnalysisInput
from pydantic import BaseModel, Field

class ResearchPaperAnalysis(BaseModel):
    title: str
    authors: List[str]
    abstract_summary: str
    methodology: str
    key_findings: List[str]
    limitations: List[str]
    future_work: List[str]

# Advanced PDF analysis
pdf_input = PDFAnalysisInput(
    url="https://arxiv.org/pdf/2301.07041.pdf",
    task="Extract all research components",
    response_format=ResearchPaperAnalysis,
    
    # Advanced options (undocumented)
    max_pages_per_chunk=40,           # Override default chunking
    enable_caching=True,              # Use Anthropic's caching
    cache_duration_minutes=10,        # Cache for 10 minutes
    parallel_chunks=True,             # Process chunks in parallel
    include_metadata=True             # Include PDF metadata
)

result = await analyze_pdf(pdf_input)

if result["success"]:
    analysis = result["result"]  # Type: ResearchPaperAnalysis
    metadata = result["metadata"]
    print(f"Total pages: {metadata['total_pages']}")
    print(f"Chunks processed: {metadata['chunks_processed']}")
    print(f"Cached tokens: {metadata['cached_tokens']}")
    print(f"Cost saved: ${metadata['cache_savings_cents'] / 100:.4f}")
```

### PDF Data Extractor

Extract structured data from PDFs with automatic schema generation:

```python
from defog.llm.pdf_data_extractor import PDFDataExtractor

extractor = PDFDataExtractor(
    model="claude-3-5-sonnet",
    provider="anthropic",
    enable_caching=True
)

# Automatic schema generation
result = await extractor.extract_data(
    pdf_path="financial_report.pdf",
    instructions="Extract all financial metrics, tables, and KPIs",
    
    # Advanced options
    confidence_threshold=0.8,         # Minimum confidence for extraction
    include_tables=True,              # Extract tabular data
    include_charts=True,              # Attempt chart data extraction
    parallel_extraction=True,         # Use parallel processing
    max_extraction_cost_cents=500    # Cost limit
)

# Manual schema specification
from pydantic import BaseModel

class FinancialMetrics(BaseModel):
    revenue: float
    profit_margin: float
    growth_rate: float
    
manual_result = await extractor.extract_data(
    pdf_path="report.pdf",
    instructions="Extract financial metrics",
    response_format=FinancialMetrics
)
```

## YouTube Transcription Tool

### Advanced YouTube Features

```python
from defog.llm.youtube import get_youtube_summary

# Detailed transcription with advanced options
transcript = await get_youtube_summary(
    video_url="https://youtube.com/watch?v=...",
    model="gemini-2.5-pro",
    verbose=True,
    
    # Advanced system instructions
    system_instructions=[
        "Include timestamps in HH:MM:SS format",
        "Identify and label different speakers",
        "Separate content into logical sections",
        "Include [MUSIC] and [APPLAUSE] markers",
        "Extract any text shown on screen",
        "Note visual demonstrations with [DEMO] tags"
    ],
    
    # Undocumented options
    include_auto_chapters=True,       # Use YouTube's chapter markers
    language_preference="en",         # Preferred transcript language
    fallback_to_auto_captions=True,   # Use auto-generated if needed
    max_retries=3                     # Retry on failure
)
```

## Enhanced Memory Management

### Advanced Memory Configuration

```python
from defog.llm.memory import (
    create_memory_manager,
    MemoryConfig,
    EnhancedMemoryManager
)

# Enhanced memory manager with cross-agent support
memory_manager = EnhancedMemoryManager(
    token_threshold=50000,
    preserve_last_n_messages=10,
    summary_max_tokens=2000,
    
    # Advanced options
    enable_cross_agent_memory=True,   # Share memory across agents
    memory_tags=["research", "analysis"],  # Tag memories
    compression_ratio=0.3,            # Target 30% compression
    preserve_tool_calls=True,         # Keep function calls
    preserve_images=False,            # Exclude images from memory
    
    # Custom summarization
    summary_model="gpt-4o-mini",      # Use cheaper model for summaries
    summary_provider="openai",
    summary_instructions="Focus on key decisions and findings"
)

# Memory statistics
stats = memory_manager.get_stats()
print(f"Total conversations: {stats['total_conversations']}")
print(f"Average compression: {stats['average_compression_ratio']:.2%}")
print(f"Memory saved: {stats['tokens_saved']} tokens")
```

### Cross-Agent Memory Sharing

```python
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

## Citations Tool

### Advanced Citation Features

```python
from defog.llm.citations import citations_tool

# Advanced citation configuration
result = await citations_tool(
    question="What are the environmental impacts?",
    instructions="Provide comprehensive analysis with citations",
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
```

## Web Search Tool

### Advanced Search Configuration

```python
from defog.llm.web_search import web_search_tool

result = await web_search_tool(
    question="Latest AI developments",
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    
    # Advanced options
    search_depth="comprehensive",     # quick, standard, comprehensive
    max_results=20,                   # Number of search results
    include_snippets=True,            # Include search snippets
    follow_links=True,                # Fetch full page content
    
    # Domain filtering
    preferred_domains=["arxiv.org", "nature.com"],
    blocked_domains=["example.com"],
    
    # Content preferences
    prefer_recent=True,               # Prioritize recent content
    date_range="1y",                  # Last year only
    content_type="academic",          # academic, news, general
    
    # Output formatting
    include_timestamps=True,
    group_by_domain=True,
    summarize_per_domain=True
)
```

## Code Interpreter Tool

### Advanced Code Execution

```python
from defog.llm.code_interp import code_interpreter_tool

result = await code_interpreter_tool(
    question="Analyze this data and create visualizations",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string=data,
    
    # Advanced options
    allowed_libraries=["pandas", "matplotlib", "seaborn", "numpy"],
    memory_limit_mb=512,              # Memory limit
    timeout_seconds=30,               # Execution timeout
    
    # Custom environment
    pre_code="""
    import warnings
    warnings.filterwarnings('ignore')
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
```

## Function Calling

### Advanced Tool Definitions

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from defog.llm.utils import create_tool_from_function

class DatabaseQuery(BaseModel):
    query: str = Field(description="SQL query to execute")
    database: Literal["prod", "staging", "dev"] = Field(default="dev")
    timeout: Optional[int] = Field(default=30, description="Query timeout in seconds")
    explain: bool = Field(default=False, description="Include query explanation")

@create_tool_from_function
def execute_database_query(params: DatabaseQuery) -> dict:
    """Execute a database query with safety controls"""
    # Implementation
    return {"results": [...], "execution_time": 0.5}

# Advanced tool configuration
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Show me user statistics"}],
    tools=[execute_database_query],
    tool_choice="required",           # Force tool use
    
    # Tool execution options
    parallel_tool_calls=True,         # Allow parallel execution
    max_tool_iterations=3,            # Limit recursive calls
    tool_timeout=60,                  # Overall timeout
    
    # Error handling
    tool_error_handling="retry",      # retry, fail, continue
    tool_retry_count=2
)
```

## MCP (Model Context Protocol) Integration

### Advanced MCP Usage

```python
from defog.llm.utils_mcp import initialize_mcp_client, MCPConfig

# Advanced MCP configuration
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
    "Calculate the square root of 144 and explain the process",
    
    # Advanced options
    prefer_local_tools=True,          # Use local tools when available
    tool_selection_strategy="best",   # best, first, all
    include_tool_reasoning=True       # Explain tool selection
)
```

## Best Practices

1. **Use structured outputs** with Pydantic models for consistent responses
2. **Enable caching** for PDF analysis to reduce costs
3. **Configure memory management** for long conversations
4. **Set cost limits** when using expensive operations
5. **Use parallel processing** for multi-chunk operations
6. **Implement proper error handling** with retry logic
7. **Monitor token usage** and costs across operations