# TLDR

This library used to be an SDK for accessing Defog's cloud hosted text-to-SQL service. It has since transformed into a comprehensive toolkit for:

1. **Cross-provider LLM operations** - Unified interface for OpenAI, Anthropic, Gemini, and Together AI
2. **Advanced AI tools** - Code interpreter, web search, YouTube transcription, document citations
3. **Database operations** - Schema generation, query execution, and connection management
4. **SQL Agent capabilities** - Natural language to SQL conversion with automatic table filtering for large databases
5. **Agent orchestration** - Hierarchical task delegation and multi-agent coordination
6. **Memory management** - Automatic conversation compactification for long-context scenarios

If you are looking for deep-research like capabilities, check out [introspect](https://github.com/defog-ai/introspect), our open-source, MIT licensed repo that uses this library as a dependency.

# Installation

```bash
pip install defog
```

# Using this library

## LLM Utilities (`defog.llm`)

The `defog.llm` module provides cross-provider LLM functionality with support for function calling, structured output, and specialized tools.

**Note:** As of the latest version, all LLM functions are async-only. Synchronous methods have been removed to improve performance and consistency.

### Multimodal Support (Images)

The library now supports image inputs across all major providers (OpenAI, Anthropic, Gemini). You can include images in your messages using a standardized format:

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

# Using base64-encoded images
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_data}"
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
            {"type": "text", "text": "Describe this image from the URL."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
    }
]

# Works with all providers
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,  # or OPENAI, GEMINI
    model="claude-sonnet-4-20250514",
    messages=messages
)
```

The library automatically handles format conversion between providers, so you can use the same message format regardless of which LLM you're using.

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

### YouTube Summary Tool

Generate detailed summaries from YouTube videos:

```python
from defog.llm.youtube import get_youtube_summary

# Get transcript with timestamps and speaker identification
transcript = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    model="gemini-2.5-pro",
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

### PDF Analysis Tool

Analyze PDFs from URLs with Claude's advanced capabilities, including input caching and smart chunking:

```python
from defog.llm.pdf_processor import analyze_pdf, PDFAnalysisInput
from pydantic import BaseModel, Field

# Define structured response format
class DocumentSummary(BaseModel):
    title: str = Field(description="Document title")
    main_topics: list[str] = Field(description="List of main topics covered")
    key_findings: list[str] = Field(description="Key findings or conclusions")

# Analyze PDF with structured output
pdf_input = PDFAnalysisInput(
    url="https://arxiv.org/pdf/2301.07041.pdf",
    task="Summarize this research paper, focusing on the main contributions and findings.",
    response_format=DocumentSummary  # Optional: structured Pydantic response
)

result = await analyze_pdf(pdf_input)

if result["success"]:
    print(f"Analysis: {result['result']}")
    print(f"Cost: ${result['metadata']['total_cost_in_cents'] / 100:.4f}")
    print(f"Cached tokens: {result['metadata']['cached_tokens']}")
else:
    print(f"Error: {result['error']}")
```

Key features:
- **Anthropic Input Caching** - 5-minute cache for repeated analysis, dramatically reducing costs
- **Smart Chunking** - Automatically splits PDFs >80 pages or >24MB for optimal processing
- **Structured Output** - Support for Pydantic models for consistent response formatting
- **Concurrent Processing** - Multiple chunks processed in parallel for faster results
- **Cost Tracking** - Detailed token usage and cost information

The tool handles large documents intelligently:
- PDFs are downloaded and processed locally
- Large files are split into manageable chunks automatically
- Each chunk is processed with proper context awareness
- Results are combined into coherent analysis

### SQL Agent Tools

Convert natural language questions to SQL queries and execute them on local databases:

```python
from defog.llm.sql import sql_answer_tool, identify_relevant_tables_tool
from defog.llm.llm_providers import LLMProvider

# Database connection credentials
db_creds = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "user": "postgres",
    "password": "password"
}

# Ask questions in natural language
result = await sql_answer_tool(
    question="What are the top 10 customers by total sales?",
    db_type="postgres",
    db_creds=db_creds,
    model="claude-sonnet-4-20250514",
    provider=LLMProvider.ANTHROPIC,
    glossary="Total Sales: Sum of all order amounts for a customer",  # Optional
    previous_context=[],  # Optional conversation history
)

if result["success"]:
    print(f"SQL Query: {result['query']}")
    print(f"Results: {result['results']}")
else:
    print(f"Error: {result['error']}")

# For large databases (>1000 columns, >5 tables), tables are automatically filtered
# You can also manually identify relevant tables:
relevance_result = await identify_relevant_tables_tool(
    question="Show me customer orders",
    db_type="postgres", 
    db_creds=db_creds,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    max_tables=10
)
```

Key features:
- **Automatic table filtering** - Intelligently filters relevant tables for large databases
- **Multiple database support** - PostgreSQL, MySQL, BigQuery, Snowflake, Databricks, SQL Server, Redshift, SQLite, DuckDB
- **Business context** - Support for glossaries and domain-specific filters
- **Conversational SQL** - Maintains context for follow-up questions
- **No API dependency** - Query execution happens locally without Defog API calls

For a complete example, see [sql_agent_example.py](sql_agent_example.py).

### Schema Documentation Tool

Automatically generate intelligent documentation for your database schemas using AI:

```python
from defog.llm.schema_documenter import SchemaDocumenter

# Initialize documenter
documenter = SchemaDocumenter(
    model="claude-3-5-sonnet",
    provider="anthropic",
    db_type="postgres",
    db_creds=db_creds
)

# Document specific tables
documented_tables = await documenter.document_tables(
    tables=["customers", "orders", "products"],
    include_existing_comments=True  # Preserve existing database comments
)

# Get documentation with confidence scores
for table_doc in documented_tables:
    print(f"Table: {table_doc['table_name']}")
    print(f"Description: {table_doc['table_description']}")
    for col in table_doc['columns']:
        print(f"  - {col['column_name']}: {col['description']} (confidence: {col['confidence']})")
```

### PDF Data Extraction Tool

Extract structured data from PDFs using intelligent AI analysis:

```python
from defog.llm.pdf_data_extractor import PDFDataExtractor

# Initialize extractor
extractor = PDFDataExtractor(
    model="claude-3-5-sonnet",
    provider="anthropic"
)

# Extract data with automatic schema generation
result = await extractor.extract_data(
    pdf_path="financial_report.pdf",
    instructions="Extract all financial metrics, key performance indicators, and revenue breakdowns"
)

if result["success"]:
    print(f"Extracted data: {result['extracted_data']}")
    print(f"Extraction cost: ${result['total_cost_in_cents'] / 100:.4f}")
else:
    print(f"Error: {result['error']}")

### HTML Data Extraction Tool

Extract structured data from HTML strings using intelligent AI analysis:

```python
from defog.llm.html_data_extractor import HTMLDataExtractor

# Initialize extractor
extractor = HTMLDataExtractor(
    analysis_model="claude-sonnet-4-20250514",
    extraction_model="claude-sonnet-4-20250514",
    analysis_provider="anthropic",
    extraction_provider="anthropic"
)

# HTML content with tables, lists, and structured data
html_content = """
<table>
    <tr><th>Product</th><th>Price</th><th>Stock</th></tr>
    <tr><td>Widget A</td><td>$29.99</td><td>100</td></tr>
    <tr><td>Widget B</td><td>$39.99</td><td>50</td></tr>
</table>
<ul class="features">
    <li>Feature 1: High performance</li>
    <li>Feature 2: Low power consumption</li>
    <li>Feature 3: Compact design</li>
</ul>
"""

# Extract all data with automatic schema generation
result = await extractor.extract_all_data(html_content)

print(f"HTML Type: {result.html_type}")
print(f"Datapoints identified: {result.total_datapoints_identified}")
print(f"Successful extractions: {result.successful_extractions}")
print(f"Total cost: ${result.total_cost_cents / 100:.4f}")

# Access extracted data
for extraction in result.extraction_results:
    if extraction.success:
        print(f"✅ {extraction.datapoint_name}: {extraction.extracted_data}")

# Or use the convenience method for dictionary output
data_dict = await extractor.extract_as_dict(html_content)
print(f"Extracted data: {data_dict['data']}")
```

Key features:
- **HTML Structure Analysis** - Automatically identifies tables, lists, forms, and structured divs
- **Embedded Image Support** - Extracts and analyzes base64-encoded images within HTML
- **Parallel Extraction** - Processes multiple datapoints concurrently for faster results
- **Flexible Schema Generation** - Creates appropriate Pydantic schemas for different HTML structures
- **Cost Tracking** - Detailed token usage and cost information for each extraction
- **Error Handling** - Graceful handling of malformed HTML and extraction failures

The tool handles various HTML structures:
- **HTML Tables** - Extracts headers and data rows in columnar format
- **Structured Lists** - Identifies patterns in `<ul>` and `<ol>` elements
- **Repeated Divs** - Extracts data from repeated layouts (cards, profiles, etc.)
- **Embedded Images** - Analyzes base64 images for visual data extraction
- **Form Fields** - Extracts field names, types, and default values
- **Key-Value Pairs** - Processes definition lists and structured spans

### Database Query Execution

Execute SQL queries directly on local databases without requiring a Defog API key:

```python
from defog.query import execute_query, async_execute_query

# Synchronous execution
colnames, results = execute_query(
    query="SELECT * FROM customers LIMIT 10",
    db_type="postgres",
    db_creds={
        "host": "localhost",
        "port": 5432,
        "database": "mydb",
        "user": "postgres",
        "password": "password"
    }
)

# Asynchronous execution (recommended)
colnames, results = await async_execute_query(
    query="SELECT COUNT(*) FROM orders",
    db_type="mysql",
    db_creds={
        "host": "localhost",
        "port": 3306,
        "database": "ecommerce",
        "user": "root",
        "password": "password"
    }
)
```

Supported database types:
- PostgreSQL (`postgres`)
- MySQL (`mysql`)
- BigQuery (`bigquery`)
- Snowflake (`snowflake`)
- Databricks (`databricks`)
- SQL Server (`sqlserver`)
- Redshift (`redshift`)
- SQLite (`sqlite`)
- DuckDB (`duckdb`)

### Enhanced Agent Orchestration

The library includes an enhanced orchestration system for complex multi-agent workflows:

```python
from defog.llm.enhanced_orchestrator import EnhancedAgentOrchestrator
from defog.llm.thinking_agent import ThinkingAgent

# Create orchestrator with shared context
orchestrator = EnhancedAgentOrchestrator(
    model="claude-3-5-sonnet",
    provider="anthropic",
    enable_shared_context=True,
    enable_thinking_agents=True
)

# Create thinking agents for complex reasoning
research_agent = ThinkingAgent(
    agent_id="researcher",
    model="claude-3-5-sonnet",
    enable_extended_thinking=True
)

# Execute coordinated tasks with shared memory
results = await orchestrator.execute_tasks(
    tasks=[...],
    agents={"researcher": research_agent}
)
```

### Metadata Caching System

Improve performance with intelligent metadata caching:

```python
from defog.metadata_cache import MetadataCache

# Initialize cache with TTL
cache = MetadataCache(ttl_seconds=3600)  # 1 hour TTL

# Cache metadata
cache.set("db_key", metadata)

# Retrieve cached metadata
cached_data = cache.get("db_key")
```

## Local LLM SQL Generation

Generate SQL queries using local LLMs without Defog API:

```python
from defog import Defog

df = Defog()  # No API key needed for local generation

# Use local LLM for SQL generation
result = await df.run_query(
    question="Show me top customers",
    db_type="postgres",
    db_creds=db_creds,
    use_llm_directly=True,  # Use local LLM
    llm_provider="openai",
    llm_model="gpt-4o",
    table_metadata=metadata  # Optional: provide schema directly
)
```

## Health Check Utilities

Validate your configuration and data:

```python
# Check golden query coverage
coverage = df.check_golden_queries_coverage()
print(f"Table coverage: {coverage['table_coverage']}%")
print(f"Column coverage: {coverage['column_coverage']}%")

# Validate metadata
is_valid = df.check_md_valid()

# Check glossary consistency
consistency = df.check_glossary_consistency()
```

## Advanced Configuration

For detailed configuration options and advanced features, see the [documentation](docs/README.md).

## Environment Variables

The library supports various environment variables for configuration:

```bash
# API Keys
export DEFOG_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export TOGETHER_API_KEY="your-together-key"

# Custom endpoints
export DEFOG_BASE_URL="https://custom-api.example.com"
```

# Testing
For developers who want to test or add tests for this client, you can run:

```
pytest tests
```

If submitting a PR, please use the `black` linter to lint your code. You can add it as a git hook to your repo by running the command below:
```bash
echo -e '#!/bin/sh\n#\n# Run linter before commit\nblack $(git rev-parse --show-toplevel)' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```
