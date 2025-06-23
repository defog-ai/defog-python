# Project Entrypoints and Usage

## Library Usage
Defog is primarily a library, not a standalone application. Users import and use it in their Python code.

## Main Entry Points

### 1. LLM Chat Operations
```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

# Basic chat completion
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 2. SQL Agent
```python
from defog.llm.sql import sql_answer_tool
from defog.llm.llm_providers import LLMProvider

# Natural language to SQL
result = await sql_answer_tool(
    question="What are the top customers?",
    db_type="postgres",
    db_creds={...},
    model="claude-sonnet-4-20250514",
    provider=LLMProvider.ANTHROPIC
)
```

### 3. Data Extraction
```python
from defog.llm import extract_pdf_data

# Extract from PDFs
data = await extract_pdf_data(
    pdf_url="https://example.com/report.pdf",
    focus_areas=["revenue", "metrics"]
)
```

### 4. Agent Orchestration
```python
# See agent_orchestrator.py in root directory
```

## Example Scripts
Located in `examples/` directory:
- `sql_agent_example.py` - SQL generation demo
- `pdf_data_extractor_example.py` - PDF extraction
- `html_data_extractor_example.py` - HTML extraction
- `text_data_extractor_example.py` - Text extraction
- `schema_documentation_example.py` - Schema documentation
- `podcast_analyzer.py` - YouTube/podcast analysis
- `image_support_example.py` - Image processing

## Running Examples
```bash
PYTHONPATH=. python examples/sql_agent_example.py
```