"""
Example of dynamic agent orchestration where the system automatically creates subagents.

This example demonstrates:
1. Dynamic subagent creation based on task requirements
2. Web search capabilities for real-time information
3. Code execution for data analysis and calculations
4. SQL querying of Cricket World Cup 2015 database using sql_answer_tool
5. File processing operations

The Cricket World Cup 2015 database contains ball-by-ball data from all matches,
including batting/bowling statistics, team performance, and match details.
Run setup_cricket_db.py first to create the database from CSV files.
"""

import asyncio
import logging
import os
from typing import Dict, Any
from pydantic import BaseModel, Field

from defog.llm.orchestrator import Agent, AgentOrchestrator
from defog.llm.web_search import web_search_tool
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.sql import sql_answer_tool
from defog.llm.llm_providers import LLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define wrapped tools that the orchestrator can distribute to subagents
class WebSearchInput(BaseModel):
    query: str = Field(description="The search query")

async def web_search(input: WebSearchInput) -> Dict[str, Any]:
    """Search the web for information using available search providers."""
    result = await web_search_tool(
        question=input.query,
        model="gpt-4.1",
        provider="openai"
    )
    return {
        "content": result.get("content", ""),
        "sources": result.get("websites_cited", [])
    }


class CodeExecutionInput(BaseModel):
    code: str = Field(description="Python code to execute")
    data: str = Field(default="", description="Optional CSV data for analysis")

async def execute_python(input: CodeExecutionInput) -> Dict[str, Any]:
    """Execute Python code in a sandboxed environment with optional data."""
    result = await code_interpreter_tool(
        question=f"Execute this code: {input.code}",
        model="gpt-4.1",
        provider="openai",
        csv_string=input.data,
        instructions="Execute the provided code and return the output"
    )
    return {
        "output": result.get("output", ""),
        "code_generated": result.get("code", "")
    }


class DataAnalysisInput(BaseModel):
    data: str = Field(description="Data to analyze as a valid JSON string")
    analysis_type: str = Field(description="Type of analysis to perform")

async def analyze_data(input: DataAnalysisInput) -> Dict[str, Any]:
    """Analyze data using statistical or visualization methods."""
    code = f"""
import json
import statistics

# Parse the data
data = '''{input.data}'''
analysis_type = '{input.analysis_type}'

# Perform analysis based on type
if analysis_type == 'statistical':
    # Try to parse as JSON array of numbers
    try:
        numbers = json.loads(data)
        result = {{
            'mean': statistics.mean(numbers),
            'median': statistics.median(numbers),
            'stdev': statistics.stdev(numbers) if len(numbers) > 1 else 0,
            'min': min(numbers),
            'max': max(numbers)
        }}
        print(json.dumps(result, indent=2))
    except:
        print("Could not perform statistical analysis on the data")
else:
    print(f"Analysis type '{analysis_type}' not implemented")
"""
    
    result = await code_interpreter_tool(
        question="Analyze this data",
        model="claude-3-7-sonnet-latest",
        provider="anthropic",
        instructions=code
    )
    return {"analysis": result.get("output", "")}


class FileProcessingInput(BaseModel):
    content: str = Field(description="File content to process")
    operation: str = Field(description="Operation to perform on the file")

async def process_file(input: FileProcessingInput) -> Dict[str, Any]:
    """Process file content with various operations."""
    code = f"""
# Process file content
content = '''{input.content}'''
operation = '{input.operation}'

if operation == 'word_count':
    words = content.split()
    lines = content.split('\\n')
    print(f"Words: {{len(words)}}")
    print(f"Lines: {{len(lines)}}")
    print(f"Characters: {{len(content)}}")
elif operation == 'extract_numbers':
    import re
    numbers = re.findall(r'\\b\\d+\\.?\\d*\\b', content)
    print(f"Found numbers: {{numbers}}")
else:
    print(f"Operation '{operation}' not supported")
"""
    
    result = await code_interpreter_tool(
        question="Process file content",
        model="claude-3-7-sonnet-latest",
        provider="anthropic",
        instructions=code
    )
    return {"result": result.get("output", "")}


class SQLQueryInput(BaseModel):
    question: str = Field(description="Natural language question to answer using Cricket World Cup 2015 data")

async def cricket_sql_query(input: SQLQueryInput) -> Dict[str, Any]:
    """Answer questions about Cricket World Cup 2015 using SQL queries on the ball-by-ball data."""
    # Database configuration for Cricket World Cup 2015
    db_path = os.path.join(os.path.dirname(__file__), "cricket_wc2015.db")
    db_creds = {
        "database": db_path
    }
    
    print(db_creds)

    try:
        result = await sql_answer_tool(
            question=input.question,
            db_type="sqlite",
            db_creds=db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC,
            temperature=0.0
        )

        print(result)
        
        if result.get("success"):
            return {
                "success": True,
                "query": result.get("query"),
                "columns": result.get("columns"),
                "results": result.get("results"),
                "error": None
            }
        else:
            return {
                "success": False,
                "query": result.get("query"),
                "results": None,
                "error": result.get("error")
            }
    except Exception as e:
        return {
            "success": False,
            "query": None,
            "results": None,
            "error": str(e)
        }


async def dynamic_orchestration_example():
    """Example where the orchestrator dynamically creates subagents based on the task."""
    
    # Create main orchestrator agent with dynamic capabilities
    main_agent = Agent(
        agent_id="dynamic_orchestrator",
        provider="openai",
        model="gpt-4.1",
        system_prompt="""You are a dynamic orchestrator that automatically creates specialized subagents.
        
        When you receive a complex task:
        1. Analyze what needs to be done
        2. Use the plan_and_create_subagents tool to dynamically create the right subagents
        3. The tool will automatically create subagents with appropriate tools and execute tasks
        4. Synthesize the results and provide a comprehensive response
        
        You don't need to use delegate_to_subagents - the planning tool handles everything.""",
        memory_config={"token_threshold": 50000, "preserve_last_n_messages": 10}
    )
    
    # Create orchestrator with available tools
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[
            web_search,
            execute_python,
            analyze_data,
            process_file,
            cricket_sql_query
        ],
        subagent_provider="openai",
        subagent_model="gpt-4.1",
        planning_provider="openai",
        planning_model="gpt-4.1"
    )
    
    # Example 1: Research and analysis task
    print("=== Example 1: Research and Analysis ===")
    messages = [{
        "role": "user",
        "content": """I need to understand the current state of quantum computing. 
        Please research recent breakthroughs, analyze the market size and growth projections, 
        and create a simple table comparing the qubit counts of major quantum computers."""
    }]
    
    response = await orchestrator.process(messages)
    print(f"Response:\n{response.content}\n")
    print(f"Cost: ${response.total_cost:.4f}\n")
    
    # Clear memory before next example
    orchestrator.clear_all_memory()
    
    # Example 2: Cricket World Cup 2015 Analysis
    print("=== Example 2: Cricket World Cup 2015 Analysis ===")
    messages = [{
        "role": "user",
        "content": """I want to analyze the Cricket World Cup 2015 data. Please help me understand:
        1. Which player scored the most runs in the tournament?
        2. Who took the most wickets?
        3. Which team had the best batting average?
        4. What was the highest individual score in an innings?
        5. Create a summary comparing the performance of the top 3 teams"""
    }]
    
    response = await orchestrator.process(messages)
    print(f"Response:\n{response.content}\n")
    print(f"Cost: ${response.total_cost:.4f}\n")
    
    # Clear memory before next example
    orchestrator.clear_all_memory()


async def simple_dynamic_example():
    """A simpler example showing dynamic agent creation."""
    
    logger.info("Creating main agent...")
    main_agent = Agent(
        agent_id="orchestrator",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        system_prompt="""You are an orchestrator that creates specialized agents dynamically.
        Use plan_and_create_subagents to break down tasks and create appropriate subagents."""
    )
    
    logger.info("Creating orchestrator...")
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[web_search, execute_python, cricket_sql_query],
        planning_provider="openai",
        planning_model="gpt-4.1",
        subagent_provider="openai",
        subagent_model="gpt-4.1"
    )
    
    logger.info("Processing request...")
    messages = [{
        "role": "user",
        "content": "Who is the current MP from Rae Bareli in 2025, how many degrees Celsius is 75 Fahrenheit, and who was the top run scorer in Cricket World Cup 2015?"
    }]
    
    try:
        response = await orchestrator.process(messages)
        print(f"Result:\n{response.content}")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        print("Running simple dynamic example...")
        asyncio.run(simple_dynamic_example())
    else:
        print("Running full dynamic orchestration examples...")
        asyncio.run(dynamic_orchestration_example())