"""Example of dynamic agent orchestration where the system automatically creates subagents."""

import asyncio
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

from defog.llm.orchestrator import Agent, AgentOrchestrator
from defog.llm.web_search import web_search_tool
from defog.llm.code_interp import code_interpreter_tool

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
            process_file
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
    print(f"Tokens: {response.total_tokens}, Cost: ${response.total_cost:.4f}\n")
    
    # Clear memory before next example
    orchestrator.clear_all_memory()
    
    # Example 2: Data processing pipeline
    print("=== Example 2: Data Processing Pipeline ===")
    messages = [{
        "role": "user",
        "content": """I have sales data: [1200, 1350, 1100, 1450, 1600, 1550, 1700, 1650, 1800, 1750, 1900, 2000].
        These are monthly sales figures for 2024. Please:
        1. Calculate statistical metrics
        2. Identify the trend
        3. Create a forecast for the next 3 months
        4. Search for industry benchmarks to compare against"""
    }]
    
    response = await orchestrator.process(messages)
    print(f"Response:\n{response.content}\n")
    
    # Example 3: Code generation and testing
    print("=== Example 3: Code Generation and Testing ===")
    messages = [{
        "role": "user",
        "content": """Create a Python class for managing a priority queue with the following features:
        1. Add items with priorities
        2. Remove highest priority item
        3. Peek at highest priority without removing
        4. Check if empty
        Then test it thoroughly and search for best practices for priority queue implementations."""
    }]
    
    response = await orchestrator.process(messages)
    print(f"Response:\n{response.content}\n")


async def simple_dynamic_example():
    """A simpler example showing dynamic agent creation."""
    
    logger.info("Creating main agent...")
    main_agent = Agent(
        agent_id="orchestrator",
        provider="anthropic",
        model="claude-opus-4-20250514",
        system_prompt="""You are an orchestrator that creates specialized agents dynamically.
        Use plan_and_create_subagents to break down tasks and create appropriate subagents."""
    )
    
    logger.info("Creating orchestrator...")
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[web_search, execute_python],
        planning_provider="openai",
        planning_model="gpt-4.1",
        subagent_provider="openai",
        subagent_model="gpt-4.1"
    )
    
    logger.info("Processing request...")
    messages = [{
        "role": "user",
        "content": "Who is the current MP from Rae Bareli in 2025 and how many degrees Celsius is 75 Fahrenheit?"
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