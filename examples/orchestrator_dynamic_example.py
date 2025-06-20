"""
Example of dynamic agent orchestration where the system automatically creates subagents.

This example demonstrates:
1. Dynamic subagent creation based on task requirements
2. Web search capabilities for real-time information
3. Code execution for data analysis and calculations
4. SQL querying of Cricket World Cup 2015 DuckDB database using sql_answer_tool
5. File processing operations

The Cricket World Cup 2015 database contains ball-by-ball data from all matches,
including batting/bowling statistics, team performance, and match details.
Run setup_cricket_db.py first to create the DuckDB database from CSV files.
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
from defog.llm.utils_orchestrator_viz import (
    generate_orchestrator_flowchart,
    generate_detailed_tool_trace,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define wrapped tools that the orchestrator can distribute to subagents
class WebSearchInput(BaseModel):
    query: str = Field(description="The search query")


async def web_search(input: WebSearchInput) -> Dict[str, Any]:
    """Search the web for information using available search providers."""
    result = await web_search_tool(
        question=input.query, model="gpt-4.1", provider="openai", verbose=False
    )
    return {
        "content": result.get("content", ""),
        "sources": result.get("websites_cited", []),
    }


class CodeExecutionInput(BaseModel):
    question: str = Field(
        description="Question to answer (note: no visualizations or charts can be created, so do not ask for them)"
    )
    data: str = Field(default="", description="Optional CSV data for analysis")


async def execute_python(input: CodeExecutionInput) -> Dict[str, Any]:
    """Get a question answered by via Python code that is executed in a sandboxed environment, with optional data."""
    result = await code_interpreter_tool(
        question=f"Answer the question: `{input.question}`",
        model="gpt-4.1",
        provider="openai",
        csv_string=input.data,
        instructions="Answer the question and return the output",
        verbose=False,
    )
    return {
        "output": result.get("output", ""),
        "code_generated": result.get("code", ""),
    }


class DataAnalysisInput(BaseModel):
    data: str = Field(description="Data to analyze as a valid JSON string")
    analysis_type: str = Field(description="Type of analysis to perform")


class SQLQueryInput(BaseModel):
    question: str = Field(
        description="Natural language question to answer using Cricket World Cup 2015 data. Limits answers to just a few rows to manage context window limits."
    )


async def cricket_sql_query(input: SQLQueryInput) -> Dict[str, Any]:
    """Answer questions about Cricket World Cup 2015 using SQL queries on the ball-by-ball data. Limits answers to just a few rows to manage context window limits."""
    # Database configuration for Cricket World Cup 2015
    db_path = os.path.join(os.path.dirname(__file__), "cricket_wc2015.duckdb")
    db_creds = {"database": db_path}
    print(f"Querying database for: {input.question}")

    try:
        result = await sql_answer_tool(
            question=input.question,
            db_type="duckdb",
            db_creds=db_creds,
            model="o4-mini",
            provider=LLMProvider.OPENAI,
            temperature=0.0,
            verbose=False,
        )

        if result.get("success"):
            response = {
                "success": True,
                "query": result.get("query"),
                "columns": result.get("columns"),
                "results": result.get("results"),
                "error": None,
            }
            # Pass through cost if available
            if result.get("cost_in_cents") is not None:
                response["cost_in_cents"] = result["cost_in_cents"]
            return response
        else:
            response = {
                "success": False,
                "query": result.get("query"),
                "results": None,
                "error": result.get("error"),
            }
            # Pass through cost even on failure if available
            if result.get("cost_in_cents") is not None:
                response["cost_in_cents"] = result["cost_in_cents"]
            return response
    except Exception as e:
        return {"success": False, "query": None, "results": None, "error": str(e)}


async def dynamic_orchestration_example():
    """Example where the orchestrator dynamically creates subagents based on the task."""

    # Create main orchestrator agent with dynamic capabilities
    main_agent = Agent(
        agent_id="dynamic_orchestrator",
        provider="anthropic",
        model="claude-opus-4-20250514",
        system_prompt="""You are an orchestrator that creates specialized agents dynamically.
        
        For each user request:
        1. Call plan_and_create_subagents ONCE to break down the task and create appropriate subagents
        2. After receiving the results from the tool, synthesize all subagent results into a comprehensive, natural language answer
        3. Do NOT call the tool multiple times for the same request
        
        IMPORTANT: You must provide a final synthesized answer after the tool returns results.""",
        memory_config={"token_threshold": 50000, "preserve_last_n_messages": 10},
    )

    # Create orchestrator with available tools
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[
            web_search,
            execute_python,
            cricket_sql_query,
        ],
        subagent_provider="openai",
        subagent_model="gpt-4.1",
        subagent_designer_provider="openai",
        subagent_designer_model="gpt-4.1",
    )

    print("=== Example: Cricket World Cup 2015 Analysis ===")
    messages = [
        {
            "role": "user",
            "content": """I want to analyze the Cricket World Cup 2015 data. Please help me understand which teams were the best batting units, and which individual performances were the best""",
        }
    ]

    response = await orchestrator.process(messages)
    print(f"Response:\n{response.content}\n")

    # Generate and display the ASCII flowchart
    print("\n=== Orchestrator Execution Flowchart ===")
    flowchart = generate_orchestrator_flowchart(response.tool_outputs)
    print(flowchart)

    print("\n=== Detailed Tool Trace ===")
    trace = generate_detailed_tool_trace(response.tool_outputs)
    print(trace)


async def simple_dynamic_example():
    """A simpler example showing dynamic agent creation."""

    logger.info("Creating main agent...")
    main_agent = Agent(
        agent_id="orchestrator",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        system_prompt="""You are an orchestrator that creates specialized agents dynamically.
        
        For each user request:
        1. Call plan_and_create_subagents ONCE to break down the task and create appropriate subagents
        2. After receiving the results from the tool, synthesize all subagent results into a comprehensive, natural language answer
        3. Do NOT call the tool multiple times for the same request
        
        IMPORTANT: You must provide a final synthesized answer after the tool returns results. Do not just return raw tool output or call the tool again.""",
    )

    logger.info("Creating orchestrator...")
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[web_search, execute_python, cricket_sql_query],
        subagent_designer_provider="openai",
        subagent_designer_model="gpt-4.1",
        subagent_provider="openai",
        subagent_model="gpt-4.1",
    )

    logger.info("Processing request...")
    messages = [
        {
            "role": "user",
            "content": "Who is the current MP from Rae Bareli in 2025, how many degrees Celsius is 75 Fahrenheit, and who was the top run scorer in Cricket World Cup 2015?",
        }
    ]

    try:
        response = await orchestrator.process(messages)
        print(f"Result:\n{response.content}")

        # Generate and display the ASCII flowchart
        print("\n=== Orchestrator Execution Flowchart ===")
        flowchart = generate_orchestrator_flowchart(response.tool_outputs)
        print(flowchart)

        print("\n=== Detailed Tool Trace ===")
        trace = generate_detailed_tool_trace(response.tool_outputs)
        print(trace)

        print("\n=== Additional final answer generation cost ===")
        print(f"${response.cost_in_cents / 100:.4f}")
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
