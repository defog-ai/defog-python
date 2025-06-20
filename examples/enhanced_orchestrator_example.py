"""Example demonstrating the enhanced orchestrator with shared context and thinking agents.

This example shows:
1. ThinkingAgent with extended reasoning capabilities
2. Cross-agent memory sharing through SharedContextStore
3. Alternative path exploration for complex tasks
4. Real tools: web search, code execution, SQL queries, and file processing
5. How agents collaborate and share insights through the filesystem
"""
# ruff: noqa: F821

import asyncio
import logging
import os
from typing import Dict, Any
from pydantic import BaseModel, Field

from defog.llm import (
    EnhancedAgentOrchestrator,
    ThinkingAgent,
    ExplorationStrategy,
    ArtifactType,
    EnhancedOrchestratorConfig,
    SharedContextConfig,
    ExplorationConfig,
    ThinkingConfig,
    EnhancedMemoryConfig,
)
from defog.llm.web_search import web_search_tool
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.sql import sql_answer_tool
from defog.llm.llm_providers import LLMProvider

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
    code: str = Field(description="Python code to execute")
    data: str = Field(default="", description="Optional CSV data for analysis")


async def execute_python(input: CodeExecutionInput) -> Dict[str, Any]:
    """Execute Python code in a sandboxed environment with optional data."""
    result = await code_interpreter_tool(
        question=f"Execute this code: {input.code}",
        model="gpt-4.1",
        provider="openai",
        csv_string=input.data,
        instructions="Execute the provided code and return the output",
        verbose=False,
    )
    return {
        "output": result.get("output", ""),
        "code_generated": result.get("code", ""),
    }


class DataAnalysisInput(BaseModel):
    data: str = Field(description="Data to analyze as a valid JSON string")
    analysis_type: str = Field(description="Type of analysis to perform")


async def analyze_data(input: DataAnalysisInput) -> Dict[str, Any]:
    """Analyze data using statistical or visualization methods."""
    code = f"""
import json
import statistics
import matplotlib.pyplot as plt
import pandas as pd

# Parse the data
data = '''{input.data}'''
analysis_type = '{input.analysis_type}'

# Perform analysis based on type
if analysis_type == 'statistical':
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
elif analysis_type == 'trend':
    try:
        df = pd.DataFrame(json.loads(data))
        # Simple trend analysis
        print(f"Data shape: {df.shape}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nDescriptive statistics:\n{df.describe()}")
    except:
        print("Could not perform trend analysis")
else:
    print(f"Analysis type '{analysis_type}' not implemented")
"""

    result = await code_interpreter_tool(
        question="Analyze this data",
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        instructions=code,
        verbose=False,
    )
    return {"analysis": result.get("output", "")}


class SQLQueryInput(BaseModel):
    question: str = Field(description="Natural language question to answer using SQL")
    db_path: str = Field(default="", description="Path to database file (for DuckDB)")


async def sql_query(input: SQLQueryInput) -> Dict[str, Any]:
    """Answer questions using SQL queries on a database."""
    # Use example database or provided path
    db_path = input.db_path or os.path.join(
        os.path.dirname(__file__), "cricket_wc2015.duckdb"
    )
    db_creds = {"database": db_path}

    try:
        result = await sql_answer_tool(
            question=input.question,
            db_type="duckdb",
            db_creds=db_creds,
            model="claude-sonnet-4-20250514",
            provider=LLMProvider.ANTHROPIC,
            temperature=0.0,
            verbose=False,
        )

        if result.get("success"):
            return {
                "success": True,
                "query": result.get("query"),
                "columns": result.get("columns"),
                "results": result.get("results"),
                "error": None,
            }
        else:
            return {
                "success": False,
                "query": result.get("query"),
                "results": None,
                "error": result.get("error"),
            }
    except Exception as e:
        return {"success": False, "query": None, "results": None, "error": str(e)}


async def enhanced_orchestration_example():
    """Demonstrate enhanced orchestrator capabilities with real tools."""

    # Create main orchestrator agent with thinking capabilities
    main_agent = ThinkingAgent(
        agent_id="enhanced_orchestrator",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        system_prompt="""You are an advanced orchestrator with enhanced capabilities:

1. **Thinking Mode**: You can think deeply about problems before acting
2. **Shared Context**: You and your subagents share a filesystem for collaboration
3. **Alternative Exploration**: You can explore multiple solution paths
4. **Cross-Agent Memory**: You can access memories from other agents

When you receive complex tasks:
- First think about the problem using your extended reasoning
- Break it down and create specialized subagents using plan_and_create_subagents
- Encourage agents to share insights through the shared context
- Explore alternative approaches when appropriate
- Learn from successful patterns for future tasks

You must use plan_and_create_subagents for complex multi-part requests.""",
        reasoning_effort="medium",
        enable_thinking_mode=True,
        memory_config={
            "enabled": True,
            "token_threshold": 50000,
            "preserve_last_n_messages": 10,
        },
    )

    # Available tools for subagents
    available_tools = [web_search, execute_python, analyze_data, sql_query]

    # Create configuration for enhanced orchestrator
    config = EnhancedOrchestratorConfig(
        shared_context=SharedContextConfig(
            base_path=".enhanced_agent_workspace", cleanup_older_than_days=7
        ),
        exploration=ExplorationConfig(
            max_parallel_explorations=3,
            exploration_timeout=300.0,
            enable_learning=True,
            default_strategy=ExplorationStrategy.ADAPTIVE,
        ),
        memory=EnhancedMemoryConfig(
            max_context_length=128000,
            summarization_threshold=100000,
            summary_model="claude-sonnet-4-20250514",
            reasoning_effort="medium",
        ),
        thinking=ThinkingConfig(
            enable_thinking_mode=True,
            thinking_timeout=120.0,
            thinking_model="claude-sonnet-4-20250514",
            reasoning_effort="medium",
        ),
        enable_thinking_agents=True,
        enable_exploration=True,
        enable_cross_agent_memory=True,
        max_parallel_tasks=3,
        global_timeout=600.0,  # 10 minutes
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
    )

    # Create enhanced orchestrator with configuration
    orchestrator = EnhancedAgentOrchestrator(
        main_agent=main_agent,
        available_tools=available_tools,
        config=config,
        # Legacy parameters that are still needed for base class
        subagent_provider="anthropic",
        subagent_model="claude-sonnet-4-20250514",
        planning_provider="anthropic",
        planning_model="claude-sonnet-4-20250514",
        reasoning_effort="medium",
        max_recursion_depth=2,
        max_total_retries=15,
        max_decomposition_depth=1,
    )

    # Example 1: Complex research with exploration
    #     print("\n=== Example 1: Research with Alternative Approaches ===")
    #     research_messages = [
    #         {
    #             "role": "user",
    #             "content": """I need a comprehensive analysis of renewable energy trends:

    # 1. Research the current state of solar and wind energy adoption globally
    # 2. Analyze the cost trends over the past 5 years using statistical methods
    # 3. Create Python code to visualize the growth projections
    # 4. Compare different forecasting approaches (linear vs exponential growth)

    # Explore multiple analytical approaches and share insights between agents."""
    #         }
    #     ]

    #     logger.info("Processing complex research request...")
    #     response1 = await orchestrator.process(research_messages)
    #     print(f"\nResearch Result:\n{response1.content}")

    # Example 2: Data analysis with SQL and Python
    print("\n\n=== Example 2: Multi-Tool Data Analysis ===")

    # First, let's check if the cricket database exists
    cricket_db_path = os.path.join(os.path.dirname(__file__), "cricket_wc2015.duckdb")

    if os.path.exists(cricket_db_path):
        analysis_messages = [
            {
                "role": "user",
                "content": f"""Analyze Cricket World Cup 2015 data with multiple approaches:

1. Use SQL to find the top 5 batsmen by total runs
2. Calculate the strike rates and averages for these players
3. Use Python to create a statistical comparison of their performance
4. Explore different ways to visualize batting performance trends

Database path: {cricket_db_path}

Have agents share their findings and build on each other's analysis.""",
            }
        ]
    else:
        # Fallback example if database doesn't exist
        analysis_messages = [
            {
                "role": "user",
                "content": """Analyze this sample dataset using multiple approaches:

Data: [45, 52, 38, 64, 42, 48, 55, 67, 71, 39, 58, 61]

1. Perform statistical analysis (mean, median, std dev)
2. Identify outliers using different methods
3. Create Python code for trend analysis
4. Compare different visualization approaches

Explore at least 2 different analytical methods.""",
            }
        ]

    logger.info("Processing data analysis request...")
    response2 = await orchestrator.process(analysis_messages)
    print(f"\nAnalysis Result:\n{response2.content}")

    # Example 3: Cross-agent collaboration and learning


#     print("\n\n=== Example 3: Cross-Agent Collaboration ===")
#     collaboration_messages = [
#         {
#             "role": "user",
#             "content": """Based on all previous analyses:

# 1. Combine insights from the renewable energy research and data analysis
# 2. Have agents retrieve and build upon each other's work from shared context
# 3. Create a unified summary that shows how different agents contributed
# 4. Identify patterns that could be reused for similar future tasks

# Show how the shared context enables better collaboration."""
#         }
#     ]

#     logger.info("Processing collaboration request...")
#     response3 = await orchestrator.process(collaboration_messages)
#     print(f"\nCollaborative Result:\n{response3.content}")

#     # Get orchestration insights
#     print("\n\n=== Orchestration Insights ===")
#     insights = await orchestrator.get_orchestration_insights()

#     print(f"\nShared Context Statistics:")
#     if insights['shared_context_stats']:
#         print(f"  Total artifacts: {insights['shared_context_stats'].get('total_artifacts', 0)}")
#         print(f"  Artifact types: {insights['shared_context_stats'].get('artifact_types', {})}")

#     print(f"\nExploration Patterns:")
#     print(f"  Successful patterns learned: {insights['exploration_patterns'].get('successful_patterns', 0)}")

#     print(f"\nCross-Agent Collaborations:")
#     for collab in insights['cross_agent_collaborations']:
#         print(f"  Agent {collab['agent_id']}: {collab['collaborations']} collaborations")

#     # Demonstrate shared context details
#     print("\n\n=== Shared Context Details ===")
#     shared_context = orchestrator.shared_context

#     # List thinking artifacts
#     thinking_artifacts = await shared_context.list_artifacts(
#         pattern="thinking/*",
#         artifact_type=ArtifactType.PLAN
#     )
#     print(f"\nThinking/Planning Artifacts ({len(thinking_artifacts)} total):")
#     for artifact in thinking_artifacts[:5]:  # Show up to 5
#         print(f"  - {artifact.key}")
#         print(f"    Agent: {artifact.agent_id}")
#         print(f"    Created: {artifact.created_at.strftime('%H:%M:%S')}")

#     # List exploration results
#     exploration_artifacts = await shared_context.list_artifacts(
#         pattern="exploration_result/*",
#         artifact_type=ArtifactType.RESULT
#     )
#     print(f"\nExploration Results ({len(exploration_artifacts)} total):")
#     for artifact in exploration_artifacts[:3]:
#         print(f"  - {artifact.key} (v{artifact.version})")

#     # List shared memories
#     memory_artifacts = await shared_context.list_artifacts(
#         pattern="memory/*"
#     )
#     print(f"\nShared Memories ({len(memory_artifacts)} total)")

#     # Show successful patterns if any
#     pattern_artifacts = await shared_context.list_artifacts(
#         pattern="successful_pattern/*"
#     )
#     if pattern_artifacts:
#         print(f"\nSuccessful Patterns Found ({len(pattern_artifacts)}):")
#         for artifact in pattern_artifacts[:3]:
#             print(f"  - {artifact.key}")

#     print("\n" + "="*50)
#     print("Enhanced orchestration example completed!")
#     print(f"Workspace location: {orchestrator.shared_context.base_path}")
#     print("You can explore the artifacts in this directory to see how agents collaborated.")


async def simple_enhanced_example():
    """A simpler example demonstrating key enhanced features."""

    logger.info("Creating enhanced orchestrator with thinking agents...")

    # Create a thinking main agent
    main_agent = ThinkingAgent(
        agent_id="simple_orchestrator",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        system_prompt="""You are an orchestrator that uses thinking mode.
        Break down tasks using plan_and_create_subagents.
        Encourage agents to explore alternatives when appropriate.""",
        enable_thinking_mode=True,
        reasoning_effort="medium",
    )

    # Create simple configuration
    simple_config = EnhancedOrchestratorConfig(
        shared_context=SharedContextConfig(base_path=".simple_enhanced_workspace"),
        enable_thinking_agents=True,
        enable_exploration=True,
        max_parallel_tasks=2,
        global_timeout=180.0,  # 3 minutes
    )

    # Create enhanced orchestrator
    orchestrator = EnhancedAgentOrchestrator(
        main_agent=main_agent,
        available_tools=[web_search, execute_python],
        config=simple_config,
        # Legacy parameters for base class
        planning_provider="openai",
        planning_model="gpt-4.1",
        subagent_provider="openai",
        subagent_model="gpt-4.1",
        max_recursion_depth=1,
    )

    # Simple request that benefits from thinking and exploration
    messages = [
        {
            "role": "user",
            "content": """I need to convert temperatures:
            1. What's 100°F in Celsius?
            2. Find the current temperature in Paris
            3. Create a temperature conversion function
            
            Think about different approaches and share your reasoning.""",
        }
    ]

    logger.info("Processing request with enhanced features...")
    response = await orchestrator.process(messages)
    print(f"\nResult:\n{response.content}")

    # Show thinking artifacts created
    artifacts = await orchestrator.shared_context.list_artifacts(
        artifact_type=ArtifactType.PLAN
    )
    print(f"\nCreated {len(artifacts)} thinking/planning artifacts")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and "simple" in sys.argv[1]:
        # Run simple example
        asyncio.run(simple_enhanced_example())
    else:
        # Run full example
        asyncio.run(enhanced_orchestration_example())
