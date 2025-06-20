"""End-to-end tests for the dynamic agent orchestration system."""

import pytest
import asyncio
from typing import Dict, Any
from pydantic import BaseModel, Field

from defog.llm.orchestrator import Agent, AgentOrchestrator


# Real tool implementations for testing
class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to calculate")


async def calculator_tool(input: CalculatorInput) -> Dict[str, Any]:
    """Simple calculator tool for testing."""
    try:
        # Safe evaluation of basic math expressions
        allowed_names = {
            k: v
            for k, v in __builtins__.items()
            if k in ["abs", "min", "max", "round", "sum"]
        }
        allowed_names.update({"__builtins__": {}})

        result = eval(input.expression, allowed_names)
        # Simulate tool cost (0.1 cents per calculation)
        return {"result": result, "expression": input.expression, "cost_in_cents": 0.1}
    except Exception as e:
        return {"error": str(e), "expression": input.expression, "cost_in_cents": 0.1}


class TextProcessorInput(BaseModel):
    text: str = Field(description="Text to process")
    operation: str = Field(
        description="Operation: 'count_words', 'reverse', 'uppercase'"
    )


async def text_processor_tool(input: TextProcessorInput) -> Dict[str, Any]:
    """Text processing tool for testing."""
    text = input.text
    operation = input.operation

    if operation == "count_words":
        result = len(text.split())
    elif operation == "reverse":
        result = text[::-1]
    elif operation == "uppercase":
        result = text.upper()
    else:
        return {"error": f"Unknown operation: {operation}", "cost_in_cents": 0.05}

    # Simulate tool cost (0.05 cents per text operation)
    return {
        "result": result,
        "operation": operation,
        "input_text": text,
        "cost_in_cents": 0.05,
    }


class DataFormatterInput(BaseModel):
    data: str = Field(description="Data to format as JSON string")
    format_type: str = Field(description="Format type: 'json', 'csv', 'summary'")


async def data_formatter_tool(input: DataFormatterInput) -> Dict[str, Any]:
    """Data formatting tool for testing."""
    try:
        import json

        if input.format_type == "json":
            # Pretty print JSON
            data = json.loads(input.data)
            result = json.dumps(data, indent=2)
        elif input.format_type == "csv":
            # Convert JSON to CSV-like format
            data = json.loads(input.data)
            if isinstance(data, list) and data:
                headers = list(data[0].keys())
                result = ",".join(headers) + "\n"
                for item in data:
                    result += ",".join(str(item.get(h, "")) for h in headers) + "\n"
            else:
                result = "Invalid data for CSV conversion"
        elif input.format_type == "summary":
            # Create a summary of the data
            data = json.loads(input.data)
            if isinstance(data, list):
                result = f"List with {len(data)} items"
            elif isinstance(data, dict):
                result = f"Object with {len(data)} keys: {', '.join(data.keys())}"
            else:
                result = f"Data type: {type(data).__name__}, value: {data}"
        else:
            return {"error": f"Unknown format type: {input.format_type}"}

        # Simulate tool cost (0.2 cents per format operation)
        return {
            "result": result,
            "format_type": input.format_type,
            "cost_in_cents": 0.2,
        }
    except Exception as e:
        return {"error": str(e), "cost_in_cents": 0.2}


@pytest.mark.asyncio
async def test_basic_agent_functionality():
    """Test basic agent creation and processing."""
    agent = Agent(
        agent_id="test_agent",
        provider="openai",
        model="gpt-4.1",
        system_prompt="You are a helpful assistant that responds concisely.",
        tools=[calculator_tool],
    )

    messages = [{"role": "user", "content": "What is 1334235 + 123124?"}]

    response = await agent.process(messages)
    assert response.content is not None
    assert len(response.content) > 0
    print(f"✅ Basic agent test passed: {response.content[:100]}...")


@pytest.mark.asyncio
async def test_orchestrator_with_predefined_subagents():
    """Test orchestrator with pre-defined subagents."""
    # Create main agent
    main_agent = Agent(
        agent_id="orchestrator",
        provider="openai",
        model="gpt-4.1",
        system_prompt="""You are an orchestrator. You can delegate tasks to subagents:
        - calculator_agent: Can perform mathematical calculations
        - text_agent: Can process text (count words, reverse, uppercase)
        
        Use delegate_to_subagents tool to assign tasks.""",
    )

    # Create subagents
    calculator_agent = Agent(
        agent_id="calculator_agent",
        provider="openai",
        model="gpt-4.1",
        system_prompt="You are a calculator. Use the calculator_tool to perform calculations.",
        tools=[calculator_tool],
    )

    text_agent = Agent(
        agent_id="text_agent",
        provider="openai",
        model="gpt-4.1",
        system_prompt="You are a text processor. Use the text_processor_tool to process text.",
        tools=[text_processor_tool],
    )

    # Create orchestrator
    orchestrator = AgentOrchestrator(main_agent=main_agent)

    # Register subagents
    orchestrator.register_subagent(calculator_agent)
    orchestrator.register_subagent(text_agent)

    messages = [
        {
            "role": "user",
            "content": "Calculate 15 * 7 and also count the words in 'Hello world this is a test'",
        }
    ]

    response = await orchestrator.process(messages)
    assert response.content is not None
    print(f"✅ Predefined subagents test passed: {response.content[:200]}...")


@pytest.mark.asyncio
async def test_dynamic_agent_creation():
    """Test dynamic agent creation and orchestration."""
    # Create main orchestrator agent
    main_agent = Agent(
        agent_id="dynamic_orchestrator",
        provider="openai",
        model="gpt-4.1",
        system_prompt="""You are a dynamic orchestrator that automatically creates specialized subagents.
        
        When you receive a complex task:
        1. Analyze what needs to be done
        2. Use the plan_and_create_subagents tool to dynamically create the right subagents
        3. The tool will automatically create subagents with appropriate tools and execute tasks
        4. Synthesize the results and provide a comprehensive response""",
    )

    # Create orchestrator with available tools
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[calculator_tool, text_processor_tool, data_formatter_tool],
        subagent_provider="openai",
        subagent_model="gpt-4.1",
        subagent_designer_provider="openai",
        subagent_designer_model="gpt-4.1",
    )

    messages = [
        {
            "role": "user",
            "content": """I need help with multiple tasks:
        1. Calculate what 25 * 4 + 12 equals
        2. Count the words in "The quick brown fox jumps over the lazy dog"
        3. Format this data as JSON: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        
        Please handle all these tasks efficiently.""",
        }
    ]

    response = await orchestrator.process(messages)
    assert response.content is not None

    print(f"Response: {response.content}")

    # Check that the response contains results for all three tasks
    content = response.content.lower()
    assert "100" in content or "112" in content  # 25*4+12 = 112
    assert "9" in content  # word count
    assert "alice" in content or "bob" in content  # JSON data

    print("✅ Dynamic agent creation test passed")
    print(f"Response length: {len(response.content)} characters")
    print(f"Response preview: {response.content[:300]}...")

    # Verify that subagents were created
    assert len(orchestrator.subagents) > 0
    print(
        f"✅ Created {len(orchestrator.subagents)} subagents: {list(orchestrator.subagents.keys())}"
    )


def test_orchestrator_tool_registry():
    """Test that the tool registry is properly populated."""
    main_agent = Agent(agent_id="test_main", provider="openai", model="gpt-4.1")

    tools = [calculator_tool, text_processor_tool, data_formatter_tool]
    orchestrator = AgentOrchestrator(main_agent=main_agent, available_tools=tools)

    # Check tool registry
    assert len(orchestrator.tool_registry) == 3
    assert "calculator_tool" in orchestrator.tool_registry
    assert "text_processor_tool" in orchestrator.tool_registry
    assert "data_formatter_tool" in orchestrator.tool_registry

    # Check tool descriptions
    calc_desc = orchestrator._get_tool_description(calculator_tool)
    assert "calculator" in calc_desc.lower()

    print("✅ Tool registry test passed")


def test_agent_memory_configuration():
    """Test agent creation with memory configuration."""
    memory_config = {"token_threshold": 10000, "preserve_last_n_messages": 5}

    agent = Agent(
        agent_id="memory_test_agent",
        provider="openai",
        model="gpt-4.1",
        memory_config=memory_config,
    )

    assert agent.memory_manager is not None
    print("✅ Memory configuration test passed")


@pytest.mark.asyncio
async def test_cost_tracking_with_tools():
    """Test that tool costs are properly tracked and aggregated."""
    # Create main agent
    main_agent = Agent(
        agent_id="cost_test_main",
        provider="openai",
        model="gpt-4.1",
        system_prompt="You are an orchestrator that delegates tasks.",
    )

    # Create orchestrator with tools that have costs
    tools = [calculator_tool, text_processor_tool, data_formatter_tool]
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=tools,
        subagent_provider="openai",
        subagent_model="gpt-4o-mini",
    )

    # Test message that will trigger multiple tools
    messages = [
        {
            "role": "user",
            "content": """I need help with three tasks:
        1. Calculate: 100 + 200 + 300
        2. Count words in "This is a test sentence"
        3. Format this data as JSON: [{"id": 1}, {"id": 2}]
        
        Use the planning tool to create specialized agents for these tasks.""",
        }
    ]

    response = await orchestrator.process(messages)

    # Check if tool outputs contain cost information
    if response.tool_outputs:
        for tool_output in response.tool_outputs:
            if tool_output.get("name") == "plan_and_create_subagents":
                result = tool_output.get("result", {})
                cost_breakdown = result.get("total_cost_breakdown", {})

                print("\n💰 Cost Breakdown:")
                print(
                    f"  Planning cost: ${cost_breakdown.get('planning_cost_in_cents', 0) / 100:.4f}"
                )
                print(
                    f"  Subagent costs: ${cost_breakdown.get('subagent_costs_in_cents', 0) / 100:.4f}"
                )
                print(
                    f"  Tool costs: ${cost_breakdown.get('tool_costs_in_cents', 0) / 100:.4f}"
                )
                print(
                    f"  Total cost: ${cost_breakdown.get('total_cost_in_cents', 0) / 100:.4f}"
                )

                # Verify tool costs are tracked (should be > 0 since our tools have costs)
                assert cost_breakdown.get("tool_costs_in_cents", 0) > 0, (
                    "Tool costs should be tracked"
                )
                assert cost_breakdown.get("total_cost_in_cents", 0) > 0, (
                    "Total cost should include all components"
                )

    print("✅ Cost tracking test passed")


if __name__ == "__main__":
    # Run tests individually to better handle API rate limits
    import sys

    print("Running Dynamic Agent Orchestration E2E Tests")
    print("=" * 50)

    async def run_tests():
        test_functions = [
            test_basic_agent_functionality,
            test_orchestrator_with_predefined_subagents,
            test_dynamic_agent_creation,
            test_cost_tracking_with_tools,
        ]

        passed = 0
        total = len(test_functions)

        for test_func in test_functions:
            try:
                print(f"\n🧪 Running {test_func.__name__}...")
                await test_func()
                passed += 1
            except Exception as e:
                print(f"❌ {test_func.__name__} failed: {e}")

        print(f"\n📊 Results: {passed}/{total} tests passed")

        # Run non-async tests
        try:
            test_orchestrator_tool_registry()
            test_agent_memory_configuration()
            passed += 2
            total += 2
        except Exception as e:
            print(f"❌ Non-async tests failed: {e}")

        print(f"📊 Final Results: {passed}/{total} tests passed")
        return passed == total

    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
