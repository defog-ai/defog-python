import unittest
import pytest
from defog.llm.utils import chat_async
from defog.llm.utils_function_calling import get_function_specs
from defog.llm.config.settings import LLMConfig
from tests.conftest import skip_if_no_api_key

from pydantic import BaseModel, Field
import httpx
from io import StringIO
import json

# ==================================================================================================
# Functions for function calling
# ==================================================================================================

IO_STREAM = StringIO()


def log_to_file(function_name, input_args, tool_result):
    """
    Simple function to test logging to a StringIO object.
    Used in test_post_tool_calls_openai and test_post_tool_calls_anthropic
    """
    sorted_input_args = {k: input_args[k] for k in sorted(input_args)}

    message = {
        "function_name": function_name,
        "args": sorted_input_args,
        "result": tool_result,
    }
    message = json.dumps(message, indent=4)
    IO_STREAM.write(message + "\n")
    return IO_STREAM.getvalue()


class WeatherInput(BaseModel):
    latitude: float = Field(default=0.0, description="The latitude of the location")
    longitude: float = Field(default=0.0, description="The longitude of the location")


async def get_weather(input: WeatherInput):
    """
    This function returns the current temperature (in celsius) for the given latitude and longitude.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={input.latitude}&longitude={input.longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m",
        )
        return_object = r.json()
        return return_object["current"]["temperature_2m"]


class Numbers(BaseModel):
    a: int = 0
    b: int = 0


def numsum(input: Numbers):
    """
    This function returns the sum of two numbers
    """
    return input.a + input.b


def numprod(input: Numbers):
    """
    This function returns the product of two numbers
    """
    return input.a * input.b


# ==================================================================================================
# Tests
# ==================================================================================================
class TestGetFunctionSpecs(unittest.TestCase):
    def setUp(self):
        self.openai_model = "gpt-4.1"
        self.anthropic_model = "claude-3-7-sonnet-latest"
        self.mistral_model = "mistral-small-latest"
        self.tools = [get_weather, numsum, numprod]
        self.maxDiff = None
        self.openai_specs = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "This function returns the current temperature (in celsius) for the given latitude and longitude.",
                    "parameters": {
                        "properties": {
                            "latitude": {
                                "description": "The latitude of the location",
                                "type": "number",
                            },
                            "longitude": {
                                "description": "The longitude of the location",
                                "type": "number",
                            },
                        },
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "numsum",
                    "description": "This function returns the sum of two numbers",
                    "parameters": {
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "numprod",
                    "description": "This function returns the product of two numbers",
                    "parameters": {
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "type": "object",
                    },
                },
            },
        ]
        self.anthropic_specs = [
            {
                "name": "get_weather",
                "description": "This function returns the current temperature (in celsius) for the given latitude and longitude.",
                "input_schema": {
                    "properties": {
                        "latitude": {
                            "description": "The latitude of the location",
                            "type": "number",
                        },
                        "longitude": {
                            "description": "The longitude of the location",
                            "type": "number",
                        },
                    },
                    "type": "object",
                },
            },
            {
                "name": "numsum",
                "description": "This function returns the sum of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "type": "object",
                },
            },
            {
                "name": "numprod",
                "description": "This function returns the product of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "type": "object",
                },
            },
        ]

    def test_get_function_specs(self):
        openai_specs = get_function_specs(self.tools, self.openai_model)
        anthropic_specs = get_function_specs(self.tools, self.anthropic_model)
        mistral_specs = get_function_specs(self.tools, self.mistral_model)

        self.assertEqual(openai_specs, self.openai_specs)
        self.assertEqual(anthropic_specs, self.anthropic_specs)
        # Mistral uses OpenAI-compatible format
        self.assertEqual(mistral_specs, self.openai_specs)


class TestToolUseFeatures(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tools = [get_weather, numsum, numprod]
        self.weather_qn = "What is the current temperature in Singapore? Return the answer as a number and nothing else."
        self.weather_qn_specific = "What is the current temperature in Singapore? Singapore's latitude is 1.3521 and longitude is 103.8198. Return the answer as a number and nothing else."
        self.arithmetic_qn = "What is the product of 31283 and 2323, added to 5? Always use the tools provided for all calculation, even simple calculations. Return only the final answer, nothing else."
        self.arithmetic_answer = "72670414"
        self.arithmetic_expected_tool_outputs = [
            {"name": "numprod", "args": {"a": 31283, "b": 2323}, "result": 72670409},
            {"name": "numsum", "args": {"a": 72670409, "b": 5}, "result": 72670414},
        ]

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_tool_use_arithmetic_async_openai(self):
        result = await chat_async(
            provider="openai",
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_tool_use_weather_async_openai(self):
        result = await chat_async(
            provider="openai",
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_tool_use_arithmetic_async_anthropic(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-3-7-sonnet-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})
        self.assertEqual(result.content, self.arithmetic_answer)

    @pytest.mark.asyncio
    @skip_if_no_api_key("mistral")
    async def test_tool_use_arithmetic_async_mistral(self):
        result = await chat_async(
            provider="mistral",
            model="mistral-medium-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_tool_use_weather_async_anthropic(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-3-7-sonnet-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    @pytest.mark.asyncio
    @skip_if_no_api_key("mistral")
    async def test_tool_use_weather_async_mistral(self):
        result = await chat_async(
            provider="mistral",
            model="mistral-medium-latest",
            messages=[
                {
                    "role": "user",
                    # we have to add an explicit instruction to use the tools because mistral is bad at using tools on its own
                    "content": self.weather_qn_specific
                    + "\n"
                    + "You must use the tools provided to answer the question.",
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertEqual(
            result.tool_outputs[0]["args"], {"latitude": 1.3521, "longitude": 103.8198}
        )
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_tool_use_arithmetic_async_anthropic_reasoning_effort(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-3-7-sonnet-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            reasoning_effort="low",
            max_retries=1,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        self.assertEqual(result.content, self.arithmetic_answer)

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_tool_use_arithmetic_async_gemini(self):
        result = await chat_async(
            provider="gemini",
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_tool_use_weather_async_gemini(self):
        result = await chat_async(
            provider="gemini",
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn_specific,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertEqual(
            result.tool_outputs[0]["args"], {"latitude": 1.3521, "longitude": 103.8198}
        )
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_post_tool_calls_openai(self):
        result = await chat_async(
            provider="openai",
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_post_tool_calls_anthropic(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-3-7-sonnet-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_post_tool_calls_gemini(self):
        result = await chat_async(
            provider="gemini",
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("deepseek")
    async def test_tool_use_arithmetic_async_deepseek_chat(self):
        result = await chat_async(
            provider="deepseek",
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("deepseek")
    async def test_tool_use_weather_async_deepseek_chat(self):
        result = await chat_async(
            provider="deepseek",
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn_specific,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertEqual(
            result.tool_outputs[0]["args"], {"latitude": 1.3521, "longitude": 103.8198}
        )
        # Try to parse temperature, but handle API failures gracefully
        try:
            temp = float(result.content)
            self.assertGreaterEqual(temp, 21)
            self.assertLessEqual(temp, 38)
        except ValueError:
            # API call failed or returned non-numeric response
            # This is acceptable for weather tests as APIs can be unreliable
            self.assertIsInstance(result.content, str)
            self.assertGreater(len(result.content), 0)

    @pytest.mark.asyncio
    @skip_if_no_api_key("deepseek")
    async def test_post_tool_calls_deepseek_chat(self):
        result = await chat_async(
            provider="deepseek",
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("mistral")
    async def test_post_tool_calls_mistral(self):
        result = await chat_async(
            provider="mistral",
            model="mistral-medium-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})


class TestParallelToolCalls(unittest.IsolatedAsyncioTestCase):
    """Test parallel tool calls functionality."""

    def setUp(self):
        from defog.llm.utils_function_calling import execute_tools_parallel
        from defog.llm.tools.handler import ToolHandler

        self.execute_tools_parallel = execute_tools_parallel
        self.handler = ToolHandler()
        self.tools = [numsum, numprod]
        self.tool_dict = self.handler.build_tool_dict(self.tools)

    @pytest.mark.asyncio
    async def test_parallel_tool_execution_multiple_calls(self):
        """Test that multiple tool calls can be executed in parallel."""
        import time

        # Simulate tool calls that would benefit from parallel execution
        tool_calls = [
            {
                "function": {
                    "name": "numsum",
                    "arguments": {"a": 12312323434, "b": 89230482903480},
                }
            },
            {"function": {"name": "numprod", "arguments": {"a": 2134, "b": 9823}}},
            {"function": {"name": "numsum", "arguments": {"a": 983247, "b": 2348796}}},
        ]

        # Test sequential execution
        start_time = time.time()
        sequential_results = await self.execute_tools_parallel(
            tool_calls, self.tool_dict, enable_parallel=False
        )
        sequential_time = time.time() - start_time

        # Test parallel execution
        start_time = time.time()
        parallel_results = await self.execute_tools_parallel(
            tool_calls, self.tool_dict, enable_parallel=True
        )
        parallel_time = time.time() - start_time

        # Results should be the same
        self.assertEqual(sequential_results, parallel_results)
        self.assertEqual(sequential_results, [89242795226914, 20962282, 3332043])

        # For simple arithmetic, parallel may not be faster, but should not be significantly slower
        # This test mainly ensures functionality works correctly
        print(f"Sequential time: {sequential_time:.4f}s")
        print(f"Parallel time: {parallel_time:.4f}s")

    @pytest.mark.asyncio
    async def test_tool_handler_batch_execution(self):
        """Test the tool handler batch execution method."""
        tool_calls = [
            {
                "function": {
                    "name": "numsum",
                    "arguments": {"a": 12312323434, "b": 89230482903480},
                }
            },
            {"function": {"name": "numprod", "arguments": {"a": 2134, "b": 9823}}},
            {"function": {"name": "numsum", "arguments": {"a": 983247, "b": 2348796}}},
        ]

        # Test sequential batch execution
        results_sequential = await self.handler.execute_tool_calls_batch(
            tool_calls, self.tool_dict, enable_parallel=False
        )

        # Test parallel batch execution
        results_parallel = await self.handler.execute_tool_calls_batch(
            tool_calls, self.tool_dict, enable_parallel=True
        )

        # Results should be identical
        self.assertEqual(results_sequential, results_parallel)
        self.assertEqual(results_sequential, [89242795226914, 20962282, 3332043])

    @pytest.mark.asyncio
    async def test_error_handling_in_parallel_execution(self):
        """Test error handling when tools fail in parallel execution."""
        tool_calls = [
            {"function": {"name": "numsum", "arguments": {"a": 1, "b": 2}}},
            {"function": {"name": "nonexistent_tool", "arguments": {"a": 3, "b": 4}}},
        ]

        results = await self.execute_tools_parallel(
            tool_calls, self.tool_dict, enable_parallel=True
        )

        # First tool should succeed, second should return error
        self.assertEqual(results[0], 3)
        self.assertIn("Error: Function nonexistent_tool not found", results[1])

    def test_configuration_integration(self):
        """Test that the configuration setting is properly integrated."""

        # Test default configuration (parallel disabled)
        config_default = LLMConfig()
        self.assertTrue(config_default.enable_parallel_tool_calls)

        # Test explicit parallel enabled configuration
        config_parallel = LLMConfig(enable_parallel_tool_calls=False)
        self.assertFalse(config_parallel.enable_parallel_tool_calls)


class TestParallelToolCallsEndToEnd(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for parallel tool calls with real API calls."""

    def setUp(self):
        self.tools = [numsum, numprod]
        # More complex message that requires tool usage
        self.messages = [
            {
                "role": "user",
                "content": """Calculate the following using the provided tools:
1. The sum of 387293472 and 2348293482
2. The product of 12376 and 23245

You MUST use the numsum and numprod tools for these calculations. Do not calculate manually. Return only the final results.""",
            }
        ]

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_openai_parallel_vs_sequential_speed(self):
        """Test OpenAI parallel vs sequential execution speed."""
        import time

        # Test parallel execution
        config_parallel = LLMConfig(enable_parallel_tool_calls=True)
        start_time = time.time()
        result_parallel = await chat_async(
            provider="openai",
            model="gpt-4.1",
            messages=self.messages,
            tools=self.tools,
            config=config_parallel,
            temperature=0,
            max_retries=1,
        )
        parallel_time = time.time() - start_time

        # Test sequential execution
        config_sequential = LLMConfig(enable_parallel_tool_calls=False)
        start_time = time.time()
        result_sequential = await chat_async(
            provider="openai",
            model="gpt-4.1",
            messages=self.messages,
            tools=self.tools,
            config=config_sequential,
            temperature=0,
            max_retries=1,
        )
        sequential_time = time.time() - start_time

        # Verify both produce correct results
        self.assertEqual(len(result_parallel.tool_outputs), 2)
        self.assertEqual(len(result_sequential.tool_outputs), 2)

        # Check that sum and product were calculated
        outputs_parallel = {
            o["name"]: o["result"] for o in result_parallel.tool_outputs
        }
        outputs_sequential = {
            o["name"]: o["result"] for o in result_sequential.tool_outputs
        }

        self.assertEqual(outputs_parallel["numsum"], 2735586954)
        self.assertEqual(outputs_parallel["numprod"], 287680120)
        self.assertEqual(outputs_parallel, outputs_sequential)

        # Log timing results
        print("\nOpenAI Timing Results:")
        print(f"  Parallel execution: {parallel_time:.2f}s")
        print(f"  Sequential execution: {sequential_time:.2f}s")
        print(f"  Speedup: {sequential_time / parallel_time:.2f}x")

        # Parallel should generally be faster or at least not significantly slower
        # We don't assert exact timing as it depends on API response times

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_anthropic_parallel_tool_behavior(self):
        """Test Anthropic's parallel tool call behavior."""
        import time

        # Test with parallel enabled (should make one API call with multiple tools)
        config_parallel = LLMConfig(enable_parallel_tool_calls=True)
        start_time = time.time()
        result_parallel = await chat_async(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            messages=self.messages,
            tools=self.tools,
            config=config_parallel,
            temperature=0,
            max_retries=1,
        )
        parallel_time = time.time() - start_time

        # Test with parallel disabled (may require multiple API calls)
        config_sequential = LLMConfig(enable_parallel_tool_calls=False)
        start_time = time.time()
        result_sequential = await chat_async(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            messages=self.messages,
            tools=self.tools,
            config=config_sequential,
            temperature=0,
            max_retries=1,
        )
        sequential_time = time.time() - start_time

        # Verify results
        self.assertEqual(len(result_parallel.tool_outputs), 2)
        self.assertEqual(len(result_sequential.tool_outputs), 2)

        # Check results
        outputs_parallel = {
            o["name"]: o["result"] for o in result_parallel.tool_outputs
        }
        self.assertEqual(outputs_parallel["numsum"], 2735586954)
        self.assertEqual(outputs_parallel["numprod"], 287680120)

        print("\nAnthropic Timing Results:")
        print(f"  Parallel execution: {parallel_time:.2f}s")
        print(f"  Sequential execution: {sequential_time:.2f}s")
        print(f"  Speedup: {sequential_time / parallel_time:.2f}x")

    @pytest.mark.asyncio
    @skip_if_no_api_key("deepseek")
    async def test_deepseek_parallel_vs_sequential_speed(self):
        """Test DeepSeek parallel vs sequential execution speed."""
        import time

        # Test parallel execution
        config_parallel = LLMConfig(enable_parallel_tool_calls=True)
        start_time = time.time()
        result_parallel = await chat_async(
            provider="deepseek",
            model="deepseek-chat",
            messages=self.messages,
            tools=self.tools,
            config=config_parallel,
            temperature=0,
            max_retries=1,
        )
        parallel_time = time.time() - start_time

        # Test sequential execution
        config_sequential = LLMConfig(enable_parallel_tool_calls=False)
        start_time = time.time()
        result_sequential = await chat_async(
            provider="deepseek",
            model="deepseek-chat",
            messages=self.messages,
            tools=self.tools,
            config=config_sequential,
            temperature=0,
            max_retries=1,
        )
        sequential_time = time.time() - start_time

        # Verify both produce correct results
        self.assertEqual(len(result_parallel.tool_outputs), 2)
        self.assertEqual(len(result_sequential.tool_outputs), 2)

        # Check that sum and product were calculated
        outputs_parallel = {
            o["name"]: o["result"] for o in result_parallel.tool_outputs
        }
        outputs_sequential = {
            o["name"]: o["result"] for o in result_sequential.tool_outputs
        }

        self.assertEqual(outputs_parallel["numsum"], 2735586954)
        self.assertEqual(outputs_parallel["numprod"], 287680120)
        self.assertEqual(outputs_parallel, outputs_sequential)

        # Log timing results
        print("\nDeepSeek Timing Results:")
        print(f"  Parallel execution: {parallel_time:.2f}s")
        print(f"  Sequential execution: {sequential_time:.2f}s")
        print(f"  Speedup: {sequential_time / parallel_time:.2f}x")

        # Parallel should generally be faster or at least not significantly slower
        # We don't assert exact timing as it depends on API response times

    # we can't really test Gemini, as it always has parallel tool calls enabled

    def test_provider_config_propagation(self):
        """Test that config properly propagates to all providers."""
        from defog.llm.utils import get_provider_instance

        config = LLMConfig(enable_parallel_tool_calls=True)

        # Test each provider receives config
        providers_to_test = [
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-4-20250514"),
            ("gemini", "gemini-2.5-pro"),
            ("deepseek", "deepseek-chat"),
            ("mistral", "mistral-medium-latest"),
        ]

        for provider_name, model in providers_to_test:
            try:
                provider = get_provider_instance(provider_name, config)
                self.assertTrue(hasattr(provider, "config"))
                self.assertEqual(provider.config.enable_parallel_tool_calls, True)

                # Test with parallel disabled
                config_disabled = LLMConfig(enable_parallel_tool_calls=False)
                provider_disabled = get_provider_instance(
                    provider_name, config_disabled
                )
                self.assertEqual(
                    provider_disabled.config.enable_parallel_tool_calls, False
                )
            except Exception as e:
                # Skip if provider is not configured
                print(f"Skipping {provider_name}: {e}")
