import unittest
from unittest.mock import Mock, patch, AsyncMock, PropertyMock
import pytest
from defog.llm.utils import (
    chat_async,
    chat_openai,
    chat_anthropic,
    chat_gemini,
    _process_openai_response,
    _process_anthropic_response,
    _process_gemini_response,
)
from defog.llm.utils_function_calling import get_function_specs
from pydantic import BaseModel, Field
import aiohttp
from io import StringIO
import json
import asyncio
from anthropic.types import ToolUseBlock
from google.genai import types

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
    async with aiohttp.ClientSession() as client:
        r = await client.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={input.latitude}&longitude={input.longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m",
        )
        return_object = await r.json()
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
        self.openai_model = "gpt-4o"
        self.anthropic_model = "claude-3-5-sonnet-20241022"
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

        self.assertEqual(openai_specs, self.openai_specs)
        self.assertEqual(anthropic_specs, self.anthropic_specs)


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
    async def test_tool_use_arithmetic_async_openai(self):
        result = await chat_async(
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
        for expected, actual in zip(
            self.arithmetic_expected_tool_outputs, result.tool_outputs
        ):
            self.assertEqual(expected["name"], actual["name"])
            self.assertEqual(expected["args"], actual["args"])
            self.assertEqual(expected["result"], actual["result"])
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    async def test_tool_use_weather_async_openai(self):
        result = await chat_async(
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
    async def test_tool_use_arithmetic_async_anthropic(self):
        result = await chat_async(
            model="claude-3-5-sonnet-20241022",
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
        for expected, actual in zip(
            self.arithmetic_expected_tool_outputs, result.tool_outputs
        ):
            self.assertEqual(expected["name"], actual["name"])
            self.assertEqual(expected["args"], actual["args"])
            self.assertEqual(expected["result"], actual["result"])
        self.assertEqual(result.content, self.arithmetic_answer)

    @pytest.mark.asyncio
    async def test_tool_use_weather_async_anthropic(self):
        result = await chat_async(
            model="claude-3-5-sonnet-20241022",
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
    async def test_tool_use_arithmetic_async_gemini(self):
        result = await chat_async(
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
        for expected, actual in zip(
            self.arithmetic_expected_tool_outputs, result.tool_outputs
        ):
            self.assertEqual(expected["name"], actual["name"])
            self.assertEqual(expected["args"], actual["args"])
            self.assertEqual(expected["result"], actual["result"])
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    async def test_tool_use_weather_async_gemini(self):
        result = await chat_async(
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
    async def test_post_tool_calls_openai(self):
        result = await chat_async(
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
        for expected, actual in zip(
            self.arithmetic_expected_tool_outputs, result.tool_outputs
        ):
            self.assertEqual(expected["name"], actual["name"])
            self.assertEqual(expected["args"], actual["args"])
            self.assertEqual(expected["result"], actual["result"])
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})
        expected_stream_value = (
            json.dumps(
                {
                    "function_name": "numprod",
                    "args": {"a": 31283, "b": 2323},
                    "result": 72670409,
                },
                indent=4,
            )
            + "\n"
            + json.dumps(
                {
                    "function_name": "numsum",
                    "args": {"a": 72670409, "b": 5},
                    "result": 72670414,
                },
                indent=4,
            )
            + "\n"
        )
        self.assertEqual(IO_STREAM.getvalue(), expected_stream_value)

        # clear IO_STREAM
        IO_STREAM.seek(0)
        IO_STREAM.truncate()

    async def test_post_tool_calls_anthropic(self):
        result = await chat_async(
            model="claude-3-5-sonnet-latest",
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
        for expected, actual in zip(
            self.arithmetic_expected_tool_outputs, result.tool_outputs
        ):
            self.assertEqual(expected["name"], actual["name"])
            self.assertEqual(expected["args"], actual["args"])
            self.assertEqual(expected["result"], actual["result"])
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})
        expected_stream_value = (
            json.dumps(
                {
                    "function_name": "numprod",
                    "args": {"a": 31283, "b": 2323},
                    "result": 72670409,
                },
                indent=4,
            )
            + "\n"
            + json.dumps(
                {
                    "function_name": "numsum",
                    "args": {"a": 72670409, "b": 5},
                    "result": 72670414,
                },
                indent=4,
            )
            + "\n"
        )
        self.assertEqual(IO_STREAM.getvalue(), expected_stream_value)

        # clear IO_STREAM
        IO_STREAM.seek(0)
        IO_STREAM.truncate()

    @pytest.mark.asyncio
    async def test_post_tool_calls_gemini(self):
        result = await chat_async(
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
        for expected, actual in zip(
            self.arithmetic_expected_tool_outputs, result.tool_outputs
        ):
            self.assertEqual(expected["name"], actual["name"])
            self.assertEqual(expected["args"], actual["args"])
            self.assertEqual(expected["result"], actual["result"])
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})
        expected_stream_value = (
            json.dumps(
                {
                    "function_name": "numprod",
                    "args": {"a": 31283, "b": 2323},
                    "result": 72670409,
                },
                indent=4,
            )
            + "\n"
            + json.dumps(
                {
                    "function_name": "numsum",
                    "args": {"a": 72670409, "b": 5},
                    "result": 72670414,
                },
                indent=4,
            )
            + "\n"
        )
        self.assertEqual(IO_STREAM.getvalue(), expected_stream_value)

        # clear IO_STREAM
        IO_STREAM.seek(0)
        IO_STREAM.truncate()

    def test_async_tool_in_sync_function_openai(self):
        result = chat_openai(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    def test_async_tool_in_sync_function_anthropic(self):
        result = chat_anthropic(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    def test_async_tool_in_sync_function_gemini(self):
        result = chat_gemini(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn_specific,
                },
            ],
            tools=self.tools,
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

    def test_required_tool_choice_openai(self):
        result = chat_openai(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Give me the result of 102 + 2",
                },
            ],
            tools=self.tools,
            tool_choice="required",
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum"})
        self.assertEqual(result.tool_outputs[0]["name"], "numsum")
        self.assertIn("104", result.content.lower())

    def test_required_tool_choice_anthropic(self):
        result = chat_anthropic(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "user",
                    "content": "Give me the result of 102 + 2",
                },
            ],
            tools=self.tools,
            tool_choice="required",
            max_completion_tokens=1000,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum"})
        self.assertEqual(result.tool_outputs[0]["name"], "numsum")
        self.assertIn("104", result.content.lower())

    def test_required_tool_choice_gemini(self):
        result = chat_gemini(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": "If I add 1 egg to 1 egg, how many eggs are there in total?",
                },
            ],
            tools=self.tools,
            tool_choice="required",
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum"})
        self.assertEqual(result.tool_outputs[0]["name"], "numsum")
        self.assertIn("2", result.content.lower())

    def test_forced_tool_choice_openai(self):
        """
        This test forces the use of numprod even though the user question asks for addition.
        """
        result = chat_openai(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Give me the result of 102 + 2",
                },
            ],
            tools=self.tools,
            tool_choice="numprod",
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertIn("numprod", set(tools_used))
        self.assertEqual(result.tool_outputs[0]["name"], "numprod")
        self.assertEqual(result.tool_outputs[0]["result"], 204)

    def test_forced_tool_choice_anthropic(self):
        """
        This test forces the use of numprod even though the user question asks for addition.
        """
        result = chat_anthropic(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "user",
                    "content": "Give me the result of 102 + 2",
                },
            ],
            tools=self.tools,
            tool_choice="numprod",
            max_completion_tokens=1000,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertIn("numprod", set(tools_used))
        self.assertEqual(result.tool_outputs[0]["name"], "numprod")
        self.assertEqual(result.tool_outputs[0]["result"], 204)

    def test_forced_tool_choice_gemini(self):
        """
        This test forces the use of numprod even though the user question asks for addition.
        """
        result = chat_gemini(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": "Give me the result of 102 + 2",
                },
            ],
            tools=self.tools,
            tool_choice="numprod",
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertIn("numprod", set(tools_used))
        self.assertEqual(result.tool_outputs[0]["name"], "numprod")
        self.assertEqual(result.tool_outputs[0]["result"], 204)

    def test_invalid_forced_tool_choice_openai(self):
        """
        This test forces the use of an invalid tool `sum` and checks that an error is raised
        """
        with self.assertRaises(ValueError) as context:
            result = chat_openai(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": self.arithmetic_qn,
                    },
                ],
                tools=self.tools,
                tool_choice="sum",
            )
        self.assertEqual(
            str(context.exception),
            "Forced function `sum` is not in the list of provided tools",
        )

    def test_invalid_forced_tool_choice_anthropic(self):
        """
        This test forces the use of an invalid tool `sum` and checks that an error is raised
        """
        with self.assertRaises(ValueError) as context:
            result = chat_anthropic(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {
                        "role": "user",
                        "content": self.arithmetic_qn,
                    },
                ],
                tools=self.tools,
                tool_choice="sum",
                max_completion_tokens=1000,
            )
        self.assertEqual(
            str(context.exception),
            "Forced function `sum` is not in the list of provided tools",
        )

    def test_invalid_forced_tool_choice_gemini(self):
        """
        This test forces the use of an invalid tool `sum` and checks that an error is raised
        """
        with self.assertRaises(ValueError) as context:
            result = chat_gemini(
                model="gemini-2.0-flash",
                messages=[
                    {
                        "role": "user",
                        "content": self.arithmetic_qn,
                    },
                ],
                tools=self.tools,
                tool_choice="sum",
            )
        self.assertEqual(
            str(context.exception),
            "Forced function `sum` is not in the list of provided tools",
        )


class TestToolChainExceptionsOpenAI(unittest.TestCase):
    def setUp(self):
        self.arithmetic_qn = "What is the product of 31283 and 2323, added to 5? Always use the tools provided for all calculation, even simple calculations. Return only the final answer, nothing else."
        self.request_params = {
            "messages": [
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ]
        }
        self.tools = [get_weather, numsum, numprod]
        self.tool_dict = {
            "get_weather": get_weather,
            "numsum": numsum,
            "numprod": numprod,
        }
        self.model = "gpt-models"
        self.response_format = None
        self.is_async = False
        self.post_tool_function = Mock()

        self.response = Mock()
        self.response.usage = Mock()
        self.response.usage.prompt_tokens = 10
        self.response.usage.prompt_tokens_details = Mock()
        self.response.usage.prompt_tokens_details.cached_tokens = 5
        self.response.usage.completion_tokens = 8
        self.response.id = "test_response_id"

        self.client = Mock()
        self.client.chat = Mock()
        self.client.chat.completions = Mock()
        self.client.chat.completions.create = Mock(return_value=self.response)

        self.message_mock = Mock()
        self.message_mock.tool_calls = [Mock(function=Mock())]
        type(self.message_mock.tool_calls[0].function).name = PropertyMock(
            return_value="numprod"
        )
        type(self.message_mock.tool_calls[0].function).arguments = PropertyMock(
            return_value="{'a': '1'}"
        )
        self.response.choices = [Mock(message=self.message_mock)]

        self.tool_not_found_exception = "Tool `non_existent_tool` not found"
        self.tool_execution_exception = (
            "Error executing tool `numprod`: something failed in tool execution"
        )
        self.post_tool_function_exception = "Error executing post_tool_function: something failed in post-tool execution"

    def test_tool_not_found_exception_openai(self):
        """
        Test that an exception is raised when a non-existent tool is consecutively called in the tool chain and error messages are added to the messages list.
        """

        # Set the tool name in the tool call to a non-existent tool
        type(self.message_mock.tool_calls[0].function).name = PropertyMock(
            return_value="non_existent_tool"
        )

        async def test_coroutine():
            return await _process_openai_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                response_format=self.response_format,
                tools=self.tools,
                tool_dict=self.tool_dict,
                model=self.model,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.tool_not_found_exception}",
        )
        self.assertEqual(
            self.request_params,
            {
                "messages": [
                    {"role": "user", "content": self.arithmetic_qn},
                    {"role": "assistant", "content": self.tool_not_found_exception},
                    {"role": "assistant", "content": self.tool_not_found_exception},
                ]
            },
        )

    @patch(
        "defog.llm.utils.execute_tool",
        side_effect=Exception("something failed in tool execution"),
    )
    def test_tool_execution_exception_openai(self, mock_execute_tool):
        """
        Test that an exception is raised when the execution of a tool fails consecutively in the tool chain and error messages are added to the messages list.
        """

        async def test_coroutine():
            return await _process_openai_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                response_format=self.response_format,
                tools=self.tools,
                tool_dict=self.tool_dict,
                model=self.model,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.tool_execution_exception}",
        )
        self.assertEqual(
            self.request_params,
            {
                "messages": [
                    {"role": "user", "content": self.arithmetic_qn},
                    {"role": "assistant", "content": self.tool_execution_exception},
                    {"role": "assistant", "content": self.tool_execution_exception},
                ]
            },
        )

    def test_post_tool_function_exception_openai(self):
        """
        Test that an exception is raised when the post-tool function fails consecutively in the tool chain and error messages are added to the messages list.
        """
        self.post_tool_function = Mock(
            side_effect=Exception("something failed in post-tool execution")
        )

        async def test_coroutine():
            return await _process_openai_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                response_format=self.response_format,
                tools=self.tools,
                tool_dict=self.tool_dict,
                model=self.model,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.post_tool_function_exception}",
        )
        self.assertEqual(
            self.request_params,
            {
                "messages": [
                    {"role": "user", "content": self.arithmetic_qn},
                    {"role": "assistant", "content": self.post_tool_function_exception},
                    {"role": "assistant", "content": self.post_tool_function_exception},
                ]
            },
        )


class TestToolChainExceptionsAnthropic(unittest.TestCase):

    def setUp(self):
        self.arithmetic_qn = "What is the product of 31283 and 2323, added to 5? Always use the tools provided for all calculation, even simple calculations. Return only the final answer, nothing else."
        self.request_params = {
            "messages": [
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ]
        }
        self.tools = [get_weather, numsum, numprod]
        self.tool_dict = {
            "get_weather": get_weather,
            "numsum": numsum,
            "numprod": numprod,
        }
        self.is_async = False
        self.post_tool_function = Mock()

        self.tool_use_block_mock = Mock(spec=ToolUseBlock)
        self.tool_use_block_mock.name = "numprod"
        self.tool_use_block_mock.input = {"a": "1"}
        self.tool_use_block_mock.id = "mock_id"

        self.response = Mock()
        self.response.usage = Mock()
        self.response.usage.input_tokens = 10
        self.response.usage.output_tokens = 8
        self.response.content = [self.tool_use_block_mock]

        self.client = Mock()
        self.client.messages = Mock()
        self.client.messages.create = Mock()
        self.client.messages.create = Mock(return_value=self.response)

        self.tool_not_found_exception = "Tool `non_existent_tool` not found"
        self.tool_execution_exception = (
            "Error executing tool `numprod`: something failed in tool execution"
        )
        self.post_tool_function_exception = "Error executing post_tool_function: something failed in post-tool execution"

    def test_tool_not_found_exception_anthropic(self):
        """
        Test that an exception is raised when a non-existent tool is consecutively called in the tool chain and error messages are added to the messages list.
        """

        # Set the tool name in the tool call to a non-existent tool
        type(self.tool_use_block_mock).name = PropertyMock(
            return_value="non_existent_tool"
        )

        async def test_coroutine():
            return await _process_anthropic_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                tools=self.tools,
                tool_dict=self.tool_dict,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.tool_not_found_exception}",
        )
        self.assertEqual(
            self.request_params,
            {
                "messages": [
                    {"role": "user", "content": self.arithmetic_qn},
                    {"role": "assistant", "content": self.tool_not_found_exception},
                    {"role": "assistant", "content": self.tool_not_found_exception},
                ]
            },
        )

    @patch(
        "defog.llm.utils.execute_tool",
        side_effect=Exception("something failed in tool execution"),
    )
    def test_tool_execution_exception_anthropic(self, mock_execute_tool):
        """
        Test that an exception is raised when the execution of a tool fails consecutively in the tool chain and error messages are added to the messages list.
        """

        async def test_coroutine():
            return await _process_anthropic_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                tools=self.tools,
                tool_dict=self.tool_dict,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.tool_execution_exception}",
        )
        self.assertEqual(
            self.request_params,
            {
                "messages": [
                    {"role": "user", "content": self.arithmetic_qn},
                    {"role": "assistant", "content": self.tool_execution_exception},
                    {"role": "assistant", "content": self.tool_execution_exception},
                ]
            },
        )

    def test_post_tool_function_exception_anthropic(self):
        """
        Test that an exception is raised when the post-tool function fails consecutively in the tool chain and error messages are added to the messages list.
        """
        self.post_tool_function = Mock(
            side_effect=Exception("something failed in post-tool execution")
        )

        async def test_coroutine():
            return await _process_anthropic_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                tools=self.tools,
                tool_dict=self.tool_dict,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.post_tool_function_exception}",
        )
        self.assertEqual(
            self.request_params,
            {
                "messages": [
                    {"role": "user", "content": self.arithmetic_qn},
                    {"role": "assistant", "content": self.post_tool_function_exception},
                    {"role": "assistant", "content": self.post_tool_function_exception},
                ]
            },
        )


class TestToolChainExceptionsGemini(unittest.TestCase):

    def setUp(self):
        self.arithmetic_qn = "What is the product of 31283 and 2323, added to 5? Always use the tools provided for all calculation, even simple calculations. Return only the final answer, nothing else."
        self.request_params = {}
        self.tools = [get_weather, numsum, numprod]
        self.tool_dict = {
            "get_weather": get_weather,
            "numsum": numsum,
            "numprod": numprod,
        }
        self.model = "gpt-models"
        self.response_format = None
        self.is_async = False
        self.post_tool_function = Mock()

        self.response = Mock()
        self.response.candidates = [Mock()]
        self.response.usage_metadata = Mock()
        self.response.usage_metadata.prompt_token_count = 10
        self.response.usage_metadata.candidates_token_count = 8
        self.response.function_calls = [Mock()]
        type(self.response.function_calls[0]).name = PropertyMock(
            return_value="numprod"
        )
        type(self.response.function_calls[0]).args = PropertyMock(
            return_value={"a": "1"}
        )

        self.client = Mock()
        self.client.models = Mock()
        self.client.models.generate_content = Mock(return_value=self.response)
        self.messages = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self.arithmetic_qn)],
            )
        ]

        self.tool_not_found_exception = "Tool `non_existent_tool` not found"
        self.tool_execution_exception = (
            "Error executing tool `numprod`: something failed in tool execution"
        )
        self.post_tool_function_exception = "Error executing post_tool_function: something failed in post-tool execution"

    def test_tool_not_found_exception_gemini(self):
        """
        Test that an exception is raised when a non-existent tool is consecutively called in the tool chain and error messages are added to the messages list.
        """

        # Set the tool name in the tool call to a non-existent tool
        type(self.response.function_calls[0]).name = PropertyMock(
            return_value="non_existent_tool"
        )

        async def test_coroutine():
            return await _process_gemini_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                messages=self.messages,
                response_format=self.response_format,
                tools=self.tools,
                tool_dict=self.tool_dict,
                model=self.model,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.tool_not_found_exception}",
        )
        self.assertEqual(
            self.messages,
            [
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.arithmetic_qn,
                        )
                    ],
                    role="user",
                ),
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.tool_not_found_exception,
                        )
                    ],
                    role="model",
                ),
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.tool_not_found_exception,
                        )
                    ],
                    role="model",
                ),
            ],
        )

    @patch(
        "defog.llm.utils.execute_tool",
        side_effect=Exception("something failed in tool execution"),
    )
    def test_tool_execution_exception_gemini(self, mock_execute_tool):
        """
        Test that an exception is raised when the execution of a tool fails consecutively in the tool chain and error messages are added to the messages list.
        """

        async def test_coroutine():
            return await _process_gemini_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                messages=self.messages,
                response_format=self.response_format,
                tools=self.tools,
                tool_dict=self.tool_dict,
                model=self.model,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.tool_execution_exception}",
        )
        self.assertEqual(
            self.messages,
            [
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.arithmetic_qn,
                        )
                    ],
                    role="user",
                ),
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.tool_execution_exception,
                        )
                    ],
                    role="model",
                ),
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.tool_execution_exception,
                        )
                    ],
                    role="model",
                ),
            ],
        )

    def test_post_tool_function_exception_gemini(self):
        """
        Test that an exception is raised when the post-tool function fails consecutively in the tool chain and error messages are added to the messages list.
        """
        self.post_tool_function = Mock(
            side_effect=Exception("something failed in post-tool execution")
        )

        async def test_coroutine():
            return await _process_gemini_response(
                client=self.client,
                response=self.response,
                request_params=self.request_params,
                messages=self.messages,
                response_format=self.response_format,
                tools=self.tools,
                tool_dict=self.tool_dict,
                model=self.model,
                is_async=self.is_async,
                post_tool_function=self.post_tool_function,
            )

        with self.assertRaises(Exception) as context:
            asyncio.run(test_coroutine())

        self.assertEqual(
            str(context.exception),
            f"Consecutive errors during tool chaining: {self.post_tool_function_exception}",
        )
        self.assertEqual(
            self.messages,
            [
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.arithmetic_qn,
                        )
                    ],
                    role="user",
                ),
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.post_tool_function_exception,
                        )
                    ],
                    role="model",
                ),
                types.Content(
                    parts=[
                        types.Part(
                            video_metadata=None,
                            thought=None,
                            code_execution_result=None,
                            executable_code=None,
                            file_data=None,
                            function_call=None,
                            function_response=None,
                            inline_data=None,
                            text=self.post_tool_function_exception,
                        )
                    ],
                    role="model",
                ),
            ],
        )
