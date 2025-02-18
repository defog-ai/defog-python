import unittest
import pytest
from defog.llm.utils import chat_async, chat_openai, chat_anthropic
from defog.llm.utils_function_calling import get_function_specs
from pydantic import BaseModel, Field
import aiohttp
from io import StringIO
import json

# ==================================================================================================
# Functions for function calling
# ==================================================================================================

IO_STREAM = StringIO()


def log_to_file(function_name, args, result):
    """
    Simple function to test logging to a StringIO object.
    Used in test_post_tool_calls_openai and test_post_tool_calls_anthropic
    """
    message = {
        "function_name": function_name,
        "args": args,
        "result": result,
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
                                "default": 0.0,
                                "description": "The latitude of the location",
                                "title": "Latitude",
                                "type": "number",
                            },
                            "longitude": {
                                "default": 0.0,
                                "description": "The longitude of the location",
                                "title": "Longitude",
                                "type": "number",
                            },
                        },
                        "title": "WeatherInput",
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
                            "a": {"default": 0, "title": "A", "type": "integer"},
                            "b": {"default": 0, "title": "B", "type": "integer"},
                        },
                        "title": "Numbers",
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
                            "a": {"default": 0, "title": "A", "type": "integer"},
                            "b": {"default": 0, "title": "B", "type": "integer"},
                        },
                        "title": "Numbers",
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
                            "default": 0.0,
                            "description": "The latitude of the location",
                            "title": "Latitude",
                            "type": "number",
                        },
                        "longitude": {
                            "default": 0.0,
                            "description": "The longitude of the location",
                            "title": "Longitude",
                            "type": "number",
                        },
                    },
                    "title": "WeatherInput",
                    "type": "object",
                },
            },
            {
                "name": "numsum",
                "description": "This function returns the sum of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"default": 0, "title": "A", "type": "integer"},
                        "b": {"default": 0, "title": "B", "type": "integer"},
                    },
                    "title": "Numbers",
                    "type": "object",
                },
            },
            {
                "name": "numprod",
                "description": "This function returns the product of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"default": 0, "title": "A", "type": "integer"},
                        "b": {"default": 0, "title": "B", "type": "integer"},
                    },
                    "title": "Numbers",
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

        self.arithmetic_qn = "What is the product of 31283 and 2323, added to 5? Return only the final answer, nothing else. Always use tools for arithmetic."
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
        self.assertSetEqual(set(result.tools_used), {"numsum", "numprod"})

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
        self.assertSetEqual(set(result.tools_used), {"get_weather"})
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
        self.assertSetEqual(set(result.tools_used), {"numsum", "numprod"})
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
        self.assertSetEqual(set(result.tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
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
        self.assertSetEqual(set(result.tools_used), {"numsum", "numprod"})
        expected_stream_value = json.dumps({
            "function_name": "numprod",
            "args": {
                "a": 31283,
                "b": 2323
                },
                "result": 72670409
            }, indent=4) + "\n" + json.dumps({
                "function_name": "numsum",
                "args": {
                    "a": 72670409,
                    "b": 5
                },
                "result": 72670414
            }, indent=4) + "\n"
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
        self.assertSetEqual(set(result.tools_used), {"numsum", "numprod"})
        expected_stream_value = json.dumps({
            "function_name": "numprod",
            "args": {
                "a": 31283,
                "b": 2323
                },
                "result": 72670409
            }, indent=4) + "\n" + json.dumps({
                "function_name": "numsum",
                "args": {
                    "a": 72670409,
                    "b": 5
                },
                "result": 72670414
            }, indent=4) + "\n"
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
        self.assertSetEqual(set(result.tools_used), {"get_weather"})
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
        self.assertSetEqual(set(result.tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
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
        self.assertSetEqual(set(result.tools_used), {"numsum"})
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
        self.assertSetEqual(set(result.tools_used), {"numsum"})
        self.assertEqual(result.tool_outputs[0]["name"], "numsum")
        self.assertIn("104", result.content.lower())

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
        self.assertSetEqual(set(result.tools_used), {"numprod"})
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
        self.assertSetEqual(set(result.tools_used), {"numprod"})
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
