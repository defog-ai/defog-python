import unittest
import pytest
from defog.llm.utils import chat_async
from pydantic import BaseModel
import aiohttp


class WeatherInput(BaseModel):
    latitude: float
    longitude: float


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


class TestToolUseFeatures(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.asyncio
    async def test_tool_use_arithmetic(self):
        class Numbers(BaseModel):
            a: int = 0
            b: int = 0

        def numsum(input: Numbers):
            """
            This function return the sum of two numbers
            """
            return input.a + input.b

        def numprod(input: Numbers):
            """
            This function return the product of two numbers
            """
            return input.a * input.b

        tools = [numsum, numprod]

        for model in ["gpt-4o", "claude-3-5-sonnet-latest"]:
            result = await chat_async(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the product of 31283 and 2323, added to 5? Return only the final answer, nothing else.",
                    },
                ],
                tools=tools,
            )
            self.assertEqual(result.content, "72670414")

    @pytest.mark.asyncio
    async def test_tool_use_async_weather(self):
        tools = [get_weather]
        for model in ["gpt-4o", "claude-3-5-sonnet-latest"]:
            result = await chat_async(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the current temperature in Singapore? Return the answer as just a number.",
                    },
                ],
                tools=tools,
                max_retries=1,
            )
            # assert that the temperature is between 21 and 38
            self.assertGreaterEqual(float(result.content), 21)
            self.assertLessEqual(float(result.content), 38)
