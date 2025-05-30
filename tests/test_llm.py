import unittest
import pytest
from defog.llm.utils import (
    map_model_to_provider,
    chat_async,
)
from defog.llm.llm_providers import LLMProvider
import re

from pydantic import BaseModel

messages_sql = [
    {
        "role": "system",
        "content": "Your task is to generate SQL given a natural language question and schema of the user's database. Do not use aliases. Return only the SQL without ```.",
    },
    {
        "role": "user",
        "content": f"""Question: What is the total number of orders?
Schema:
```sql
CREATE TABLE orders (
    order_id int,
    customer_id int,
    employee_id int,
    order_date date
);
```
""",
    },
]

acceptable_sql = [
    "select count(*) from orders",
    "select count(order_id) from orders",
    "select count(*) as total_orders from orders",
    "select count(order_id) as total_orders from orders",
]


class ResponseFormat(BaseModel):
    reasoning: str
    sql: str


messages_sql_structured = [
    {
        "role": "system",
        "content": "Your task is to generate SQL given a natural language question and schema of the user's database. Do not use aliases.",
    },
    {
        "role": "user",
        "content": f"""Question: What is the total number of orders?
Schema:
```sql
CREATE TABLE orders (
    order_id int,
    customer_id int,
    employee_id int,
    order_date date
);
```
""",
    },
]


class TestChatClients(unittest.IsolatedAsyncioTestCase):
    def check_sql(self, sql: str):
        sql = sql.replace("```sql", "").replace("```", "").strip(";\n").lower()
        sql = re.sub(r"(\s+)", " ", sql)
        self.assertIn(sql, acceptable_sql)

    def test_map_model_to_provider(self):
        self.assertEqual(
            map_model_to_provider("claude-3-5-sonnet-20241022"),
            LLMProvider.ANTHROPIC,
        )

        self.assertEqual(
            map_model_to_provider("gemini-1.5-flash-002"),
            LLMProvider.GEMINI,
        )

        self.assertEqual(map_model_to_provider("gpt-4o-mini"), LLMProvider.OPENAI)

        self.assertEqual(map_model_to_provider("deepseek-chat"), LLMProvider.DEEPSEEK)

        self.assertEqual(
            map_model_to_provider("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            LLMProvider.TOGETHER,
        )

        with self.assertRaises(Exception):
            map_model_to_provider("unknown-model")

    @pytest.mark.asyncio(loop_scope="session")
    async def test_simple_chat_async(self):
        models = [
            "claude-3-7-sonnet-latest",
            "gpt-4.1-mini",
            "o4-mini",
            "o3",
            "gemini-2.0-flash",
            "gemini-2.5-pro-preview-03-25",
        ]
        messages = [
            {"role": "user", "content": "Return a greeting in not more than 2 words\n"}
        ]
        for model in models:
            provider = map_model_to_provider(model)
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages,
                max_completion_tokens=4000,
                temperature=0.0,
                seed=0,
                max_retries=1,
            )
            self.assertIsInstance(response.content, str)
            self.assertIsInstance(response.time, float)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sql_chat_async(self):
        models = [
            "gpt-4o-mini",
            "o1",
            "gemini-2.0-flash",
            "gemini-2.5-pro-preview-03-25",
            "o3",
            "o4-mini",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
        ]
        for model in models:
            provider = map_model_to_provider(model)
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages_sql,
                max_completion_tokens=4000,
                temperature=0.0,
                seed=0,
                max_retries=1,
            )
            self.check_sql(response.content)
            self.assertIsInstance(response.time, float)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sql_chat_structured_reasoning_effort_async(self):
        reasoning_effort = ["low", "medium", "high", None]
        for effort in reasoning_effort:
            for model in ["o4-mini", "claude-3-7-sonnet-latest"]:
                provider = map_model_to_provider(model)
                response = await chat_async(
                    provider=provider,
                    model=model,
                    messages=messages_sql_structured,
                    max_completion_tokens=32000,
                    temperature=0.0,
                    seed=0,
                    response_format=ResponseFormat,
                    reasoning_effort=effort,
                    max_retries=1,
                )
                self.check_sql(response.content.sql)
                self.assertIsInstance(response.content.reasoning, str)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sql_chat_structured_async(self):
        models = [
            "gpt-4o",
            "o1",
            "gemini-2.0-flash",
            "gemini-2.5-pro-preview-03-25",
            "claude-3-7-sonnet-latest",  # Added Anthropic model to test structured output
            "o3",
            "o4-mini",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
        ]
        for model in models:
            provider = map_model_to_provider(model)
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages_sql_structured,
                max_completion_tokens=4000,
                temperature=0.0,
                seed=0,
                response_format=ResponseFormat,
                max_retries=1,
            )
            self.check_sql(response.content.sql)
            self.assertIsInstance(response.content.reasoning, str)


if __name__ == "__main__":
    unittest.main()
