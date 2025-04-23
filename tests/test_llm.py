import unittest
import pytest
from defog.llm.utils import (
    map_model_to_chat_fn_async,
    chat_async,
    chat_anthropic_async,
    chat_gemini_async,
    chat_openai_async,
)
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

    def test_map_model_to_chat_fn_async(self):
        self.assertEqual(
            map_model_to_chat_fn_async("claude-3-5-sonnet-20241022"),
            chat_anthropic_async,
        )

        self.assertEqual(
            map_model_to_chat_fn_async("gemini-1.5-flash-002"),
            chat_gemini_async,
        )

        self.assertEqual(map_model_to_chat_fn_async("gpt-4o-mini"), chat_openai_async)

        with self.assertRaises(ValueError):
            map_model_to_chat_fn_async("unknown-model")

    @pytest.mark.asyncio
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
            response = await chat_async(
                model,
                messages,
                max_completion_tokens=4000,
                temperature=0.0,
                seed=0,
                max_retries=1,
            )
            self.assertIsInstance(response.content, str)
            self.assertIsInstance(response.time, float)

    @pytest.mark.asyncio
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
            response = await chat_async(
                model,
                messages_sql,
                max_completion_tokens=4000,
                temperature=0.0,
                seed=0,
                max_retries=1,
            )
            self.check_sql(response.content)
            self.assertIsInstance(response.time, float)

    @pytest.mark.asyncio
    async def test_sql_chat_structured_reasoning_effort_async(self):
        reasoning_effort = ["low", "medium", "high", None]
        for effort in reasoning_effort:
            response = await chat_async(
                model="o1",
                messages=messages_sql_structured,
                max_completion_tokens=4000,
                temperature=0.0,
                seed=0,
                response_format=ResponseFormat,
                reasoning_effort=effort,
                max_retries=1,
            )
            self.check_sql(response.content.sql)
            self.assertIsInstance(response.content.reasoning, str)

    @pytest.mark.asyncio
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
            response = await chat_async(
                model,
                messages_sql_structured,
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
