import unittest
import pytest
import os
from defog.llm.citations import citations_tool
from defog.llm.llm_providers import LLMProvider


class TestCitations(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Skip tests if API keys are not available
        self.skip_openai = not os.getenv("OPENAI_API_KEY")
        self.skip_anthropic = not os.getenv("ANTHROPIC_API_KEY")

    @pytest.mark.asyncio
    async def test_simple_anthropic_citations(self):
        if self.skip_anthropic:
            self.skipTest("ANTHROPIC_API_KEY not set")

        question = "What is Rishabh's favourite food?"
        instructions = "Answer the question with high quality citations. If you don't know the answer, say 'I don't know'."
        documents = [
            {
                "document_name": "Rishabh's favourite food.txt",
                "document_content": "Rishabh's favourite food is pizza.",
            },
            {
                "document_name": "Medha's favourite food.txt",
                "document_content": "Medha's favourite food is pineapple.",
            },
        ]

        response = await citations_tool(
            question,
            instructions,
            documents,
            "claude-3-7-sonnet-latest",
            LLMProvider.ANTHROPIC,
        )

        # Check that response is a list of blocks
        self.assertIsInstance(response, list)
        self.assertGreater(len(response), 0)

        # Check that at least one block contains text mentioning pizza
        found_pizza = False
        for block in response:
            if block.get("type") == "text" and "pizza" in block.get("text", "").lower():
                found_pizza = True
                break
        self.assertTrue(
            found_pizza, "Response should mention pizza as Rishabh's favourite food"
        )

    @pytest.mark.asyncio
    async def test_complex_anthropic_citations(self):
        if self.skip_anthropic:
            self.skipTest("ANTHROPIC_API_KEY not set")

        question = "What are the key differences between Python and JavaScript programming languages?"
        instructions = "Provide a comprehensive comparison with citations. Focus on syntax, use cases, and performance characteristics."
        documents = [
            {
                "document_name": "Python Overview.txt",
                "document_content": """Python is a high-level, interpreted programming language known for its simple and readable syntax. 
                It uses indentation to define code blocks and is dynamically typed. Python is widely used for web development, 
                data science, artificial intelligence, and automation. The language emphasizes code readability and allows 
                developers to express concepts in fewer lines of code compared to languages like C++ or Java.""",
            },
            {
                "document_name": "JavaScript Fundamentals.txt",
                "document_content": """JavaScript is a high-level, interpreted programming language primarily used for web development.
                It uses curly braces to define code blocks and supports both functional and object-oriented programming paradigms.
                JavaScript runs in web browsers and on servers through Node.js. It is event-driven and non-blocking, making it
                excellent for handling asynchronous operations and real-time applications.""",
            },
            {
                "document_name": "Performance Comparison.txt",
                "document_content": """In terms of performance, JavaScript typically executes faster than Python due to its V8 engine
                optimization and just-in-time compilation. Python's interpreted nature and dynamic typing can lead to slower
                execution times, especially for computationally intensive tasks. However, Python's extensive library ecosystem
                and C extensions can help mitigate performance issues for specific use cases.""",
            },
        ]

        response = await citations_tool(
            question,
            instructions,
            documents,
            "claude-3-7-sonnet-latest",
            LLMProvider.ANTHROPIC,
        )

        # Check response structure
        self.assertIsInstance(response, list)
        self.assertGreater(len(response), 0)

        # Check that response mentions both languages
        response_text = " ".join(
            [block.get("text", "") for block in response if block.get("type") == "text"]
        ).lower()
        self.assertIn("python", response_text)
        self.assertIn("javascript", response_text)

        # Check for citations
        has_citations = False
        for block in response:
            if block.get("type") == "text" and block.get("citations"):
                has_citations = True
                break
        self.assertTrue(has_citations, "Response should include citations")

    @pytest.mark.asyncio
    async def test_openai_citations(self):
        if self.skip_openai:
            self.skipTest("OPENAI_API_KEY not set")

        question = "What are the main benefits of renewable energy?"
        instructions = "Provide a detailed answer with proper citations from the provided documents."
        documents = [
            {
                "document_name": "Solar Energy Benefits.txt",
                "document_content": """Solar energy offers numerous environmental and economic benefits. It produces clean electricity
                without greenhouse gas emissions during operation, helping combat climate change. Solar installations can reduce
                electricity bills and provide energy independence. The technology has become increasingly cost-effective with
                falling panel prices and improved efficiency.""",
            },
            {
                "document_name": "Wind Power Advantages.txt",
                "document_content": """Wind power is one of the fastest-growing renewable energy sources globally. It generates
                electricity without air pollution or water consumption during operation. Wind farms can be built on land or
                offshore, providing flexibility in deployment. The technology creates jobs in manufacturing, installation,
                and maintenance sectors.""",
            },
            {
                "document_name": "Renewable Energy Economics.txt",
                "document_content": """Renewable energy sources have become increasingly competitive with fossil fuels in terms of cost.
                The levelized cost of electricity from renewables has decreased significantly over the past decade. Investment
                in renewable energy infrastructure stimulates economic growth and reduces dependence on volatile fossil fuel markets.""",
            },
        ]

        response = await citations_tool(
            question, instructions, documents, "gpt-4o", LLMProvider.OPENAI
        )

        # Check response structure
        self.assertIsInstance(response, list)
        self.assertGreater(len(response), 0)

        # Check that all blocks have required structure
        for block in response:
            self.assertIn("type", block)
            self.assertEqual(block["type"], "text")
            self.assertIn("text", block)
            self.assertIn("citations", block)
            self.assertIsInstance(block["citations"], list)

    @pytest.mark.asyncio
    async def test_no_relevant_documents_anthropic(self):
        if self.skip_anthropic:
            self.skipTest("ANTHROPIC_API_KEY not set")

        question = "What is the capital of Mars?"
        instructions = "Answer the question based only on the provided documents. If the information is not available, say 'I don't know'."
        documents = [
            {
                "document_name": "Earth Geography.txt",
                "document_content": "London is the capital of England. Paris is the capital of France. Berlin is the capital of Germany.",
            },
            {
                "document_name": "Ocean Facts.txt",
                "document_content": "The Pacific Ocean is the largest ocean on Earth. The Atlantic Ocean separates Europe and Africa from the Americas.",
            },
        ]

        response = await citations_tool(
            question,
            instructions,
            documents,
            "claude-3-7-sonnet-latest",
            LLMProvider.ANTHROPIC,
        )

        # Check that response indicates lack of information
        response_text = " ".join(
            [block.get("text", "") for block in response if block.get("type") == "text"]
        ).lower()
        self.assertTrue(
            "don't know" in response_text
            or "not" in response_text
            or "no information" in response_text,
            "Response should indicate lack of information about Mars capital",
        )

    def test_unsupported_provider(self):
        question = "Test question"
        instructions = "Test instructions"
        documents = [{"document_name": "test.txt", "document_content": "test content"}]

        with self.assertRaises(ValueError) as context:
            # Using asyncio.run since this should fail immediately
            import asyncio

            asyncio.run(
                citations_tool(
                    question,
                    instructions,
                    documents,
                    "test-model",
                    "unsupported_provider",
                )
            )

        self.assertIn("not supported", str(context.exception))


if __name__ == "__main__":
    unittest.main()
