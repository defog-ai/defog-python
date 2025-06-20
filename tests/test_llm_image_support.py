import unittest
import pytest
import os
import base64
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider
from pydantic import BaseModel
from tests.conftest import skip_if_no_api_key


class ImageAnalysis(BaseModel):
    description: str
    objects_count: int
    main_colors: list[str]


class TestLLMImageSupport(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Use real image URLs that are stable and accessible
        self.test_image_urls = [
            "https://www.gstatic.com/webp/gallery/1.jpg",
            "https://www.gstatic.com/webp/gallery/4.jpg",
        ]

        # Load example.png if it exists
        self.example_image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "example.png"
        )
        self.local_image_data_url = None

        if os.path.exists(self.example_image_path):
            with open(self.example_image_path, "rb") as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                self.local_image_data_url = f"data:image/png;base64,{base64_data}"

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_api_key("anthropic")
    async def test_anthropic_image_support(self):
        """Test Anthropic provider with image support"""
        # Test with URL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image? Be very brief."},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.test_image_urls[0]},
                    },
                ],
            }
        ]

        response = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-latest",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsInstance(response.content, str)
        self.assertGreater(len(response.content), 0)

        # Test with local image if available
        if self.local_image_data_url:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one sentence.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self.local_image_data_url},
                        },
                    ],
                }
            ]

            response = await chat_async(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-7-sonnet-latest",
                messages=messages,
                temperature=0.0,
                max_retries=1,
            )

            self.assertIsInstance(response.content, str)
            self.assertGreater(len(response.content), 0)

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_api_key("openai")
    async def test_openai_image_support(self):
        """Test OpenAI provider with image support"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image? Answer in 5 words or less.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.test_image_urls[1]},
                    },
                ],
            }
        ]

        response = await chat_async(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsInstance(response.content, str)
        self.assertGreater(len(response.content), 0)

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_api_key("gemini")
    async def test_gemini_image_support(self):
        """Test Gemini provider with image support"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one short sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.test_image_urls[0]},
                    },
                ],
            }
        ]

        response = await chat_async(
            provider=LLMProvider.GEMINI,
            model="gemini-2.0-flash",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsInstance(response.content, str)
        self.assertGreater(len(response.content), 0)

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_api_key("anthropic")
    async def test_multiple_images(self):
        """Test multiple images in a single message"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "How many images do you see? Just give the number.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.test_image_urls[0]},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.test_image_urls[1]},
                    },
                ],
            }
        ]

        # Test with Anthropic
        response = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-latest",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsInstance(response.content, str)
        self.assertIn(
            "2", response.content.lower() or response.content.lower() == "two"
        )

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_api_key("openai")
    async def test_structured_output_with_images(self):
        """Test structured output with image analysis"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and provide structured data.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.test_image_urls[0]},
                    },
                ],
            }
        ]

        # Test with OpenAI (supports structured output)
        response = await chat_async(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            messages=messages,
            response_format=ImageAnalysis,
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsInstance(response.content, ImageAnalysis)
        self.assertIsInstance(response.content.description, str)
        self.assertIsInstance(response.content.objects_count, int)
        self.assertIsInstance(response.content.main_colors, list)

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_api_key("anthropic")
    async def test_conversation_with_images(self):
        """Test a conversation that includes images"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.test_image_urls[0]},
                    },
                ],
            }
        ]

        # First response
        response1 = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-latest",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )

        # Add assistant response and follow-up
        messages.append({"role": "assistant", "content": response1.content})
        messages.append({"role": "user", "content": "What color is the main object?"})

        response2 = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-latest",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsInstance(response2.content, str)
        self.assertGreater(len(response2.content), 0)


if __name__ == "__main__":
    unittest.main()
