import unittest
import base64
import os
from defog.llm.image_utils import (
    validate_image_content,
    download_image_from_url,
    convert_to_anthropic_format,
    convert_to_openai_format,
    convert_to_gemini_parts,
)
from defog.llm.exceptions import ProviderError


class TestImageUtils(unittest.TestCase):
    def setUp(self):
        # Real image URLs for testing
        self.real_image_urls = [
            "https://www.gstatic.com/webp/gallery/1.jpg",
            "https://www.gstatic.com/webp/gallery/4.jpg",
        ]

        # Create a test image file path (we'll use the example.png in the repo)
        self.test_image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "example.png"
        )

        # If example.png exists, create a data URL from it
        if os.path.exists(self.test_image_path):
            with open(self.test_image_path, "rb") as f:
                image_data = f.read()
                self.local_base64 = base64.b64encode(image_data).decode("utf-8")
                self.data_url_png = f"data:image/png;base64,{self.local_base64}"
        else:
            # Fallback to downloading an image to create data URL
            image_bytes, mime_type = download_image_from_url(self.real_image_urls[0])
            self.local_base64 = base64.b64encode(image_bytes).decode("utf-8")
            self.data_url_png = f"data:{mime_type};base64,{self.local_base64}"

    def test_validate_image_content_valid(self):
        # Test valid image content with data URL
        content = {"type": "image_url", "image_url": {"url": self.data_url_png}}
        # Should not raise any exception
        validate_image_content(content)

        # Test valid image content with HTTP URL
        content = {"type": "image_url", "image_url": {"url": self.real_image_urls[0]}}
        # Should not raise any exception
        validate_image_content(content)

    def test_validate_image_content_invalid(self):
        # Test missing type
        content = {"image_url": {"url": self.data_url_png}}
        with self.assertRaises(ProviderError):
            validate_image_content(content)

        # Test invalid type
        content = {"type": "video", "image_url": {"url": self.data_url_png}}
        with self.assertRaises(ProviderError):
            validate_image_content(content)

        # Test missing image_url
        content = {"type": "image_url"}
        with self.assertRaises(ProviderError):
            validate_image_content(content)

        # Test missing url in image_url
        content = {"type": "image_url", "image_url": {}}
        with self.assertRaises(ProviderError):
            validate_image_content(content)

    def test_download_image_from_url_real(self):
        # Test downloading real images
        for url in self.real_image_urls:
            image_bytes, mime_type = download_image_from_url(url)

            # Check that we got valid data
            self.assertIsInstance(mime_type, str)
            self.assertTrue(mime_type.startswith("image/"))
            self.assertIsInstance(image_bytes, bytes)
            self.assertGreater(len(image_bytes), 0)

    def test_download_image_from_url_invalid(self):
        # Test with invalid URL
        with self.assertRaises(ProviderError):
            download_image_from_url("https://example.com/nonexistent-image-12345.jpg")

    def test_convert_to_anthropic_format(self):
        # Test with data URL - convert functions expect a list of content blocks
        content = [{"type": "image_url", "image_url": {"url": self.data_url_png}}]
        result = convert_to_anthropic_format(content)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "image")
        self.assertEqual(result[0]["source"]["type"], "base64")
        self.assertTrue(result[0]["source"]["media_type"].startswith("image/"))
        self.assertEqual(result[0]["source"]["data"], self.local_base64)

        # Test with already correct format
        anthropic_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": self.local_base64,
                },
            }
        ]
        result = convert_to_anthropic_format(anthropic_content)
        self.assertEqual(result, anthropic_content)

    def test_convert_to_anthropic_format_with_url(self):
        # Test with real URL
        content = [{"type": "image_url", "image_url": {"url": self.real_image_urls[0]}}]
        result = convert_to_anthropic_format(content)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "image")
        self.assertEqual(result[0]["source"]["type"], "url")
        self.assertEqual(result[0]["source"]["url"], self.real_image_urls[0])

    def test_convert_to_openai_format(self):
        # Test with Anthropic format
        anthropic_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": self.local_base64,
                },
            }
        ]
        result = convert_to_openai_format(anthropic_content)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "image_url")
        self.assertTrue(result[0]["image_url"]["url"].startswith("data:image/"))
        self.assertIn(self.local_base64, result[0]["image_url"]["url"])

        # Test with already correct format
        openai_content = [
            {"type": "image_url", "image_url": {"url": self.data_url_png}}
        ]
        result = convert_to_openai_format(openai_content)
        self.assertEqual(result, openai_content)

    def test_convert_to_gemini_parts_with_data_url(self):
        # Test with data URL
        # Import genai types for testing
        try:
            from google import genai

            genai_types = genai.types
        except ImportError:
            self.skipTest("Google genai not installed")

        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": self.data_url_png}},
        ]

        parts = convert_to_gemini_parts(content, genai_types)

        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].text, "What's in this image?")
        self.assertTrue(parts[1].inline_data.mime_type.startswith("image/"))
        self.assertEqual(parts[1].inline_data.data, base64.b64decode(self.local_base64))

    def test_convert_to_gemini_parts_with_real_url(self):
        # Test with real URL
        # Import genai types for testing
        try:
            from google import genai

            genai_types = genai.types
        except ImportError:
            self.skipTest("Google genai not installed")

        content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": self.real_image_urls[0]}},
        ]

        parts = convert_to_gemini_parts(content, genai_types)

        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].text, "Describe this image")
        self.assertIn("image/", parts[1].inline_data.mime_type)
        self.assertIsInstance(parts[1].inline_data.data, bytes)
        self.assertGreater(len(parts[1].inline_data.data), 0)

    def test_convert_to_gemini_parts_text_only(self):
        # Test with text-only message
        # Import genai types for testing
        try:
            from google import genai

            genai_types = genai.types
        except ImportError:
            self.skipTest("Google genai not installed")

        content = "Just text"

        parts = convert_to_gemini_parts(content, genai_types)

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].text, "Just text")

    def test_convert_to_gemini_parts_multiple_images(self):
        # Test with multiple images
        # Import genai types for testing
        try:
            from google import genai

            genai_types = genai.types
        except ImportError:
            self.skipTest("Google genai not installed")

        content = [
            {"type": "text", "text": "Compare these images:"},
            {"type": "image_url", "image_url": {"url": self.real_image_urls[0]}},
            {"type": "image_url", "image_url": {"url": self.real_image_urls[1]}},
        ]

        parts = convert_to_gemini_parts(content, genai_types)

        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0].text, "Compare these images:")
        self.assertIn("image/", parts[1].inline_data.mime_type)
        self.assertIn("image/", parts[2].inline_data.mime_type)


if __name__ == "__main__":
    unittest.main()
