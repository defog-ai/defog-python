"""
Tests for image support in tool results.

Tests the enhanced chat_async functionality that detects image data
in tool results and injects them as separate messages.
"""

import pytest
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from typing import Optional
from pydantic import BaseModel, Field

from defog.llm.utils_image_support import (
    detect_image_in_result,
    process_tool_results_with_images,
)


def create_test_image(text: str = "TEST") -> str:
    """Create a simple test image and return as base64."""
    img = Image.new("RGB", (100, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(0, 0, 0))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_multi_part_images() -> list[str]:
    """Create multiple test images to simulate multi-part screenshots."""
    return [create_test_image("PART 1"), create_test_image("PART 2")]


class MockToolResult(BaseModel):
    """Mock tool result for testing."""

    result: str = Field(description="Text result")
    image_base64: Optional[str | list[str]] = Field(
        None, description="Base64 image data - can be single string or list of strings"
    )
    other_field: Optional[str] = Field(None, description="Other data")


class MockToolBlock:
    """Mock tool call block for testing."""

    def __init__(self, tool_id: str, name: str):
        self.id = tool_id
        self.name = name


class TestImageDetection:
    """Test image detection in tool results."""

    def test_no_image_keys_specified(self):
        """Test that no image is detected when no keys are specified."""
        result = MockToolResult(result="test", image_base64=create_test_image())

        # No keys specified
        text, image = detect_image_in_result(result, None)
        assert image is None
        assert text == str(result)

        # Empty keys list
        text, image = detect_image_in_result(result, [])
        assert image is None
        assert text == str(result)

    def test_correct_image_key(self):
        """Test that image is detected with correct key."""
        image_data = create_test_image()
        result = MockToolResult(result="test", image_base64=image_data)

        text, image = detect_image_in_result(result, ["image_base64"])
        assert image == image_data
        assert text == "test"  # Should return the cleaned result field

    def test_wrong_image_key(self):
        """Test that image is not detected with wrong key."""
        result = MockToolResult(result="test", image_base64=create_test_image())

        text, image = detect_image_in_result(result, ["wrong_key"])
        assert image is None
        assert text == str(result)

    def test_multiple_keys(self):
        """Test detection with multiple possible keys."""
        image_data = create_test_image()
        result = MockToolResult(result="test", image_base64=image_data)

        text, image = detect_image_in_result(
            result, ["wrong_key", "image_base64", "other_key"]
        )
        assert image == image_data

    def test_no_image_field(self):
        """Test tool result without image field."""
        result = MockToolResult(result="test", other_field="data")

        text, image = detect_image_in_result(result, ["image_base64"])
        assert image is None
        assert text == str(result)

    def test_empty_image_field(self):
        """Test tool result with empty image field."""
        result = MockToolResult(result="test", image_base64=None)

        text, image = detect_image_in_result(result, ["image_base64"])
        assert image is None
        assert text == str(result)

    def test_multi_part_image_detection(self):
        """Test detection of multi-part images (list of base64 strings)."""
        image_list = create_multi_part_images()
        result = MockToolResult(result="multi-part screenshot", image_base64=image_list)

        text, image = detect_image_in_result(result, ["image_base64"])
        assert image == image_list
        assert text == "multi-part screenshot"

    def test_empty_image_list(self):
        """Test tool result with empty image list."""
        result = MockToolResult(result="test", image_base64=[])

        text, image = detect_image_in_result(result, ["image_base64"])
        assert image is None  # Empty list should be treated as no image
        assert text == str(result)


class TestToolResultProcessing:
    """Test processing of tool results with images."""

    def test_no_images(self):
        """Test processing tool results without images."""
        tool_blocks = [MockToolBlock("1", "test_tool")]
        results = [MockToolResult(result="test result")]

        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )

        assert len(tool_data_list) == 1
        assert tool_data_list[0].tool_id == "1"
        assert tool_data_list[0].tool_name == "test_tool"
        assert tool_data_list[0].tool_result_text == str(results[0])
        assert tool_data_list[0].image_data is None

    def test_single_image(self):
        """Test processing tool results with one image."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "screenshot_tool")]
        results = [MockToolResult(result="took screenshot", image_base64=image_data)]

        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )

        assert len(tool_data_list) == 1
        tool_data = tool_data_list[0]
        
        # Check tool data
        assert tool_data.tool_id == "1"
        assert tool_data.tool_name == "screenshot_tool"
        assert tool_data.tool_result_text == "took screenshot"  # Should return cleaned text
        assert tool_data.image_data == image_data

    def test_mixed_results(self):
        """Test processing mix of results with and without images."""
        image_data = create_test_image()
        tool_blocks = [
            MockToolBlock("1", "text_tool"),
            MockToolBlock("2", "image_tool"),
        ]
        results = [
            MockToolResult(result="text only"),
            MockToolResult(result="with image", image_base64=image_data),
        ]

        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )

        assert len(tool_data_list) == 2
        
        # First tool should have no image
        assert tool_data_list[0].tool_name == "text_tool"
        assert tool_data_list[0].image_data is None
        
        # Second tool should have image
        assert tool_data_list[1].tool_name == "image_tool"
        assert tool_data_list[1].image_data == image_data

    def test_wrong_image_keys(self):
        """Test that wrong image keys don't detect images."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "test_tool")]
        results = [MockToolResult(result="test", image_base64=image_data)]

        # Use wrong key
        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["wrong_key"]
        )

        assert len(tool_data_list) == 1
        assert tool_data_list[0].image_data is None  # No image detected

    def test_no_image_keys(self):
        """Test that no image keys means no image detection."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "test_tool")]
        results = [MockToolResult(result="test", image_base64=image_data)]

        # No keys specified
        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, None
        )

        assert len(tool_data_list) == 1
        assert tool_data_list[0].image_data is None  # No image detected

    def test_multi_part_screenshot(self):
        """Test processing tool results with multi-part images."""
        image_list = create_multi_part_images()
        tool_blocks = [MockToolBlock("1", "screenshot_tool")]
        results = [
            MockToolResult(
                result="Page was 15000px tall - split into 2 screenshot segments",
                image_base64=image_list,
            )
        ]

        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )

        assert len(tool_data_list) == 1
        tool_data = tool_data_list[0]
        
        # Check tool data
        assert tool_data.tool_id == "1"
        assert tool_data.tool_name == "screenshot_tool"
        assert "split into 2 screenshot segments" in tool_data.tool_result_text
        
        # Check that image data is the list of images
        assert tool_data.image_data == image_list
        assert isinstance(tool_data.image_data, list)
        assert len(tool_data.image_data) == 2


class TestProviderMessageCreation:
    """Test provider-specific image message creation."""

    def test_anthropic_image_message_creation(self):
        """Test that Anthropic provider creates correct image messages."""
        from defog.llm.providers.anthropic_provider import AnthropicProvider
        
        provider = AnthropicProvider(api_key="test")
        image_data = create_test_image()
        
        # Test single image
        msg = provider.create_image_message(image_data, "Test image")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Test image"
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["source"]["type"] == "base64"
        assert msg["content"][1]["source"]["data"] == image_data
        
        # Test multiple images
        image_list = create_multi_part_images()
        msg = provider.create_image_message(image_list, "Multi-part screenshot")
        assert len(msg["content"]) == 3  # text + 2 images
        assert msg["content"][1]["source"]["data"] == image_list[0]
        assert msg["content"][2]["source"]["data"] == image_list[1]

    def test_openai_image_message_creation(self):
        """Test that OpenAI provider creates correct image messages."""
        from defog.llm.providers.openai_provider import OpenAIProvider
        
        provider = OpenAIProvider(api_key="test")
        image_data = create_test_image()
        
        # Test single image
        msg = provider.create_image_message(image_data, "Test image")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "input_text"
        assert msg["content"][0]["text"] == "Test image"
        assert msg["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in msg["content"][1]["image_url"]["url"]
        assert image_data in msg["content"][1]["image_url"]["url"]
        
        # Test multiple images
        image_list = create_multi_part_images()
        msg = provider.create_image_message(image_list, "Multi-part screenshot")
        assert len(msg["content"]) == 3  # text + 2 images
        for i, img_data in enumerate(image_list):
            assert img_data in msg["content"][i+1]["image_url"]["url"]

    def test_gemini_image_message_creation(self):
        """Test that Gemini provider creates correct image messages."""
        from defog.llm.providers.gemini_provider import GeminiProvider
        
        provider = GeminiProvider(api_key="test")
        image_data = create_test_image()
        
        # Test single image
        msg = provider.create_image_message(image_data, "Test image")
        # Gemini returns a Content object, not a dict
        assert msg.role == "user"
        assert len(msg.parts) == 2
        assert msg.parts[0].text == "Test image"
        assert hasattr(msg.parts[1], 'inline_data')  # Image part
        assert msg.parts[1].inline_data.mime_type == "image/png"
        
        # Verify the bytes data is correct
        import base64
        expected_bytes = base64.b64decode(image_data)
        assert msg.parts[1].inline_data.data == expected_bytes
