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

from defog.llm.utils_image_support import detect_image_in_result, process_tool_results_with_images


def create_test_image() -> str:
    """Create a simple test image and return as base64."""
    img = Image.new('RGB', (100, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "TEST", fill=(0, 0, 0))
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class MockToolResult(BaseModel):
    """Mock tool result for testing."""
    result: str = Field(description="Text result")
    image_base64: Optional[str] = Field(None, description="Base64 image data")
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
        
        text, image = detect_image_in_result(result, ["wrong_key", "image_base64", "other_key"])
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


class TestToolResultProcessing:
    """Test processing of tool results with images."""
    
    def test_no_images(self):
        """Test processing tool results without images."""
        tool_blocks = [MockToolBlock("1", "test_tool")]
        results = [MockToolResult(result="test result")]
        
        tool_results, image_messages = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )
        
        assert len(tool_results) == 1
        assert len(image_messages) == 0
        assert tool_results[0]["content"] == str(results[0])
    
    def test_single_image(self):
        """Test processing tool results with one image."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "screenshot_tool")]
        results = [MockToolResult(result="took screenshot", image_base64=image_data)]
        
        tool_results, image_messages = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )
        
        assert len(tool_results) == 1
        assert len(image_messages) == 1
        
        # Check tool result
        assert tool_results[0]["tool_use_id"] == "1"
        assert tool_results[0]["content"] == "took screenshot"  # Should return cleaned text
        
        # Check image message
        image_msg = image_messages[0]
        assert image_msg["role"] == "user"
        assert len(image_msg["content"]) == 2
        assert image_msg["content"][0]["type"] == "text"
        assert image_msg["content"][1]["type"] == "image"
        assert image_msg["content"][1]["source"]["data"] == image_data
    
    def test_mixed_results(self):
        """Test processing mix of results with and without images."""
        image_data = create_test_image()
        tool_blocks = [
            MockToolBlock("1", "text_tool"),
            MockToolBlock("2", "image_tool")
        ]
        results = [
            MockToolResult(result="text only"),
            MockToolResult(result="with image", image_base64=image_data)
        ]
        
        tool_results, image_messages = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )
        
        assert len(tool_results) == 2
        assert len(image_messages) == 1  # Only one image
        
        # Only second tool should generate image message
        image_msg = image_messages[0]
        assert "image_tool" in image_msg["content"][0]["text"]
    
    def test_wrong_image_keys(self):
        """Test that wrong image keys don't detect images."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "test_tool")]
        results = [MockToolResult(result="test", image_base64=image_data)]
        
        # Use wrong key
        tool_results, image_messages = process_tool_results_with_images(
            tool_blocks, results, ["wrong_key"]
        )
        
        assert len(tool_results) == 1
        assert len(image_messages) == 0  # No image detected
    
    def test_no_image_keys(self):
        """Test that no image keys means no image detection."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "test_tool")]
        results = [MockToolResult(result="test", image_base64=image_data)]
        
        # No keys specified
        tool_results, image_messages = process_tool_results_with_images(
            tool_blocks, results, None
        )
        
        assert len(tool_results) == 1
        assert len(image_messages) == 0  # No image detected


class TestImageMessageFormat:
    """Test the format of generated image messages."""
    
    def test_image_message_structure(self):
        """Test that image messages have correct Anthropic format."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "screenshot_tool")]
        results = [MockToolResult(result="screenshot taken", image_base64=image_data)]
        
        _, image_messages = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )
        
        assert len(image_messages) == 1
        msg = image_messages[0]
        
        # Check overall structure
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        
        # Check text part
        text_part = msg["content"][0]
        assert text_part["type"] == "text"
        assert "screenshot_tool" in text_part["text"]
        
        # Check image part
        image_part = msg["content"][1]
        assert image_part["type"] == "image"
        assert image_part["source"]["type"] == "base64"
        assert image_part["source"]["media_type"] == "image/png"
        assert image_part["source"]["data"] == image_data