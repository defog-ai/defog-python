"""
Type definitions for multimodal content support in LLM providers.
"""
from typing import Dict, List, Union, Any, Literal, TypedDict
from typing_extensions import NotRequired


class TextContent(TypedDict):
    """Text content block."""
    type: Literal["text"]
    text: str


class ImageSource(TypedDict):
    """Image source information."""
    type: Literal["base64", "url"]
    media_type: NotRequired[str]  # e.g., "image/jpeg", "image/png"
    data: NotRequired[str]  # base64 data for type="base64"
    url: NotRequired[str]  # URL for type="url"


class ImageContent(TypedDict):
    """Image content block."""
    type: Literal["image", "image_url"]  # "image" for Anthropic, "image_url" for OpenAI
    source: NotRequired[ImageSource]  # Anthropic format
    image_url: NotRequired[Dict[str, str]]  # OpenAI format


# Content can be a string (for backward compatibility) or a list of content blocks
ContentBlock = Union[TextContent, ImageContent]
Content = Union[str, List[ContentBlock]]


class Message(TypedDict):
    """Message with multimodal content support."""
    role: str
    content: Content
    name: NotRequired[str]
    tool_calls: NotRequired[List[Any]]
    tool_call_id: NotRequired[str]


def normalize_content(content: Content) -> List[ContentBlock]:
    """
    Normalize content to a list of content blocks.
    
    Args:
        content: String or list of content blocks
        
    Returns:
        List of content blocks
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def is_multimodal_content(content: Content) -> bool:
    """
    Check if content contains multimodal elements (images).
    
    Args:
        content: String or list of content blocks
        
    Returns:
        True if content contains images, False otherwise
    """
    if isinstance(content, str):
        return False
    
    for block in content:
        if block.get("type") in ["image", "image_url"]:
            return True
    
    return False


def extract_text_content(content: Content) -> str:
    """
    Extract only text content from multimodal content.
    
    Args:
        content: String or list of content blocks
        
    Returns:
        Concatenated text content
    """
    if isinstance(content, str):
        return content
    
    text_parts = []
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    
    return "\n".join(text_parts)