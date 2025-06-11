"""
Utilities for detecting and handling image data in tool results.

This module provides functionality to detect when tool results contain
image data and convert them to appropriate message formats for LLM providers.
"""

import base64
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path


def detect_image_in_result(result: Any, image_keys: Optional[List[str]] = None) -> Tuple[str, Optional[str]]:
    """
    Detect if a tool result contains image data.
    
    Args:
        result: The tool result to examine
        image_keys: List of keys to check for image data. If None, uses default keys.
    
    Returns:
        Tuple of (text_result, base64_image_data)
        - text_result: String representation for the tool result
        - base64_image_data: Base64-encoded image if found, None otherwise
    """
    text_result = str(result)
    base64_image = None
    
    # Only use provided image keys - no defaults to avoid conflicts
    if image_keys is None or len(image_keys) == 0:
        return text_result, None
    
    # Case 1: Result is a dict-like object with image fields
    if hasattr(result, '__dict__') or isinstance(result, dict):
        obj = result.__dict__ if hasattr(result, '__dict__') else result
        
        # Look for specified image field names
        for field in image_keys:
            if field in obj and obj[field]:
                base64_image = obj[field]
                break
        
        # Update text result to remove image data (keep it clean)
        if base64_image and hasattr(result, 'result'):
            text_result = str(result.result)
        elif base64_image and 'result' in obj:
            text_result = str(obj['result'])
    
    # Case 2: Result string contains base64 image marker
    if not base64_image and isinstance(result, str):
        # Look for base64 image patterns in the string
        base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
        match = re.search(base64_pattern, text_result)
        if match:
            base64_image = match.group(1)
            # Remove the data URL from text result
            text_result = re.sub(base64_pattern, '[Image data]', text_result)
        else:
            # Look for standalone base64 patterns (very long alphanumeric strings)
            standalone_b64_pattern = r'\b([A-Za-z0-9+/]{100,}={0,2})\b'
            match = re.search(standalone_b64_pattern, text_result)
            if match and _is_likely_image_base64(match.group(1)):
                base64_image = match.group(1)
                text_result = text_result.replace(match.group(1), '[Image data]')
    
    return text_result, base64_image


def _load_image_from_path(path: str) -> Optional[str]:
    """Load image from file path and convert to base64."""
    try:
        file_path = Path(path)
        if file_path.exists() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
    except Exception:
        pass
    return None


def _is_likely_image_base64(data: str) -> bool:
    """Heuristic to determine if a base64 string is likely an image."""
    if len(data) < 100:  # Too short to be an image
        return False
    
    try:
        # Try to decode and check if it starts with common image headers
        decoded = base64.b64decode(data[:100])  # Just check first part
        
        # Common image file signatures
        image_signatures = [
            b'\x89PNG',  # PNG
            b'\xff\xd8\xff',  # JPEG
            b'GIF8',  # GIF
            b'RIFF',  # WEBP (RIFF container)
            b'BM',  # BMP
        ]
        
        return any(decoded.startswith(sig) for sig in image_signatures)
    except Exception:
        return False


def create_image_message(image_base64: str, description: str = "Tool generated image") -> Dict[str, Any]:
    """
    Create a message with image content for Anthropic's format.
    
    Args:
        image_base64: Base64-encoded image data
        description: Description of the image
    
    Returns:
        Message dict in Anthropic format
    """
    # Determine media type from base64 data
    media_type = "image/png"  # Default
    try:
        decoded = base64.b64decode(image_base64[:100])
        if decoded.startswith(b'\xff\xd8\xff'):
            media_type = "image/jpeg"
        elif decoded.startswith(b'GIF8'):
            media_type = "image/gif"
        elif decoded.startswith(b'RIFF'):
            media_type = "image/webp"
    except Exception:
        pass
    
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": description
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64
                }
            }
        ]
    }


def process_tool_results_with_images(
    tool_call_blocks: List[Any], 
    results: List[Any],
    image_keys: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process tool results and extract images, returning both tool results and image messages.
    
    Args:
        tool_call_blocks: List of tool call blocks from the LLM response
        results: List of tool execution results
        image_keys: List of keys to check for image data in results
    
    Returns:
        Tuple of (tool_results_content, image_messages)
    """
    tool_results_content = []
    image_messages = []
    
    for tool_call_block, result in zip(tool_call_blocks, results):
        tool_id = tool_call_block.id
        tool_name = tool_call_block.name
        
        # Detect and extract image data
        text_result, image_base64 = detect_image_in_result(result, image_keys)
        
        # Create tool result content (always text)
        tool_results_content.append({
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": text_result,
        })
        
        # Create image message if image data was found
        if image_base64:
            image_message = create_image_message(
                image_base64, 
                f"Image generated by {tool_name} tool"
            )
            image_messages.append(image_message)
    
    return tool_results_content, image_messages