"""
Utilities for detecting and handling image data in tool results.

This module provides functionality to detect when tool results contain
image data and convert them to appropriate message formats for LLM providers.
"""

import base64
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, NamedTuple
from pathlib import Path


def detect_image_in_result(
    result: Any, image_keys: Optional[List[str]] = None
) -> Tuple[str, Optional[Union[str, List[str]]]]:
    """
    Detect if a tool result contains image data.

    Args:
        result: The tool result to examine
        image_keys: List of keys to check for image data. If None, uses default keys.

    Returns:
        Tuple of (text_result, base64_image_data)
        - text_result: String representation for the tool result
        - base64_image_data: Base64-encoded image(s) if found - can be single string or list of strings, None otherwise
    """
    text_result = str(result)
    base64_image = None

    # Only use provided image keys - no defaults to avoid conflicts
    if image_keys is None or len(image_keys) == 0:
        return text_result, None

    # Case 1: Result is a dict-like object with image fields
    if hasattr(result, "__dict__") or isinstance(result, dict):
        obj = result.__dict__ if hasattr(result, "__dict__") else result

        # Look for specified image field names
        for field in image_keys:
            if field in obj and obj[field]:
                # Check if it's a list and handle empty lists
                if isinstance(obj[field], list):
                    if len(obj[field]) > 0:
                        base64_image = obj[field]
                    # Empty list is treated as no image
                else:
                    base64_image = obj[field]
                break

        # Update text result to remove image data (keep it clean)
        if base64_image and hasattr(result, "result"):
            text_result = str(result.result)
        elif base64_image and "result" in obj:
            text_result = str(obj["result"])

    return text_result, base64_image


class ToolResultData(NamedTuple):
    """Structured data for a single tool result."""

    tool_id: Optional[str]
    tool_name: str
    tool_result_text: str
    image_data: Optional[Union[str, List[str]]]


def process_tool_results_with_images(
    tool_call_blocks: List[Any],
    results: List[Any],
    image_keys: Optional[List[str]] = None,
) -> List[ToolResultData]:
    """
    Process tool results and extract images, returning structured data for each tool.

    This function only extracts and structures the data - it does NOT create provider-specific
    image messages. That's the provider's responsibility.

    Args:
        tool_call_blocks: List of tool call blocks from the LLM response
        results: List of tool execution results
        image_keys: List of keys to check for image data in results

    Returns:
        List of ToolResultData objects, one per tool
    """
    processed_tools = []

    for tool_call_block, result in zip(tool_call_blocks, results):
        tool_id = getattr(tool_call_block, "id", None)
        tool_name = getattr(tool_call_block, "name", "unknown_tool")

        # Detect and extract image data
        text_result, image_base64 = detect_image_in_result(result, image_keys)

        processed_tools.append(
            ToolResultData(
                tool_id=tool_id,
                tool_name=tool_name,
                tool_result_text=text_result,
                image_data=image_base64,
            )
        )

    return processed_tools
