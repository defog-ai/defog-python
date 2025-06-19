"""
Utilities for detecting and handling image data in tool results.

This module provides functionality to detect when tool results contain
image data and convert them to appropriate message formats for LLM providers.
"""

import base64
import logging
from typing import Any, List, Optional, Tuple, Union, NamedTuple

# Constants for image validation
MAX_IMAGE_SIZE_MB = 20  # Maximum image size in MB
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}
DEFAULT_TEXT_FIELDS = ["result", "text", "content", "output", "response"]

# Text length limit to prevent misidentifying base64-encoded data as regular text.
# Base64-encoded images typically exceed 10K characters even for small images.
# This helps distinguish between actual text content and encoded binary data.
TEXT_LENGTH_LIMIT = 10000  # Maximum length for text fields to avoid encoded data

logger = logging.getLogger(__name__)


def validate_base64_image(image_data: str) -> Tuple[bool, Optional[str]]:
    """
    Validate base64 image data for security and format compliance.

    Args:
        image_data: Base64-encoded image string (with or without data URI prefix)

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes
        - error_message: Description of validation failure, None if valid
    """
    if not image_data or not isinstance(image_data, str):
        return False, "Image data must be a non-empty string"

    try:
        # Handle data URI prefix
        clean_base64 = image_data
        media_type = None

        if image_data.startswith("data:"):
            try:
                header, clean_base64 = image_data.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
            except (ValueError, IndexError):
                return False, "Malformed data URI prefix"

        # Validate base64 format
        try:
            decoded_data = base64.b64decode(clean_base64, validate=True)
        except Exception:
            return False, "Invalid base64 encoding"

        # Check size limits
        if len(decoded_data) > MAX_IMAGE_SIZE_BYTES:
            return False, f"Image size exceeds {MAX_IMAGE_SIZE_MB}MB limit"

        # Validate image format by checking magic bytes
        detected_format = detect_image_format(decoded_data)
        if not detected_format:
            return False, "Unrecognized image format"

        # If media type was specified in data URI, validate consistency
        if media_type and media_type != detected_format:
            logger.warning(
                f"Media type mismatch: URI says {media_type}, detected {detected_format}"
            )

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def detect_image_format(image_bytes: bytes) -> Optional[str]:
    """
    Detect image format from binary data using magic bytes.

    Args:
        image_bytes: Raw image binary data

    Returns:
        MIME type string if recognized format, None otherwise
    """
    if len(image_bytes) < 12:
        return None

    # Check magic bytes for common formats
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif image_bytes.startswith(b"GIF8"):
        return "image/gif"
    elif (
        image_bytes.startswith(b"RIFF")
        and len(image_bytes) >= 12
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"

    return None


def extract_text_from_result(result: Any, image_found: bool = False) -> str:
    """
    Extract text content from tool result with flexible field detection.

    Args:
        result: The tool result to extract text from
        image_found: Whether image data was found (affects extraction strategy)

    Returns:
        Extracted text content
    """
    # If result is already a string, return it
    if isinstance(result, str):
        return result

    # For object-like results, try multiple approaches
    if hasattr(result, "__dict__") or isinstance(result, dict):
        obj = result.__dict__ if hasattr(result, "__dict__") else result

        # If image was found, prioritize extracting clean text fields
        if image_found:
            for field in DEFAULT_TEXT_FIELDS:
                if field in obj and obj[field] is not None:
                    return str(obj[field])

        # Fallback: look for any string-valued field that's not image data
        for key, value in obj.items():
            if (
                isinstance(value, str)
                and not key.lower().endswith(("_base64", "_image", "_img", "_data"))
                and len(value) < TEXT_LENGTH_LIMIT
            ):  # Avoid very long strings that might be encoded data
                return value

    # Final fallback: string representation
    return str(result)


def detect_image_in_result(
    result: Any, image_keys: Optional[List[str]] = None
) -> Tuple[str, Optional[Union[str, List[str]]]]:
    """
    Detect if a tool result contains image data with validation.

    Args:
        result: The tool result to examine
        image_keys: List of keys to check for image data. If None, uses default keys.

    Returns:
        Tuple of (text_result, base64_image_data)
        - text_result: String representation for the tool result
        - base64_image_data: Base64-encoded image(s) if found and valid - can be single string or list of strings, None otherwise
    """
    # Only use provided image keys - no defaults to avoid conflicts
    if image_keys is None or len(image_keys) == 0:
        return extract_text_from_result(result), None

    base64_image = None
    image_found = False

    # Case 1: Result is a dict-like object with image fields
    if hasattr(result, "__dict__") or isinstance(result, dict):
        obj = result.__dict__ if hasattr(result, "__dict__") else result

        # Look for specified image field names
        for field in image_keys:
            if field in obj and obj[field]:
                candidate_image = obj[field]

                # Handle list of images
                if isinstance(candidate_image, list):
                    if len(candidate_image) > 0:
                        validated_images = []
                        for img in candidate_image:
                            if isinstance(img, str):
                                is_valid, error = validate_base64_image(img)
                                if is_valid:
                                    validated_images.append(img)
                                else:
                                    logger.warning(f"Invalid image in list: {error}")

                        if validated_images:
                            base64_image = validated_images
                            image_found = True

                # Handle single image
                elif isinstance(candidate_image, str):
                    is_valid, error = validate_base64_image(candidate_image)
                    if is_valid:
                        base64_image = candidate_image
                        image_found = True
                    else:
                        logger.warning(f"Invalid image data: {error}")

                if image_found:
                    break

    # Extract text using improved logic
    text_result = extract_text_from_result(result, image_found)

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


def validate_and_process_image_data(
    image_base64: Union[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Validate and process image data for provider use.

    Args:
        image_base64: Base64-encoded image data - can be single string or list of strings

    Returns:
        Tuple of (valid_images, errors)
        - valid_images: List of validated base64 strings
        - errors: List of validation error messages
    """
    if isinstance(image_base64, str):
        image_list = [image_base64]
    else:
        image_list = image_base64

    valid_images = []
    errors = []

    for img_data in image_list:
        if not isinstance(img_data, str):
            errors.append(f"Image data must be string, got {type(img_data)}")
            continue

        is_valid, error = validate_base64_image(img_data)
        if is_valid:
            valid_images.append(img_data)
        else:
            errors.append(error)

    return valid_images, errors


def safe_extract_media_type_and_data(image_base64: str) -> Tuple[str, str]:
    """
    Safely extract media type and clean base64 data from image string.

    Args:
        image_base64: Base64 image string (with or without data URI prefix)

    Returns:
        Tuple of (media_type, clean_base64_data)
    """
    try:
        if image_base64.startswith("data:"):
            header, clean_data = image_base64.split(",", 1)
            media_type = header.split(";")[0].split(":")[1]
            return media_type, clean_data
        else:
            # Detect format from the raw data
            try:
                decoded = base64.b64decode(image_base64, validate=True)
                detected_format = detect_image_format(decoded)
                return detected_format or "image/jpeg", image_base64
            except Exception:
                # Fallback to JPEG if detection fails
                return "image/jpeg", image_base64
    except Exception:
        # Fallback for any parsing errors
        return "image/jpeg", image_base64
