"""
Utilities for handling images across different LLM providers.
"""

import base64
from typing import Dict, Any, List, Tuple
import httpx
from .exceptions import ProviderError


def validate_image_content(image_data: Dict[str, Any]) -> None:
    """
    Validate image content format and size.

    Args:
        image_data: Image content dictionary

    Raises:
        ProviderError: If image format is invalid
    """
    if image_data.get("type") not in ["image", "image_url"]:
        raise ProviderError(
            "validation", f"Invalid image type: {image_data.get('type')}"
        )

    # Validate based on type
    if image_data.get("type") == "image":
        source = image_data.get("source", {})
        if source.get("type") == "base64":
            if not source.get("data"):
                raise ProviderError("validation", "Base64 image data is required")
            if not source.get("media_type"):
                raise ProviderError(
                    "validation", "Media type is required for base64 images"
                )
        elif source.get("type") == "url":
            if not source.get("url"):
                raise ProviderError("validation", "URL is required for URL images")
        else:
            raise ProviderError(
                "validation", f"Invalid image source type: {source.get('type')}"
            )
    elif image_data.get("type") == "image_url":
        image_url = image_data.get("image_url", {})
        if not image_url.get("url"):
            raise ProviderError("validation", "URL is required for image_url type")


def download_image_from_url(url: str) -> Tuple[bytes, str]:
    """
    Download image from URL and return data and mime type.

    Args:
        url: Image URL

    Returns:
        Tuple of (image_bytes, mime_type)

    Raises:
        ProviderError: If download fails
    """
    # Validate URL scheme and host
    if not url.startswith(("http://", "https://")):
        raise ProviderError("download", "Only HTTP/HTTPS URLs are allowed")

    # Block private/local addresses
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(url)

    try:
        ip = ipaddress.ip_address(parsed.hostname)
        if ip.is_private or ip.is_loopback:
            raise ProviderError("download", "Private/local URLs not allowed")
    except ValueError:
        pass  # Hostname, not IP

    try:
        with httpx.Client() as client:
            response = client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()

            # Get mime type from content-type header
            content_type = response.headers.get("content-type", "image/jpeg")
            mime_type = content_type.split(";")[0].strip()

            # Validate it's an image
            if not mime_type.startswith("image/"):
                raise ProviderError(
                    "download", f"URL returned non-image content type: {mime_type}"
                )

            MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB limit

            if len(response.content) > MAX_IMAGE_SIZE:
                raise ProviderError(
                    "download", f"Image too large: {len(response.content)} bytes"
                )

            return response.content, mime_type
    except Exception as e:
        raise ProviderError("download", f"Failed to download image from URL: {str(e)}")


def extract_media_type_from_data_url(data_url: str) -> str:
    """
    Extract media type from a data URL.

    Args:
        data_url: Data URL string

    Returns:
        Media type (e.g., "image/jpeg")
    """
    if not data_url.startswith("data:"):
        return "image/jpeg"

    header = data_url.split(",", 1)[0]
    if "image/png" in header:
        return "image/png"
    elif "image/gif" in header:
        return "image/gif"
    elif "image/webp" in header:
        return "image/webp"
    else:
        return "image/jpeg"


def convert_to_anthropic_format(content: Any) -> Any:
    """
    Convert message content to Anthropic format.

    Args:
        content: String or list of content blocks

    Returns:
        Content in Anthropic format
    """
    if isinstance(content, str):
        return content

    anthropic_content = []
    for block in content:
        if block.get("type") in ["image", "image_url"]:
            validate_image_content(block)

            if block.get("type") == "image":
                # Already in Anthropic format
                anthropic_content.append(block)
            elif block.get("type") == "image_url":
                # Convert from OpenAI format
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Extract base64 data from data URL
                    parts = url.split(",", 1)
                    if len(parts) == 2:
                        data = parts[1]
                        media_type = extract_media_type_from_data_url(url)

                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                else:
                    # Regular URL
                    anthropic_content.append(
                        {"type": "image", "source": {"type": "url", "url": url}}
                    )
        else:
            anthropic_content.append(block)

    return anthropic_content


def convert_to_openai_format(content: Any) -> Any:
    """
    Convert message content to OpenAI format.

    Args:
        content: String or list of content blocks

    Returns:
        Content in OpenAI format
    """
    if isinstance(content, str):
        return content

    openai_content = []
    for block in content:
        if block.get("type") in ["image", "image_url"]:
            validate_image_content(block)

            if block.get("type") == "image_url":
                # Already in OpenAI format
                openai_content.append(block)
            elif block.get("type") == "image":
                # Convert from Anthropic format
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    url = f"data:{media_type};base64,{data}"
                    openai_content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                elif source.get("type") == "url":
                    openai_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": source.get("url", "")},
                        }
                    )
        else:
            openai_content.append(block)

    return openai_content


def convert_to_gemini_parts(content: Any, genai_types) -> List[Any]:
    """
    Convert message content to Gemini Part objects.

    Args:
        content: String or list of content blocks
        genai_types: Google genai types module

    Returns:
        List of Gemini Part objects
    """
    parts = []

    if isinstance(content, str):
        parts.append(genai_types.Part.from_text(text=content))
        return parts

    for block in content:
        if block.get("type") in ["image", "image_url"]:
            validate_image_content(block)

            # Get image data and mime type
            image_data = None
            mime_type = "image/jpeg"

            if block.get("type") == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    image_data = base64.b64decode(source.get("data", ""))
                    mime_type = source.get("media_type", "image/jpeg")
                elif source.get("type") == "url":
                    image_data, mime_type = download_image_from_url(
                        source.get("url", "")
                    )
            elif block.get("type") == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Extract from data URL
                    parts_split = url.split(",", 1)
                    if len(parts_split) == 2:
                        data = parts_split[1]
                        image_data = base64.b64decode(data)
                        mime_type = extract_media_type_from_data_url(url)
                else:
                    # Download from URL
                    image_data, mime_type = download_image_from_url(url)

            if image_data:
                parts.append(
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            data=image_data, mime_type=mime_type
                        )
                    )
                )
        else:
            parts.append(block)
    return parts
