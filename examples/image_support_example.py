"""
Example demonstrating image support in chat_async across different providers.
"""

import asyncio
import base64
from pathlib import Path
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider
from pydantic import BaseModel


class ImageAnalysis(BaseModel):
    """Structure for image analysis results."""

    description: str
    main_objects: list[str]
    dominant_colors: list[str]
    image_type: str


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load an image file and convert to base64.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (base64_data, media_type)
    """
    path = Path(image_path)

    # Determine media type from extension
    ext = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    # Read and encode image
    with open(path, "rb") as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode("utf-8")

    return base64_data, media_type


async def analyze_local_image(image_path: str):
    """Analyze a local image file using different providers."""
    print("\n=== Local Image Analysis ===")
    print(f"Image path: {image_path}")

    # Load image
    base64_data, media_type = load_image_as_base64(image_path)

    # Use standardized OpenAI format - it works across all providers
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_data}"},
                },
            ],
        }
    ]

    # Test with different providers
    providers = [
        (LLMProvider.ANTHROPIC, "claude-sonnet-4-20250514"),
        (LLMProvider.OPENAI, "gpt-4.1"),
        (LLMProvider.GEMINI, "gemini-2.5-pro"),
    ]

    for provider, model in providers:
        print(f"\n--- {provider.value} ---")
        try:
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages,
                max_completion_tokens=2000,
            )
            print(f"Response: {response.content}...")
            print(f"Cost: ${response.cost_in_cents / 100:.4f}")
        except Exception as e:
            print(f"Error: {e}")


async def analyze_image_url(image_url: str):
    """Analyze an image from URL using different providers."""
    print("\n=== Image URL Analysis ===")
    print(f"Image URL: {image_url}")

    # Use standardized format that works across all providers
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image from the URL."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    # Test with different providers
    providers = [
        (LLMProvider.ANTHROPIC, "claude-sonnet-4-20250514"),
        (LLMProvider.OPENAI, "gpt-4.1"),
        (LLMProvider.GEMINI, "gemini-2.5-pro"),
    ]

    for provider, model in providers:
        print(f"\n--- {provider.value} ---")
        try:
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages,
                max_completion_tokens=2000,
            )
            print(f"Response: {response.content}...")
            print(f"Cost: ${response.cost_in_cents / 100:.4f}")
        except Exception as e:
            print(f"Error: {e}")


async def structured_image_analysis(image_url: str):
    """Get structured analysis of an image."""
    print("\n=== Structured Image Analysis ===")
    print(f"Image URL: {image_url}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this image and provide a structured response with description, objects, colors, and type.",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    # Get structured response using Anthropic
    response = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        messages=messages,
        response_format=ImageAnalysis,
        max_completion_tokens=2000,
    )

    print(f"Description: {response.content.description}")
    print(f"Objects: {', '.join(response.content.main_objects)}")
    print(f"Colors: {', '.join(response.content.dominant_colors)}")
    print(f"Type: {response.content.image_type}")


async def multi_image_conversation():
    """Demonstrate a conversation with multiple images."""
    print("\n=== Multi-Image Conversation ===")

    # Example URLs
    image_url_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    image_url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/320px-React-icon.svg.png"

    messages = [
        {"role": "system", "content": "You are an expert image analyst."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's the first image:"},
                {"type": "image_url", "image_url": {"url": image_url_1}},
            ],
        },
        {
            "role": "assistant",
            "content": "I can see the first image. It appears to be a PNG transparency demonstration image.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Now here's a second image. Can you compare them?",
                },
                {"type": "image_url", "image_url": {"url": image_url_2}},
            ],
        },
    ]

    response = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        messages=messages,
        max_completion_tokens=2000,
    )

    print(f"Comparison: {response.content}")


async def main():
    """Run all examples."""
    image_path = "example.png"
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    # Check if local image exists
    if not Path(image_path).exists():
        print(
            f"Note: No local image found at '{image_path}'. Skipping local image examples."
        )
        print("To test local images, place an image file at 'example.png'\n")
    else:
        # Test local image
        await analyze_local_image(image_path)

    # Test image URLs
    await analyze_image_url(test_image_url)

    # Test structured output
    await structured_image_analysis(test_image_url)

    # Test multi-image conversation
    await multi_image_conversation()


if __name__ == "__main__":
    asyncio.run(main())
