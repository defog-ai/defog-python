import asyncio
import os
import pytest
from defog.llm.youtube import get_youtube_summary


@pytest.mark.asyncio
async def test_youtube_transcript_end_to_end():
    """End-to-end test for YouTube transcript generation."""
    # Skip test if GEMINI_API_KEY is not set
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    # Use a short, public YouTube video for testing
    video_url = "https://www.youtube.com/watch?v=EysJTNLQVZw"

    # Get transcript
    transcript = await get_youtube_summary(video_url)

    # Basic assertions
    assert transcript is not None
    assert isinstance(transcript, str)
    assert len(transcript) > 0

    # Check that transcript contains some expected content
    # (This will vary by video, but should contain some words)
    assert len(transcript.split()) > 10

    print(f"Generated transcript ({len(transcript)} characters):")
    print(transcript[:200] + "..." if len(transcript) > 200 else transcript)


@pytest.mark.asyncio
async def test_youtube_transcript_end_to_end_with_system_instructions():
    """End-to-end test for YouTube transcript generation with system instructions."""
    # Skip test if GEMINI_API_KEY is not set
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    video_url = "https://www.youtube.com/watch?v=EysJTNLQVZw"
    summary = await get_youtube_summary(
        video_url,
        system_instructions=[
            "Focus on the overall message of the video, not a step by step transcript."
        ],
        task_description="Please explain this video like I am a 5 year old.",
    )
    assert summary is not None
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary.split()) > 10
    print(f"Generated summary:\n{summary}")


if __name__ == "__main__":
    asyncio.run(test_youtube_transcript_end_to_end())
