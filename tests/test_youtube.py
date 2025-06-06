import asyncio
import os
import pytest
from defog.llm.youtube_transcript import get_transcript


@pytest.mark.asyncio
async def test_youtube_transcript_end_to_end():
    """End-to-end test for YouTube transcript generation."""
    # Skip test if GEMINI_API_KEY is not set
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    # Use a short, public YouTube video for testing
    video_url = "https://www.youtube.com/watch?v=EysJTNLQVZw"

    # Get transcript
    transcript = await get_transcript(video_url)

    # Basic assertions
    assert transcript is not None
    assert isinstance(transcript, str)
    assert len(transcript) > 0

    # Check that transcript contains some expected content
    # (This will vary by video, but should contain some words)
    assert len(transcript.split()) > 10

    print(f"Generated transcript ({len(transcript)} characters):")
    print(transcript[:200] + "..." if len(transcript) > 200 else transcript)


if __name__ == "__main__":
    asyncio.run(test_youtube_transcript_end_to_end())
