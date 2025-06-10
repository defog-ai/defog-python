# converts a youtube video to a detailed, ideally diarized transcript
from defog.llm.utils_logging import ToolProgressTracker, SubTaskLogger
import os


async def get_transcript(
    video_url: str,
    model: str = "gemini-2.5-pro-preview-05-06"
) -> str:
    """
    Get a detailed, diarized transcript of a YouTube video.

    Args:
        video_url: The URL of the YouTube video.

    Returns:
        A detailed, ideally diarized transcript of the video.
    """
    async with ToolProgressTracker(
        "YouTube Transcript",
        f"Transcribing video from: {video_url[:50]}{'...' if len(video_url) > 50 else ''}",
    ) as tracker:
        subtask_logger = SubTaskLogger()
        subtask_logger.log_provider_info("Gemini", model)

        if os.getenv("GEMINI_API_KEY") is None:
            raise ValueError("GEMINI_API_KEY is not set")

        from google import genai
        from google.genai.types import (
            Content,
            Part,
            FileData,
            VideoMetadata,
            GenerateContentConfig,
        )

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        tracker.update(10, "Processing. Takes ~1s for every ~10s of video")
        subtask_logger.log_subtask(
            "Using low FPS (0.0) for efficient processing", "info"
        )

        print("\n🎥 Streaming transcript as it's generated:\n")
        print("=" * 60)

        full_transcript = ""

        async for chunk in await client.aio.models.generate_content_stream(
            model=model,
            contents=Content(
                parts=[
                    Part(
                        file_data=FileData(file_uri=video_url),
                        video_metadata=VideoMetadata(fps=0.0),
                    ),
                    Part(
                        text="Please provide a detailed, accurate transcript of the video."
                    ),
                ]
            ),
            config=GenerateContentConfig(
                system_instruction=[
                    "Please provide a detailed, accurate transcript of the video. Please include timestamps in the format HH:MM:SS and names (if available) for each speaker. Do not describe what you *see* in the video, just create a great transcript based on what you *hear*.",
                    "You should skip umms, ahhs, small talk, and other filler words.",
                    "If you find yourself repeating the same words, you should stop.",
                ],
                top_p=1.0,
                
            ),
        ):
            if chunk.text:
                print(chunk.text)
                full_transcript += chunk.text

        print("\n" + "=" * 60)
        print("✅ Transcript complete!\n")

        tracker.update(90, "Finalizing transcript")
        transcript_length = len(full_transcript)

        subtask_logger.log_result_summary(
            "YouTube Transcript",
            {
                "transcript_length": f"{transcript_length} characters",
                "model_used": model,
            },
        )

        return full_transcript
