# converts a youtube video to a detailed, ideally diarized transcript
from defog.llm.utils_logging import ToolProgressTracker, SubTaskLogger
import os


async def get_transcript(
    video_url: str,
    model: str = "gemini-2.5-pro-preview-06-05"
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

        tracker.update(10, "Processing video")
        subtask_logger.log_subtask(
            "Using low FPS (0.2) for efficient processing", "info"
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=Content(
                parts=[
                    Part(
                        file_data=FileData(file_uri=video_url),
                        video_metadata=VideoMetadata(fps=0.2),
                    ),
                    Part(
                        text="Please provide a detailed, accurate transcript of the video."
                    ),
                ]
            ),
            config=GenerateContentConfig(
                system_instruction=[
                    'Please provide a detailed, accurate transcript of the video. Please include timestamps and names (if available) for each speaker in the format [00:00:00]. Do not describe the video, just create a great transcript.',
                    'You should skip umms, ahhs, small talk, and other filler words.',
                    'You should skip any ads that are in the video.',
                ]
            )
        )
        

        tracker.update(90, "Finalizing transcript")
        transcript_length = len(response.text) if response.text else 0

        subtask_logger.log_result_summary(
            "YouTube Transcript",
            {
                "transcript_length": f"{transcript_length} characters",
                "model_used": model,
            },
        )

        return response.text
