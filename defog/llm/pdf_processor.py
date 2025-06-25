"""
Claude PDF Support Tool with Anthropic Input Caching and Structured Responses.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field

from .pdf_utils import download_and_process_pdf
from .utils import chat_async
from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


class PDFAnalysisInput(BaseModel):
    """Input model for PDF analysis requests."""

    url: str = Field(description="URL of the PDF to analyze")
    task: str = Field(description="Analysis task to perform on the PDF")
    response_format: Optional[Type[BaseModel]] = Field(
        default=None, description="Pydantic model for structured response"
    )


class PDFAnalysisResult(BaseModel):
    """Result model for PDF analysis."""

    success: bool = Field(description="Whether the analysis was successful")
    result: Any = Field(description="Analysis result content")
    metadata: Dict[str, Any] = Field(description="PDF metadata and processing info")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    chunks_processed: int = Field(description="Number of PDF chunks processed")


class ClaudePDFProcessor:
    """PDF processor that integrates with Claude's API and input caching."""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
    ):
        """
        Initialize Claude PDF processor.

        Args:
            provider: LLM provider (should be anthropic for input caching)
            model: Claude model to use
            temperature: Sampling temperature
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

    async def _create_pdf_message(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Create message content for Claude API with PDF and cache control.

        Args:
            pdf_chunks: List of base64 encoded PDF chunks
            task: Analysis task
            chunk_index: Current chunk index (0-based)
            total_chunks: Total number of chunks

        Returns:
            List of messages for Claude API
        """
        # System message with caching
        system_content = f"""You are an expert PDF analyzer. Analyze the provided PDF content and complete the requested task.

PDF Context:
- This is chunk {chunk_index + 1} of {total_chunks} total chunks
- Focus on the content in this specific chunk
- If this is part of a multi-chunk document, note that context may be limited

Provide a thorough analysis based on the PDF content."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_chunks[chunk_index],
                        },
                        "cache_control": {"type": "ephemeral"},  # Cache PDF content
                    },
                    {
                        "type": "text",
                        "text": f"{system_content}\n\nThe task to be completed is: `{task}`",
                    },
                ],
            },
        ]

        return messages

    async def _process_single_chunk(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int,
        total_chunks: int,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single PDF chunk with Claude.

        Args:
            pdf_chunks: List of base64 encoded PDF chunks
            task: Analysis task
            chunk_index: Current chunk index
            total_chunks: Total number of chunks
            response_format: Optional Pydantic model for structured response

        Returns:
            Processing result for this chunk
        """
        try:
            messages = await self._create_pdf_message(
                pdf_chunks, task, chunk_index, total_chunks
            )

            # Prepare chat parameters
            chat_params = {
                "provider": self.provider,
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "reasoning_effort": "low",  # 2048 token thinking budget
            }

            # Add response format if specified
            if response_format:
                chat_params["response_format"] = response_format

            # Call Claude API
            response = await chat_async(**chat_params)

            return {
                "success": True,
                "content": response.content,
                "chunk_index": chunk_index,
                "input_tokens": response.input_tokens or 0,
                "output_tokens": response.output_tokens or 0,
                "cost_in_cents": response.cost_in_cents or 0,
                "cached_tokens": getattr(response, "cached_input_tokens", None) or 0,
            }

        except Exception as e:
            logger.error(f"Error processing PDF chunk {chunk_index}: {e}")
            return {"success": False, "error": str(e), "chunk_index": chunk_index}

    async def _process_multiple_chunks(
        self,
        pdf_chunks: List[str],
        task: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF chunks concurrently.

        Args:
            pdf_chunks: List of base64 encoded PDF chunks
            task: Analysis task
            response_format: Optional Pydantic model for structured response

        Returns:
            List of processing results for all chunks
        """
        total_chunks = len(pdf_chunks)

        # Process chunks concurrently (limit concurrency to avoid rate limits)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests

        async def process_with_semaphore(chunk_index: int):
            async with semaphore:
                return await self._process_single_chunk(
                    pdf_chunks, task, chunk_index, total_chunks, response_format
                )

        # Create tasks for all chunks
        tasks = [process_with_semaphore(i) for i in range(total_chunks)]

        # Wait for all chunks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {"success": False, "error": str(result), "chunk_index": i}
                )
            else:
                processed_results.append(result)

        return processed_results

    def _combine_chunk_results(
        self, chunk_results: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> str:
        """
        Combine results from multiple chunks into a coherent response.

        Args:
            chunk_results: Results from processing each chunk
            metadata: PDF metadata

        Returns:
            Combined analysis result
        """
        successful_results = [r for r in chunk_results if r.get("success")]
        failed_results = [r for r in chunk_results if not r.get("success")]

        if not successful_results:
            return "Failed to process any PDF chunks successfully."

        combined_content = []

        if len(successful_results) > 1:
            combined_content.append(
                f"PDF Analysis Summary ({len(successful_results)} chunks processed):"
            )
            combined_content.append("=" * 60)

            for result in successful_results:
                chunk_idx = result["chunk_index"]
                combined_content.append(f"\n--- Chunk {chunk_idx + 1} Analysis ---")
                combined_content.append(str(result["content"]))
        else:
            # Single chunk result
            combined_content.append(str(successful_results[0]["content"]))

        if failed_results:
            combined_content.append(
                f"\n\nNote: {len(failed_results)} chunks failed to process."
            )

        return "\n".join(combined_content)

    async def analyze_pdf(
        self, url: str, task: str, response_format: Optional[Type[BaseModel]] = None
    ) -> PDFAnalysisResult:
        """
        Analyze a PDF from URL with Claude.

        Args:
            url: PDF URL to analyze
            task: Analysis task to perform
            response_format: Optional Pydantic model for structured response

        Returns:
            PDFAnalysisResult with analysis results
        """
        try:
            # Download and process PDF
            logger.info(f"Processing PDF from URL: {url}")
            pdf_chunks, metadata = await download_and_process_pdf(url)

            logger.info(f"PDF processed: {metadata['chunk_count']} chunks")

            # Process chunks
            if len(pdf_chunks) == 1:
                # Single chunk processing
                result = await self._process_single_chunk(
                    pdf_chunks, task, 0, 1, response_format
                )
                chunk_results = [result]
            else:
                # Multiple chunk processing
                chunk_results = await self._process_multiple_chunks(
                    pdf_chunks, task, response_format
                )

            # Combine results
            if (
                response_format
                and len(pdf_chunks) == 1
                and chunk_results[0].get("success")
            ):
                # Return structured response directly for single chunk
                final_result = chunk_results[0]["content"]
            else:
                # Combine text results for multiple chunks or unstructured response
                final_result = self._combine_chunk_results(chunk_results, metadata)

            # Calculate total tokens and cost (handle None values)
            total_input_tokens = sum(
                r.get("input_tokens") or 0 for r in chunk_results if r.get("success")
            )
            total_output_tokens = sum(
                r.get("output_tokens") or 0 for r in chunk_results if r.get("success")
            )
            total_cost = sum(
                r.get("cost_in_cents") or 0 for r in chunk_results if r.get("success")
            )
            total_cached_tokens = sum(
                r.get("cached_tokens") or 0 for r in chunk_results if r.get("success")
            )

            # Add processing metadata
            processing_metadata = {
                **metadata,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_in_cents": total_cost,
                "cached_tokens": total_cached_tokens,
                "successful_chunks": len(
                    [r for r in chunk_results if r.get("success")]
                ),
                "failed_chunks": len(
                    [r for r in chunk_results if not r.get("success")]
                ),
            }

            return PDFAnalysisResult(
                success=True,
                result=final_result,
                metadata=processing_metadata,
                chunks_processed=len(pdf_chunks),
            )

        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return PDFAnalysisResult(
                success=False,
                result=None,
                metadata={"error_type": type(e).__name__},
                error=str(e),
                chunks_processed=0,
            )


# Default processor instance
_default_processor = ClaudePDFProcessor()


async def analyze_pdf(input: PDFAnalysisInput) -> Dict[str, Any]:
    """
    Tool function for PDF analysis that can be used in orchestrator.

    Args:
        input: PDFAnalysisInput with URL, task, and optional response format

    Returns:
        Dictionary with analysis results
    """
    try:
        # Use the Pydantic model directly as response_format
        response_format = input.response_format
        task = input.task

        # Analyze PDF
        result = await _default_processor.analyze_pdf(
            url=input.url, task=task, response_format=response_format
        )

        return {
            "success": result.success,
            "result": result.result,
            "metadata": result.metadata,
            "error": result.error,
            "chunks_processed": result.chunks_processed,
        }

    except Exception as e:
        logger.error(f"PDF analysis tool failed: {e}")
        return {
            "success": False,
            "result": None,
            "metadata": {},
            "error": str(e),
            "chunks_processed": 0,
        }
