"""
OpenAI PDF Support Tool with base64 encoding and structured responses.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel
import openai
from defog import config as defog_config

from .pdf_utils import download_and_process_pdf
from .llm_providers import LLMProvider
from .pdf_processor import PDFAnalysisResult

logger = logging.getLogger(__name__)


class OpenAIPDFProcessor:
    """PDF processor that integrates with OpenAI's API using base64 encoding."""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI PDF processor.

        Args:
            provider: LLM provider (should be openai)
            model: OpenAI model to use
            temperature: Sampling temperature
            api_key: OpenAI API key (optional, will use env var if not provided)
            base_url: OpenAI base URL (optional)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=api_key or defog_config.get("OPENAI_API_KEY"),
            base_url=base_url or "https://api.openai.com/v1/",
        )

    def _create_pdf_input(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Create input content for OpenAI API with PDF.

        Args:
            pdf_chunks: List of base64 encoded PDF chunks
            task: Analysis task
            chunk_index: Current chunk index (0-based)
            total_chunks: Total number of chunks

        Returns:
            Input array for OpenAI API
        """
        # Construct the prompt with task and context
        prompt = f"""You are an expert PDF analyzer. Analyze the provided PDF content and complete the requested task.

PDF Context:
- This is chunk {chunk_index + 1} of {total_chunks} total chunks
- Focus on the content in this specific chunk
- If this is part of a multi-chunk document, note that context may be limited

Task: {task}

Provide a thorough analysis based on the PDF content."""

        input_content = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": f"document_chunk_{chunk_index + 1}.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_chunks[chunk_index]}",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            },
        ]

        return input_content

    async def _process_single_chunk(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int,
        total_chunks: int,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single PDF chunk with OpenAI.

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
            input_content = self._create_pdf_input(
                pdf_chunks, task, chunk_index, total_chunks
            )

            # Prepare API parameters
            api_params = {
                "model": self.model,
                "input": input_content,
            }

            # Only add temperature if not 0 (some models don't support it)
            if self.temperature > 0:
                api_params["temperature"] = self.temperature

            # Call OpenAI API using appropriate method
            if response_format:
                # Use parse method for structured output
                api_params["text_format"] = response_format
                response = await self.client.responses.parse(**api_params)
                content = response.output_parsed
            else:
                # Use create method for unstructured output
                response = await self.client.responses.create(**api_params)
                content = response.output_text

            # Get token usage from response
            usage = getattr(response, "usage", None)
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0

            # Calculate cost (using approximate OpenAI pricing)
            # Note: Adjust these rates based on actual model pricing
            input_cost = input_tokens * 0.0025 / 1000  # $2.50 per 1M input tokens
            output_cost = output_tokens * 0.01 / 1000  # $10 per 1M output tokens
            cost_in_cents = (input_cost + output_cost) * 100

            return {
                "success": True,
                "content": content,
                "chunk_index": chunk_index,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_in_cents": cost_in_cents,
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
        Analyze a PDF from URL with OpenAI.

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

            # Calculate total tokens and cost
            total_input_tokens = sum(
                r.get("input_tokens", 0) for r in chunk_results if r.get("success")
            )
            total_output_tokens = sum(
                r.get("output_tokens", 0) for r in chunk_results if r.get("success")
            )
            total_cost = sum(
                r.get("cost_in_cents", 0) for r in chunk_results if r.get("success")
            )

            # Add processing metadata
            processing_metadata = {
                **metadata,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_in_cents": total_cost,
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
