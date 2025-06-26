"""
Text Data Extractor Agent Orchestrator

This module provides intelligent text analysis to:
1. Identify interesting datapoints that can be extracted as tabular data
2. Generate appropriate response_format schemas for each datapoint
3. Orchestrate parallel extraction of all identified datapoints
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field, create_model

from .utils import chat_async
from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)

# Constants for text processing
MAX_TEXT_SIZE_MB = 5  # Maximum text size in megabytes
MAX_TEXT_SIZE_BYTES = MAX_TEXT_SIZE_MB * 1024 * 1024


class SchemaField(BaseModel):
    """Schema field definition for data extraction."""

    name: str = Field(description="Snake_case field name")
    type: str = Field(description="Data type (string, int, float, etc.)")
    description: str = Field(description="What the field contains")
    optional: bool = Field(default=True, description="Whether the field is optional")


class DataPointIdentification(BaseModel):
    """Identified data point in text that can be extracted."""

    name: str = Field(description="Name of the datapoint (e.g., 'economic_indicators')")
    description: str = Field(description="Description of what this datapoint contains")
    data_type: str = Field(
        description="Type of data: 'key_value_pairs', 'list', 'table', 'metrics', 'statements'"
    )
    schema_fields: List[SchemaField] = Field(
        description="List of data fields for extraction."
    )


class TextAnalysisResponse(BaseModel):
    """Response from text structure analysis."""

    document_type: str = Field(
        description="Type of document (e.g., 'transcript', 'speech', 'report', 'article', 'policy_document')"
    )
    content_description: str = Field(
        description="Brief description of the text content"
    )
    identified_datapoints: List[DataPointIdentification] = Field(
        description="List of all identified extractable datapoints"
    )


class DataExtractionResult(BaseModel):
    """Result of extracting a single datapoint."""

    datapoint_name: str
    success: bool
    extracted_data: Optional[Any] = None
    error: Optional[str] = None
    cost_cents: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class TextDataExtractionResult(BaseModel):
    """Complete result of text data extraction."""

    text_content_hash: str
    document_type: str
    total_datapoints_identified: int
    successful_extractions: int
    failed_extractions: int
    extraction_results: List[DataExtractionResult]
    total_time_ms: int
    total_cost_cents: float
    metadata: Dict[str, Any]


class TextDataExtractor:
    """
    Intelligent text data extractor that identifies and extracts
    structured data from text strings using parallel agent orchestration.
    """

    def __init__(
        self,
        analysis_provider: Union[str, LLMProvider] = "anthropic",
        analysis_model: str = "claude-sonnet-4-20250514",
        extraction_provider: Union[str, LLMProvider] = "anthropic",
        extraction_model: str = "claude-sonnet-4-20250514",
        max_parallel_extractions: int = 5,
        temperature: float = 0.0,
        max_text_size_mb: int = MAX_TEXT_SIZE_MB,
        enable_caching: bool = True,
    ):
        """
        Initialize Text Data Extractor.

        Args:
            analysis_provider: Provider for text analysis
            analysis_model: Model for text analysis
            extraction_provider: Provider for data extraction
            extraction_model: Model for data extraction
            max_parallel_extractions: Maximum parallel extraction tasks
            temperature: Sampling temperature
            max_text_size_mb: Maximum text size in megabytes
            enable_caching: Whether to enable caching of analysis results
        """
        self.analysis_provider = analysis_provider
        self.analysis_model = analysis_model
        self.extraction_provider = extraction_provider
        self.extraction_model = extraction_model
        self.max_parallel_extractions = max_parallel_extractions
        self.temperature = temperature
        self.max_text_size_bytes = max_text_size_mb * 1024 * 1024
        self.enable_caching = enable_caching
        self._analysis_cache = {} if enable_caching else None

    async def analyze_text_structure(
        self, text_content: str, focus_areas: Optional[List[str]] = None
    ) -> tuple[TextAnalysisResponse, Dict[str, Any]]:
        """
        Analyze text to identify extractable datapoints.

        Args:
            text_content: Text string to analyze
            focus_areas: Optional list of areas to focus on

        Returns:
            Tuple of (TextAnalysisResponse with identified datapoints, cost metadata dict)
        """
        logger.info(
            f"Starting text structure analysis, content size: {len(text_content)} chars"
        )
        if focus_areas:
            logger.info(f"Focus areas: {focus_areas}")

        analysis_task = f"""Analyze this text to identify extractable structured data suitable for SQL databases.

Find all structured data patterns:
- Key-value pairs (metrics, statistics, dates)
- Numerical data (percentages, amounts, rates)

ONLY focus on numerically focused data. Do not focus on general qualitative statements, general questions and answers, or other non-numerical data.

{f"Focus on: {', '.join(focus_areas)}" if focus_areas else ""}

For each datapoint provide:
- name: snake_case (e.g., 'economic_indicators')
- data_type: 'key_value_pairs', 'list', 'table', or 'metrics'
- schema_fields: [{{name, type, description, optional: true}}]

Extract RAW DATA values. Each datapoint should yield MULTIPLE ROWS when applicable."""

        # Prepare message with text content
        messages = [
            {
                "role": "user",
                "content": f"<text_content>\n{text_content}\n</text_content>\n\n{analysis_task}",
            }
        ]

        # Call LLM with structured response
        logger.info(
            f"Calling {self.analysis_provider}/{self.analysis_model} for analysis"
        )
        start_time = asyncio.get_event_loop().time()

        result = await chat_async(
            provider=self.analysis_provider,
            model=self.analysis_model,
            messages=messages,
            temperature=self.temperature,
            response_format=TextAnalysisResponse,
        )

        analysis_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")

        # chat_async returns the response directly
        if not result.content:
            raise Exception("Text analysis failed: No response content")

        # Extract cost metadata from LLMResponse
        cost_metadata = {
            "cost_cents": result.cost_in_cents or 0.0,
            "input_tokens": result.input_tokens or 0,
            "output_tokens": result.output_tokens or 0,
            "cached_tokens": result.cached_input_tokens or 0,
        }

        logger.info(
            f"Analysis identified {len(result.content.identified_datapoints)} datapoints"
        )
        logger.info(f"Analysis cost: ${cost_metadata['cost_cents'] / 100:.4f}")

        return result.content, cost_metadata

    def _preprocess_text(self, text_content: str) -> str:
        """
        Preprocess text to improve extraction quality.

        Args:
            text_content: Raw text string

        Returns:
            Preprocessed text string
        """
        # Basic preprocessing while preserving structure
        # Remove excessive whitespace while keeping paragraph breaks
        lines = text_content.split("\n")
        processed_lines = []

        for line in lines:
            # Trim whitespace from each line
            line = line.strip()
            if line:  # Keep non-empty lines
                processed_lines.append(line)
            elif (
                processed_lines and processed_lines[-1]
            ):  # Keep single empty lines for paragraph breaks
                processed_lines.append("")

        # Join back with single newlines and strip trailing whitespace
        result = "\n".join(processed_lines)
        return result.rstrip()

    def _generate_pydantic_schema(
        self, datapoint: DataPointIdentification
    ) -> Type[BaseModel]:
        """
        Generate a Pydantic model dynamically based on identified schema fields.

        Args:
            datapoint: Identified datapoint with schema information

        Returns:
            Dynamically created Pydantic model
        """
        if datapoint.data_type == "table" or datapoint.data_type == "metrics":
            # For tables and metrics, use columnar format for efficiency
            column_names = []
            for field_info in datapoint.schema_fields:
                field_name = field_info.name.replace(" ", "_").lower()
                if field_name:
                    column_names.append(field_name)

            return create_model(
                datapoint.name,
                columns=(
                    List[str],
                    Field(
                        default=column_names if column_names else None,
                        description="Column names for the extracted data",
                    ),
                ),
                data=(
                    List[List[Optional[Union[str, int, float, bool]]]],
                    Field(
                        description="Data as array of arrays (each inner array is a row)"
                    ),
                ),
                row_count=(
                    Optional[int],
                    Field(default=None, description="Number of rows extracted"),
                ),
            )
        elif datapoint.data_type == "key_value_pairs":
            # For key-value pairs, create fields directly from schema
            field_definitions = {}
            for field_info in datapoint.schema_fields:
                field_name = field_info.name.replace(" ", "_").lower()
                field_type_str = field_info.type.lower()
                field_description = field_info.description

                # Map string types to Python types
                type_mapping = {
                    "string": str,
                    "str": str,
                    "text": str,
                    "int": int,
                    "integer": int,
                    "float": float,
                    "decimal": float,
                    "number": float,
                    "bool": bool,
                    "boolean": bool,
                    "date": str,
                    "datetime": str,
                }

                python_type = type_mapping.get(field_type_str, str)
                # All fields are optional
                python_type = Optional[python_type]

                field_definitions[field_name] = (
                    python_type,
                    Field(description=field_description),
                )

            return create_model(datapoint.name, **field_definitions)
        elif datapoint.data_type == "list":
            # For lists, create a simple list structure
            return create_model(
                datapoint.name,
                items=(
                    List[str],
                    Field(description="List of extracted items"),
                ),
                item_count=(
                    Optional[int],
                    Field(default=None, description="Number of items extracted"),
                ),
            )
        else:  # statements or other types
            # For statements and other types, use a flexible format
            return create_model(
                datapoint.name,
                items=(
                    List[Dict[str, Optional[Union[str, int, float, bool]]]],
                    Field(description="Extracted structured items"),
                ),
                item_count=(
                    Optional[int],
                    Field(default=None, description="Number of items extracted"),
                ),
            )

    def _aggregate_cost_and_token_metadata(
        self,
        analysis_metadata: Dict[str, Any],
        extraction_results: List[DataExtractionResult],
    ) -> Dict[str, Any]:
        """
        Aggregate cost and token metadata from analysis and extraction results.

        Args:
            analysis_metadata: Cost metadata from text analysis
            extraction_results: List of extraction results with individual costs

        Returns:
            Dictionary with aggregated totals
        """
        # Start with analysis costs
        total_cost = analysis_metadata.get("cost_cents", 0.0)
        total_input_tokens = analysis_metadata.get("input_tokens", 0)
        total_output_tokens = analysis_metadata.get("output_tokens", 0)
        total_cached_tokens = analysis_metadata.get("cached_tokens", 0)

        # Aggregate costs from individual extractions
        for result in extraction_results:
            if not isinstance(result, Exception):
                total_cost += result.cost_cents
                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
                total_cached_tokens += result.cached_tokens

        return {
            "total_cost_cents": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "analysis_cost_cents": analysis_metadata.get("cost_cents", 0.0),
            "extraction_cost_cents": total_cost
            - analysis_metadata.get("cost_cents", 0.0),
        }

    async def extract_single_datapoint(
        self,
        text_content: str,
        datapoint: DataPointIdentification,
        schema: Type[BaseModel],
    ) -> DataExtractionResult:
        """
        Extract a single datapoint from the text.

        Args:
            text_content: Text string
            datapoint: Datapoint to extract
            schema: Pydantic schema for extraction

        Returns:
            DataExtractionResult
        """
        logger.info(
            f"Starting extraction of datapoint: {datapoint.name} ({datapoint.data_type})"
        )
        extraction_task = f"""Extract: {datapoint.name} ({datapoint.data_type})

For tables/lists: Use columnar format with 'columns' and 'data' arrays.
For key-value pairs: Extract fields with proper types (numbers without symbols).

Extract RAW VALUES only. Empty fields = null."""

        try:
            # Prepare message with text
            messages = [
                {
                    "role": "user",
                    "content": f"<text_content>\n{text_content}\n</text_content>\n\n{extraction_task}",
                }
            ]

            # Call LLM directly with schema
            logger.info(
                f"Calling {self.extraction_provider}/{self.extraction_model} for extraction"
            )
            start_time = asyncio.get_event_loop().time()

            result = await chat_async(
                provider=self.extraction_provider,
                model=self.extraction_model,
                messages=messages,
                temperature=self.temperature,
                response_format=schema,
            )

            extraction_time = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"Extraction of {datapoint.name} completed in {extraction_time:.2f} seconds"
            )

            # chat_async returns the response directly
            return DataExtractionResult(
                datapoint_name=datapoint.name,
                success=True,
                extracted_data=result.content,
                cost_cents=result.cost_in_cents or 0.0,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cached_tokens=result.cached_input_tokens or 0,
            )

        except Exception as e:
            logger.error(f"Error extracting datapoint {datapoint.name}: {e}")
            return DataExtractionResult(
                datapoint_name=datapoint.name,
                success=False,
                error=str(e),
                cost_cents=0.0,
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
            )

    async def extract_all_data(
        self,
        text_content: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> TextDataExtractionResult:
        """
        Extract all identified datapoints from text in parallel.

        Args:
            text_content: Text string to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            TextDataExtractionResult with all extracted data
        """
        start_time = asyncio.get_event_loop().time()

        # Validate text size
        text_size = len(text_content.encode("utf-8"))
        if text_size > self.max_text_size_bytes:
            raise ValueError(
                f"Text content exceeds maximum size limit of {self.max_text_size_bytes / (1024 * 1024):.2f} MB. "
                f"Actual size: {text_size / (1024 * 1024):.2f} MB"
            )

        # Generate a secure hash for the text content
        content_hash = hashlib.sha256(text_content.encode()).hexdigest()[:16]

        # Preprocess text
        preprocessed_text = self._preprocess_text(text_content)

        # Check cache if enabled
        cache_key = f"{content_hash}:{','.join(focus_areas or [])}:{','.join(datapoint_filter or [])}"
        if self.enable_caching and cache_key in self._analysis_cache:
            logger.info(f"Using cached analysis for hash: {content_hash}")
            cached_result = self._analysis_cache[cache_key]
            return cached_result

        # Step 1: Analyze text structure
        logger.info(f"Analyzing text structure (hash: {content_hash})")
        analysis, analysis_cost_metadata = await self.analyze_text_structure(
            preprocessed_text, focus_areas
        )

        logger.info("Step 1 - Text Analysis completed:")
        logger.info(f"  • Identified {len(analysis.identified_datapoints)} datapoints")
        logger.info(
            f"  • Cost: ${analysis_cost_metadata.get('cost_cents', 0.0) / 100:.4f}"
        )
        logger.info(
            f"  • Input tokens: {analysis_cost_metadata.get('input_tokens', 0):,}"
        )
        logger.info(
            f"  • Output tokens: {analysis_cost_metadata.get('output_tokens', 0):,}"
        )
        logger.info(
            f"  • Cached tokens: {analysis_cost_metadata.get('cached_tokens', 0):,}"
        )

        # Filter datapoints if requested
        datapoints_to_extract = analysis.identified_datapoints
        if datapoint_filter:
            datapoints_to_extract = [
                dp for dp in datapoints_to_extract if dp.name in datapoint_filter
            ]

        # Step 2: Generate schemas for each datapoint
        logger.info("Step 2 - Generating schemas for datapoints")
        schemas = {}
        for datapoint in datapoints_to_extract:
            try:
                schema = self._generate_pydantic_schema(datapoint)
                schemas[datapoint.name] = schema
                logger.info(f"Generated schema for {datapoint.name}")
            except Exception as e:
                logger.error(f"Failed to generate schema for {datapoint.name}: {e}")

        # Step 3: Extract data in parallel
        extraction_tasks = []
        for datapoint in datapoints_to_extract:
            if datapoint.name in schemas:
                task = self.extract_single_datapoint(
                    preprocessed_text, datapoint, schemas[datapoint.name]
                )
                extraction_tasks.append(task)

        logger.info(f"Created {len(extraction_tasks)} extraction tasks")

        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel_extractions)

        async def extract_with_limit(task):
            async with semaphore:
                return await task

        logger.info(
            f"Step 3 - Starting parallel extraction of {len(extraction_tasks)} datapoints"
        )
        logger.info(f"Max parallel extractions: {self.max_parallel_extractions}")

        extraction_start_time = asyncio.get_event_loop().time()
        extraction_results = await asyncio.gather(
            *[extract_with_limit(task) for task in extraction_tasks],
            return_exceptions=True,
        )

        extraction_duration = asyncio.get_event_loop().time() - extraction_start_time
        logger.info(f"All extractions completed in {extraction_duration:.2f} seconds")

        logger.info("Step 3 - Individual extraction costs:")

        # Process results
        final_results = []
        successful = 0
        failed = 0

        for result in extraction_results:
            if isinstance(result, Exception):
                failed += 1
                final_results.append(
                    DataExtractionResult(
                        datapoint_name="unknown",
                        success=False,
                        error=str(result),
                        cost_cents=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        cached_tokens=0,
                    )
                )
            else:
                final_results.append(result)

                # Log individual extraction costs
                if result.cost_cents > 0:
                    logger.info(
                        f"  • {result.datapoint_name}: ${result.cost_cents / 100:.4f} "
                        f"(in:{result.input_tokens:,}, out:{result.output_tokens:,}, "
                        f"cached:{result.cached_tokens:,}) - {'✅' if result.success else '❌'}"
                    )

                if result.success:
                    successful += 1
                else:
                    failed += 1

        # Aggregate cost and token metadata using helper method
        cost_metadata = self._aggregate_cost_and_token_metadata(
            analysis_cost_metadata, final_results
        )

        end_time = asyncio.get_event_loop().time()

        # Log final summary
        logger.info("Text Data Extraction completed:")
        logger.info(f"  • Total time: {(end_time - start_time):.2f} seconds")
        logger.info(f"  • Total cost: ${cost_metadata['total_cost_cents'] / 100:.4f}")
        logger.info(
            f"  • Analysis cost: ${cost_metadata['analysis_cost_cents'] / 100:.4f}"
        )
        logger.info(
            f"  • Extraction cost: ${cost_metadata['extraction_cost_cents'] / 100:.4f}"
        )
        logger.info(
            f"  • Total tokens: {cost_metadata['total_input_tokens'] + cost_metadata['total_output_tokens']:,} "
            f"(in:{cost_metadata['total_input_tokens']:,}, out:{cost_metadata['total_output_tokens']:,}, "
            f"cached:{cost_metadata['total_cached_tokens']:,})"
        )
        logger.info(
            f"  • Success rate: {successful}/{successful + failed} "
            f"({100 * successful / (successful + failed) if (successful + failed) > 0 else 0:.1f}%)"
        )

        result = TextDataExtractionResult(
            text_content_hash=content_hash,
            document_type=analysis.document_type,
            total_datapoints_identified=len(analysis.identified_datapoints),
            successful_extractions=successful,
            failed_extractions=failed,
            extraction_results=final_results,
            total_time_ms=int((end_time - start_time) * 1000),
            total_cost_cents=cost_metadata["total_cost_cents"],
            metadata={
                "content_description": analysis.content_description,
                "filtered_datapoints": len(datapoints_to_extract),
                "schemas_generated": len(schemas),
                "total_input_tokens": cost_metadata["total_input_tokens"],
                "total_output_tokens": cost_metadata["total_output_tokens"],
                "total_cached_tokens": cost_metadata["total_cached_tokens"],
                "analysis_cost_cents": cost_metadata["analysis_cost_cents"],
                "extraction_cost_cents": cost_metadata["extraction_cost_cents"],
            },
        )

        # Cache the result if enabled
        if self.enable_caching:
            self._analysis_cache[cache_key] = result

        return result

    async def extract_as_dict(
        self,
        text_content: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all data and return as a dictionary for easy access.

        Args:
            text_content: Text string to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            Dictionary with datapoint names as keys and extracted data as values
        """
        result = await self.extract_all_data(
            text_content, focus_areas, datapoint_filter
        )

        extracted_data = {
            "metadata": {
                "text_content_hash": result.text_content_hash,
                "document_type": result.document_type,
                "content_description": result.metadata.get("content_description", ""),
                "extraction_summary": {
                    "total_identified": result.total_datapoints_identified,
                    "successful": result.successful_extractions,
                    "failed": result.failed_extractions,
                    "time_ms": result.total_time_ms,
                    "cost_cents": result.total_cost_cents,
                },
            },
            "data": {},
        }

        for extraction in result.extraction_results:
            if extraction.success and extraction.extracted_data:
                # Convert Pydantic model to dict if needed
                data = extraction.extracted_data
                if hasattr(data, "model_dump"):
                    data = data.model_dump()
                extracted_data["data"][extraction.datapoint_name] = data

        return extracted_data


# Convenience function for simple usage
async def extract_text_data(
    text_content: str, focus_areas: Optional[List[str]] = None, **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to extract all data from text.

    Args:
        text_content: Text string to process
        focus_areas: Optional areas to focus on
        **kwargs: Additional arguments for TextDataExtractor

    Returns:
        Dictionary with extracted data
    """
    extractor = TextDataExtractor(**kwargs)
    return await extractor.extract_as_dict(text_content, focus_areas)
