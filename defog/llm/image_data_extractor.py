"""
Image Data Extractor Agent Orchestrator

This module provides intelligent image analysis to:
1. Identify interesting datapoints that can be extracted as tabular data
2. Generate appropriate response_format schemas for each datapoint
3. Orchestrate parallel extraction of all identified datapoints
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field, create_model

from .utils import chat_async
from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


class SchemaField(BaseModel):
    """Schema field definition for data extraction."""

    name: str = Field(description="Snake_case field name")
    type: str = Field(description="Data type (string, int, float, etc.)")
    description: str = Field(description="What the field contains")
    optional: bool = Field(default=True, description="Whether the field is optional")


class DataPointIdentification(BaseModel):
    """Identified data point in an image that can be extracted."""

    name: str = Field(
        description="Name of the datapoint (e.g., 'product_inventory_table')"
    )
    description: str = Field(description="Description of what this datapoint contains")
    data_type: str = Field(description="Type of data: 'table' or 'chart_data' ONLY")
    location_hint: str = Field(
        description="Hint about where in the image this data is located"
    )
    schema_fields: List[SchemaField] = Field(
        description="List of fields for extraction. For tables with multiple columns, use descriptive names."
    )


class ImageAnalysisResponse(BaseModel):
    """Response from image structure analysis."""

    image_type: str = Field(
        description="Type of image (e.g., 'chart', 'diagram', 'screenshot', 'document', 'infographic')"
    )
    content_description: str = Field(
        description="Brief description of the image content"
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


class ImageDataExtractionResult(BaseModel):
    """Complete result of image data extraction."""

    image_url: str
    image_type: str
    total_datapoints_identified: int
    successful_extractions: int
    failed_extractions: int
    extraction_results: List[DataExtractionResult]
    total_time_ms: int
    total_cost_cents: float
    metadata: Dict[str, Any]


class ImageDataExtractor:
    """
    Intelligent image data extractor that identifies and extracts
    structured data from images using parallel agent orchestration.
    """

    def __init__(
        self,
        analysis_provider: Union[str, LLMProvider] = "anthropic",
        analysis_model: str = "claude-sonnet-4-20250514",
        extraction_provider: Union[str, LLMProvider] = "anthropic",
        extraction_model: str = "claude-sonnet-4-20250514",
        max_parallel_extractions: int = 5,
        temperature: float = 0.0,
    ):
        """
        Initialize Image Data Extractor.

        Args:
            analysis_provider: Provider for image analysis
            analysis_model: Model for image analysis
            extraction_provider: Provider for data extraction
            extraction_model: Model for data extraction
            max_parallel_extractions: Maximum parallel extraction tasks
            temperature: Sampling temperature
        """
        self.analysis_provider = analysis_provider
        self.analysis_model = analysis_model
        self.extraction_provider = extraction_provider
        self.extraction_model = extraction_model
        self.max_parallel_extractions = max_parallel_extractions
        self.temperature = temperature

        # No need for a separate processor - chat_async handles images natively

    async def analyze_image_structure(
        self, image_url: str, focus_areas: Optional[List[str]] = None
    ) -> tuple[ImageAnalysisResponse, Dict[str, Any]]:
        """
        Analyze image to identify extractable datapoints.

        Args:
            image_url: URL of the image to analyze
            focus_areas: Optional list of areas to focus on

        Returns:
            Tuple of (ImageAnalysisResponse with identified datapoints, cost metadata dict)
        """
        analysis_task = f"""Analyze this image to identify all data that can be extracted and converted to tabular format suitable for SQL databases.

This tool is designed to extract TABULAR DATA from charts, graphs, and tables. Every datapoint should represent multiple rows of data that can be stored in a database table.

Focus on identifying:
1. Tables - Data already in tabular format with rows and columns
2. Bar charts - Extract categories and values for each bar
3. Line charts - Extract x-axis points and y-values for each series
4. Pie charts - Extract labels and percentages/values for each slice
5. Scatter plots - Extract x,y coordinates for each point
6. Heatmaps - Extract row/column labels and cell values
7. Infographics with data series - Extract the underlying data points

{f"Specifically focus on: {', '.join(focus_areas)}" if focus_areas else ""}

DATA TYPE RULES (use only these):
- Use 'table' for: Any data already in rows/columns or lists of entities with same attributes
- Use 'chart_data' for: Any chart/graph where you need to extract the underlying data points

For each identified datapoint:
- Provide a descriptive name using snake_case (e.g., 'electricity_generation_by_country')
- Use ONLY 'table' or 'chart_data' as data_type
- Indicate where it's located in the image
- Define schema_fields as a list where each field has:
  * name: snake_case field name (e.g., 'region', 'sales_amount')
  * type: data type ('string', 'int', 'float', etc.)
  * description: what the field contains
  * optional: true (always set to true for flexibility)
- Examples:
  * Bar chart of sales by region: [{{name: 'region', type: 'string', description: 'Region name', optional: true}}, {{name: 'sales_amount', type: 'float', description: 'Sales in dollars', optional: true}}]
  * Time series: [{{name: 'date', type: 'string', description: 'Date', optional: true}}, {{name: 'value', type: 'float', description: 'Measured value', optional: true}}]

CRITICAL: 
- We want the RAW DATA from charts, not descriptions
- Each datapoint should result in MULTIPLE ROWS when extracted
- If you see a chart, think "what would the data table behind this chart look like?"
- Never create schemas for single entities - we want collections of data"""

        # Prepare message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": analysis_task},
                ],
            }
        ]

        # Call LLM with structured response
        result = await chat_async(
            provider=self.analysis_provider,
            model=self.analysis_model,
            messages=messages,
            temperature=self.temperature,
            response_format=ImageAnalysisResponse,
        )

        # chat_async returns the response directly
        if not result.content:
            raise Exception("Image analysis failed: No response content")

        # Extract cost metadata from LLMResponse
        cost_metadata = {
            "cost_cents": result.cost_in_cents or 0.0,
            "input_tokens": result.input_tokens or 0,
            "output_tokens": result.output_tokens or 0,
            "cached_tokens": result.cached_input_tokens or 0,
        }

        return result.content, cost_metadata

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
        # Since we only support table and chart_data, use a simple columnar format
        if datapoint.data_type == "chart_data":
            # For charts, use a simple tabular structure
            # The schema_fields should define the columns of the resulting table
            return create_model(
                datapoint.name,
                columns=(
                    List[str],
                    Field(description="Column names for the extracted data"),
                ),
                data=(
                    List[List[Optional[Union[str, int, float]]]],
                    Field(
                        description="Chart data as array of arrays (each inner array is a row)"
                    ),
                ),
                chart_type=(
                    Optional[str],
                    Field(default=None, description="Type of chart if applicable"),
                ),
                row_count=(
                    Optional[int],
                    Field(default=None, description="Number of data points"),
                ),
            )
        else:
            # Default to table format for everything
            # Extract column names from schema_fields
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
                        description="Column names for the table",
                    ),
                ),
                data=(
                    List[List[Optional[Union[str, int, float]]]],
                    Field(
                        description="Table data as array of arrays (each inner array is a row)"
                    ),
                ),
                table_name=(
                    Optional[str],
                    Field(default=datapoint.name, description="Table name"),
                ),
                row_count=(
                    Optional[int],
                    Field(default=None, description="Number of rows"),
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
            analysis_metadata: Cost metadata from image analysis
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
        image_url: str,
        datapoint: DataPointIdentification,
        schema: Type[BaseModel],
    ) -> DataExtractionResult:
        """
        Extract a single datapoint from the image.

        Args:
            image_url: URL of the image
            datapoint: Datapoint to extract
            schema: Pydantic schema for extraction

        Returns:
            DataExtractionResult
        """
        extraction_task = f"""Extract the following data from this image:

Datapoint: {datapoint.name}
Description: {datapoint.description}
Type: {datapoint.data_type}
Location hint: {datapoint.location_hint}

IMPORTANT EXTRACTION GUIDELINES:
Extract data in COLUMNAR FORMAT for efficiency:
- 'columns' field: array of column names
- 'data' field: array of arrays, where each inner array is a row
- The order of values in each row MUST match the order of columns

For example, a bar chart showing sales by region:
- columns: ["region", "sales"]
- data: [["North", 1500], ["South", 1200], ["East", 1800], ["West", 900]]

CRITICAL:
- Extract the RAW DATA VALUES from the chart/table, not descriptions
- Numbers should be pure numeric values (no currency symbols or commas)
- Empty cells should be null
- Each row in 'data' represents one data point from the chart/table

Think of it this way: You're creating the data table that would be used to generate this chart."""

        try:
            # Prepare message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": extraction_task},
                    ],
                }
            ]

            # Call LLM directly with schema
            result = await chat_async(
                provider=self.extraction_provider,
                model=self.extraction_model,
                messages=messages,
                temperature=self.temperature,
                response_format=schema,
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
        image_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> ImageDataExtractionResult:
        """
        Extract all identified datapoints from an image in parallel.

        Args:
            image_url: URL of the image to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            ImageDataExtractionResult with all extracted data
        """
        start_time = asyncio.get_event_loop().time()

        # Step 1: Analyze image structure
        logger.info(f"Analyzing image structure: {image_url}")
        analysis, analysis_cost_metadata = await self.analyze_image_structure(
            image_url, focus_areas
        )

        logger.info("Step 1 - Image Analysis completed:")
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
                    image_url, datapoint, schemas[datapoint.name]
                )
                extraction_tasks.append(task)

        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel_extractions)

        async def extract_with_limit(task):
            async with semaphore:
                return await task

        logger.info(
            f"Step 2 - Starting parallel extraction of {len(extraction_tasks)} datapoints"
        )
        extraction_results = await asyncio.gather(
            *[extract_with_limit(task) for task in extraction_tasks],
            return_exceptions=True,
        )

        logger.info("Step 2 - Individual extraction costs:")

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
        logger.info("Image Data Extraction completed:")
        logger.info(f"  • Total time: {(end_time - start_time):.2f} seconds")
        logger.info(f"  • Total cost: ${cost_metadata['total_cost_cents'] / 100:.4f}")
        logger.info(
            f"  • Analysis cost: ${cost_metadata['analysis_cost_cents'] / 100:.4f}"
        )
        logger.info(
            f"  • Extraction cost: ${cost_metadata['extraction_cost_cents'] / 100:.4f}"
        )
        logger.info(
            f"  • Total tokens: {cost_metadata['total_input_tokens'] + cost_metadata['total_output_tokens']:,} (in:{cost_metadata['total_input_tokens']:,}, out:{cost_metadata['total_output_tokens']:,}, cached:{cost_metadata['total_cached_tokens']:,})"
        )
        logger.info(
            f"  • Success rate: {successful}/{successful + failed} ({100 * successful / (successful + failed) if (successful + failed) > 0 else 0:.1f}%)"
        )

        return ImageDataExtractionResult(
            image_url=image_url,
            image_type=analysis.image_type,
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

    async def extract_as_dict(
        self,
        image_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all data and return as a dictionary for easy access.

        Args:
            image_url: URL of the image to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            Dictionary with datapoint names as keys and extracted data as values
        """
        result = await self.extract_all_data(image_url, focus_areas, datapoint_filter)

        extracted_data = {
            "metadata": {
                "image_url": result.image_url,
                "image_type": result.image_type,
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
async def extract_image_data(
    image_url: str, focus_areas: Optional[List[str]] = None, **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to extract all data from an image.

    Args:
        image_url: URL of the image to process
        focus_areas: Optional areas to focus on
        **kwargs: Additional arguments for ImageDataExtractor

    Returns:
        Dictionary with extracted data
    """
    extractor = ImageDataExtractor(**kwargs)
    return await extractor.extract_as_dict(image_url, focus_areas)
