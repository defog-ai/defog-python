"""
PDF Data Extractor Agent Orchestrator

This module provides intelligent PDF analysis to:
1. Identify interesting datapoints that can be extracted as tabular data
2. Generate appropriate response_format schemas for each datapoint
3. Orchestrate parallel extraction of all identified datapoints
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field, create_model

from .pdf_processor import ClaudePDFProcessor, OpenAIPDFProcessor

logger = logging.getLogger(__name__)


class SchemaField(BaseModel):
    """Schema field definition for data extraction."""

    name: str = Field(description="Snake_case field name")
    type: str = Field(description="Data type (string, int, float, date, etc.)")
    description: str = Field(description="What the field contains")
    optional: bool = Field(default=True, description="Whether the field is optional")


class DataPointIdentification(BaseModel):
    """Identified data point in a PDF that can be extracted."""

    name: str = Field(
        description="Name of the datapoint (e.g., 'financial_summary_table')"
    )
    description: str = Field(description="Description of what this datapoint contains")
    data_type: str = Field(
        description="Type of data: 'table', 'key_value_pairs', 'list', 'metrics', 'chart_data'"
    )
    schema_fields: List[SchemaField] = Field(
        description="List of fields for extraction. For financial tables, use descriptive names like 'revenue_q1_2024' instead of generic names."
    )


class PDFAnalysisResponse(BaseModel):
    """Response from PDF structure analysis."""

    document_type: str = Field(
        description="Type of document (e.g., 'financial_report', 'research_paper')"
    )
    total_pages: int = Field(description="Total number of pages analyzed")
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


class PDFDataExtractionResult(BaseModel):
    """Complete result of PDF data extraction."""

    pdf_url: str
    document_type: str
    total_datapoints_identified: int
    successful_extractions: int
    failed_extractions: int
    extraction_results: List[DataExtractionResult]
    total_time_ms: int
    total_cost_cents: float
    metadata: Dict[str, Any]


class PDFDataExtractor:
    """
    Intelligent PDF data extractor that identifies and extracts
    structured data from PDFs using parallel agent orchestration.
    """

    # Supported providers
    SUPPORTED_PROVIDERS = ["anthropic", "openai"]

    def __init__(
        self,
        analysis_provider: str = "anthropic",
        analysis_model: str = "claude-sonnet-4-20250514",
        extraction_provider: str = "anthropic",
        extraction_model: str = "claude-sonnet-4-20250514",
        max_parallel_extractions: int = 5,
        temperature: float = 0.0,
    ):
        """
        Initialize PDF Data Extractor.

        Args:
            analysis_provider: Provider for PDF analysis
            analysis_model: Model for PDF analysis
            extraction_provider: Provider for data extraction
            extraction_model: Model for data extraction
            max_parallel_extractions: Maximum parallel extraction tasks
            temperature: Sampling temperature
        """
        # Validate providers
        if analysis_provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported analysis provider: {analysis_provider}. "
                f"Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )
        if extraction_provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported extraction provider: {extraction_provider}. "
                f"Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        self.analysis_provider = analysis_provider
        self.analysis_model = analysis_model
        self.extraction_provider = extraction_provider
        self.extraction_model = extraction_model
        self.max_parallel_extractions = max_parallel_extractions
        self.temperature = temperature

        # Initialize PDF processor for initial analysis based on provider
        if analysis_provider == "openai":
            self.pdf_processor = OpenAIPDFProcessor(
                provider=analysis_provider,
                model=analysis_model,
                temperature=temperature,
            )
        else:
            self.pdf_processor = ClaudePDFProcessor(
                provider=analysis_provider,
                model=analysis_model,
                temperature=temperature,
            )

    async def analyze_pdf_structure(
        self, pdf_url: str, focus_areas: Optional[List[str]] = None
    ) -> tuple[PDFAnalysisResponse, Dict[str, Any]]:
        """
        Analyze PDF to identify extractable datapoints.

        Args:
            pdf_url: URL of the PDF to analyze
            focus_areas: Optional list of areas to focus on

        Returns:
            Tuple of (PDFAnalysisResponse with identified datapoints, cost metadata dict)
        """
        analysis_task = f"""Analyze this PDF to identify all structured data that can be extracted and converted to a tabular format.

Focus on identifying:
1. Tables - extract these with proper column headers
2. Chart/graph data that can be tabulated
3. Key-value pairs - extract these with proper keys and values
4. Lists and enumerations
5. Metrics and measurements with clear labels

{f"Specifically focus on: {', '.join(focus_areas)}" if focus_areas else ""}

For each identified datapoint:
- Provide a descriptive name using snake_case (e.g., 'quarterly_revenue_by_segment')
- Specify the data type
- Define schema fields with:
  * Clear, descriptive field names in snake_case
  * Appropriate data types (string, int, float, date, etc.)
  * Mark ALL fields as optional=true to handle missing data gracefully

IMPORTANT: For tables, ensure you identify meaningful column names based on the table headers. If a table has multiple time periods, create fields like 'q1_2024', 'q2_2024' etc. rather than generic names.

Be thorough and identify ALL potential datapoints that could be valuable when converted to structured format for database storage."""

        result = await self.pdf_processor.analyze_pdf(
            url=pdf_url, task=analysis_task, response_format=PDFAnalysisResponse
        )

        if not result.success:
            raise Exception(f"PDF analysis failed: {result.error}")

        # Extract cost metadata
        cost_metadata = {
            "cost_cents": result.metadata.get("total_cost_in_cents", 0.0),
            "input_tokens": result.metadata.get("total_input_tokens", 0),
            "output_tokens": result.metadata.get("total_output_tokens", 0),
            "cached_tokens": result.metadata.get("cached_tokens", 0),
        }

        return result.result, cost_metadata

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
            "date": str,  # Will be extracted as string
            "datetime": str,  # Will be extracted as string
            "list": List[str],  # Default to list of strings
            "dict": Dict[str, Any],
        }

        # Build field definitions
        field_definitions = {}
        for field_info in datapoint.schema_fields:
            field_name = field_info.name.replace(" ", "_").lower()
            field_type_str = field_info.type.lower()
            field_description = field_info.description

            # Get Python type
            python_type = type_mapping.get(field_type_str, str)

            # Make all fields optional by default for better data handling
            is_optional = field_info.optional
            if is_optional:
                python_type = Optional[python_type]

            # Create field with description
            field_definitions[field_name] = (
                python_type,
                Field(description=field_description),
            )

        # Handle different data types
        if datapoint.data_type == "table":
            # For tables, if we only have generic field definitions, create a flexible schema
            if len(field_definitions) == 0 or all(
                name.startswith("column") or name == ""
                for name in field_definitions.keys()
            ):
                # Create a flexible dict-based schema for tables with dynamic columns
                return create_model(
                    datapoint.name,
                    rows=(
                        List[Dict[str, Optional[Union[str, int, float]]]],
                        Field(description="Table rows with dynamic columns"),
                    ),
                    column_headers=(
                        Optional[List[str]],
                        Field(default=None, description="Column headers if available"),
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
            else:
                # For tables with known columns, use columnar format to save tokens
                # Extract column names from field definitions
                column_names = list(field_definitions.keys())

                # Create a model with columns and data arrays
                return create_model(
                    datapoint.name,
                    columns=(
                        List[str],
                        Field(
                            default=column_names,
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
        elif datapoint.data_type == "key_value_pairs":
            # For key-value pairs, use the fields directly
            return create_model(datapoint.name, **field_definitions)
        elif datapoint.data_type == "list":
            # For lists, create a model with an items field
            item_type = (
                list(field_definitions.values())[0][0] if field_definitions else str
            )
            return create_model(
                datapoint.name,
                items=(List[item_type], Field(description="List items")),
                count=(int, Field(description="Number of items")),
            )
        else:
            # Default: use fields as-is
            return create_model(datapoint.name, **field_definitions)

    def _aggregate_cost_and_token_metadata(
        self,
        analysis_metadata: Dict[str, Any],
        extraction_results: List[DataExtractionResult],
    ) -> Dict[str, Any]:
        """
        Aggregate cost and token metadata from analysis and extraction results.

        Args:
            analysis_metadata: Cost metadata from PDF analysis
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
        self, pdf_url: str, datapoint: DataPointIdentification, schema: Type[BaseModel]
    ) -> DataExtractionResult:
        """
        Extract a single datapoint from the PDF.

        Args:
            pdf_url: URL of the PDF
            datapoint: Datapoint to extract
            schema: Pydantic schema for extraction

        Returns:
            DataExtractionResult
        """
        extraction_task = f"""Extract the following data from this PDF:

Datapoint: {datapoint.name}
Description: {datapoint.description}
Type: {datapoint.data_type}

IMPORTANT EXTRACTION GUIDELINES:
1. For tables:
   - Extract data in COLUMNAR FORMAT to minimize tokens:
     * 'columns' field: array of column names matching the PDF headers
     * 'data' field: array of arrays, where each inner array is a row
   - The order of values in each row MUST match the order of columns
   - Example: columns: ["item", "q1_2024", "q2_2024"], data: [["Revenue", 100, 150], ["Cost", 50, 60]]
   - If a cell is empty, use null
   - Extract numbers as pure numeric values without currency symbols or commas
   - For financial data with time periods, use descriptive column names like 'q1_2024_revenue'

2. For key-value pairs:
   - Use the actual labels/keys from the document
   - Extract values in their appropriate data type

3. For all data:
   - Preserve the structure and relationships in the original document
   - Make the data SQL-friendly with clear, unambiguous field names
   - Extract dates in ISO format (YYYY-MM-DD) when possible
   - For percentages, store as decimals (0.15 for 15%)

Please extract this data and format it according to the provided schema.
Be precise and include all available data that matches the schema."""

        try:
            # Use appropriate PDF processor for extraction with schema
            if (
                isinstance(self.extraction_provider, str)
                and self.extraction_provider == "openai"
            ):
                processor = OpenAIPDFProcessor(
                    provider=self.extraction_provider,
                    model=self.extraction_model,
                    temperature=self.temperature,
                )
            else:
                processor = ClaudePDFProcessor(
                    provider=self.extraction_provider,
                    model=self.extraction_model,
                    temperature=self.temperature,
                )

            result = await processor.analyze_pdf(
                url=pdf_url, task=extraction_task, response_format=schema
            )

            if result.success:
                return DataExtractionResult(
                    datapoint_name=datapoint.name,
                    success=True,
                    extracted_data=result.result,
                    cost_cents=result.metadata.get("total_cost_in_cents", 0.0),
                    input_tokens=result.metadata.get("total_input_tokens", 0),
                    output_tokens=result.metadata.get("total_output_tokens", 0),
                    cached_tokens=result.metadata.get("cached_tokens", 0),
                )
            else:
                return DataExtractionResult(
                    datapoint_name=datapoint.name,
                    success=False,
                    error=result.error,
                    cost_cents=result.metadata.get("total_cost_in_cents", 0.0),
                    input_tokens=result.metadata.get("total_input_tokens", 0),
                    output_tokens=result.metadata.get("total_output_tokens", 0),
                    cached_tokens=result.metadata.get("cached_tokens", 0),
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
        pdf_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> PDFDataExtractionResult:
        """
        Extract all identified datapoints from a PDF in parallel.

        Args:
            pdf_url: URL of the PDF to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            PDFDataExtractionResult with all extracted data
        """
        start_time = asyncio.get_event_loop().time()

        # Step 1: Analyze PDF structure
        logger.info(f"Analyzing PDF structure: {pdf_url}")
        analysis, analysis_cost_metadata = await self.analyze_pdf_structure(
            pdf_url, focus_areas
        )

        logger.info("Step 1 - PDF Analysis completed:")
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
                    pdf_url, datapoint, schemas[datapoint.name]
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
        logger.info("PDF Data Extraction completed:")
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

        return PDFDataExtractionResult(
            pdf_url=pdf_url,
            document_type=analysis.document_type,
            total_datapoints_identified=len(analysis.identified_datapoints),
            successful_extractions=successful,
            failed_extractions=failed,
            extraction_results=final_results,
            total_time_ms=int((end_time - start_time) * 1000),
            total_cost_cents=cost_metadata["total_cost_cents"],
            metadata={
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
        pdf_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all data and return as a dictionary for easy access.

        Args:
            pdf_url: URL of the PDF to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            Dictionary with datapoint names as keys and extracted data as values
        """
        result = await self.extract_all_data(pdf_url, focus_areas, datapoint_filter)

        extracted_data = {
            "metadata": {
                "pdf_url": result.pdf_url,
                "document_type": result.document_type,
                "total_datapoints_identified": result.total_datapoints_identified,
                "successful_extractions": result.successful_extractions,
                "failed_extractions": result.failed_extractions,
                "total_time_ms": result.total_time_ms,
                "total_cost_cents": result.total_cost_cents,
                "total_input_tokens": result.metadata.get("total_input_tokens", 0),
                "total_output_tokens": result.metadata.get("total_output_tokens", 0),
                "total_cached_tokens": result.metadata.get("total_cached_tokens", 0),
                "analysis_cost_cents": result.metadata.get("analysis_cost_cents", 0.0),
                "extraction_cost_cents": result.metadata.get(
                    "extraction_cost_cents", 0.0
                ),
            },
            "data": {},
        }

        for extraction in result.extraction_results:
            if extraction.success and extraction.extracted_data:
                # Convert Pydantic models to dict for JSON serialization
                if hasattr(extraction.extracted_data, "model_dump"):
                    extracted_data["data"][extraction.datapoint_name] = (
                        extraction.extracted_data.model_dump()
                    )
                else:
                    extracted_data["data"][extraction.datapoint_name] = (
                        extraction.extracted_data
                    )

        return extracted_data


# Convenience function for simple usage
async def extract_pdf_data(
    pdf_url: str,
    focus_areas: Optional[List[str]] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to extract all data from a PDF.

    Args:
        pdf_url: URL of the PDF to process
        focus_areas: Optional areas to focus on
        provider: LLM provider to use (e.g., "anthropic", "openai")
        model: Model to use (if not specified, uses default for provider)
        **kwargs: Additional arguments for PDFDataExtractor

    Returns:
        Dictionary with extracted data
    """
    # Validate provider
    if provider not in PDFDataExtractor.SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(PDFDataExtractor.SUPPORTED_PROVIDERS)}"
        )

    # Set default model based on provider if not specified
    if model is None:
        if provider == "openai":
            model = "o4-mini"
        else:
            model = "claude-sonnet-4-20250514"

    extractor = PDFDataExtractor(
        analysis_provider=provider,
        analysis_model=model,
        extraction_provider=provider,
        extraction_model=model,
        **kwargs,
    )
    return await extractor.extract_as_dict(pdf_url, focus_areas)
