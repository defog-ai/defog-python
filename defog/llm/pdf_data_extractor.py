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

from .pdf_processor import ClaudePDFProcessor
from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


class DataPointIdentification(BaseModel):
    """Identified data point in a PDF that can be extracted."""
    
    name: str = Field(description="Name of the datapoint (e.g., 'financial_summary_table')")
    description: str = Field(description="Description of what this datapoint contains")
    data_type: str = Field(description="Type of data: 'table', 'key_value_pairs', 'list', 'metrics', 'chart_data'")
    location_hint: str = Field(description="Hint about where in the PDF this data is located")
    schema_fields: List[Dict[str, Any]] = Field(
        description="List of fields for extraction. Each field should have: 'name' (snake_case field name), 'type' (data type), 'description' (what the field contains), and 'optional' (boolean, preferably true). For financial tables, use descriptive names like 'revenue_q1_2024' instead of generic names."
    )


class PDFAnalysisResponse(BaseModel):
    """Response from PDF structure analysis."""
    
    document_type: str = Field(description="Type of document (e.g., 'financial_report', 'research_paper')")
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
        Initialize PDF Data Extractor.
        
        Args:
            analysis_provider: Provider for PDF analysis
            analysis_model: Model for PDF analysis
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
        
        # Initialize PDF processor for initial analysis
        self.pdf_processor = ClaudePDFProcessor(
            provider=analysis_provider,
            model=analysis_model,
            temperature=temperature
        )
    
    async def analyze_pdf_structure(
        self,
        pdf_url: str,
        focus_areas: Optional[List[str]] = None
    ) -> PDFAnalysisResponse:
        """
        Analyze PDF to identify extractable datapoints.
        
        Args:
            pdf_url: URL of the PDF to analyze
            focus_areas: Optional list of areas to focus on
            
        Returns:
            PDFAnalysisResponse with identified datapoints
        """
        analysis_task = f"""Analyze this PDF to identify all structured data that can be extracted and converted to tabular format suitable for SQL databases.

Focus on identifying:
1. Tables (financial, statistical, comparison) - Extract with proper column headers
2. Key-value pairs (metadata, properties, specifications)
3. Lists and enumerations
4. Metrics and measurements with clear labels
5. Chart/graph data that can be tabulated
6. Repeated patterns suggesting structured data

{f"Specifically focus on: {', '.join(focus_areas)}" if focus_areas else ""}

For each identified datapoint:
- Provide a descriptive name using snake_case (e.g., 'quarterly_revenue_by_segment')
- Specify the data type
- Indicate where it's located in the document
- Define schema fields with:
  * Clear, descriptive field names in snake_case
  * Appropriate data types (string, int, float, date, etc.)
  * Mark ALL fields as optional=true to handle missing data gracefully

IMPORTANT: For tables, ensure you identify meaningful column names based on the table headers. If a table has multiple time periods, create fields like 'q1_2024', 'q2_2024' etc. rather than generic names.

Be thorough and identify ALL potential datapoints that could be valuable when converted to structured format for database storage."""

        result = await self.pdf_processor.analyze_pdf(
            url=pdf_url,
            task=analysis_task,
            response_format=PDFAnalysisResponse
        )
        
        if not result.success:
            raise Exception(f"PDF analysis failed: {result.error}")
            
        return result.result
    
    def _generate_pydantic_schema(
        self,
        datapoint: DataPointIdentification
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
            field_name = field_info.get("name", "").replace(" ", "_").lower()
            field_type_str = field_info.get("type", "string").lower()
            field_description = field_info.get("description", f"Field {field_name}")
            
            # Get Python type
            python_type = type_mapping.get(field_type_str, str)
            
            # Make all fields optional by default for better data handling
            is_optional = field_info.get("optional", True)
            if is_optional:
                python_type = Optional[python_type]
            
            # Create field with description
            field_definitions[field_name] = (
                python_type,
                Field(description=field_description)
            )
        
        # Handle different data types
        if datapoint.data_type == "table":
            # For tables, if we only have generic field definitions, create a flexible schema
            if len(field_definitions) == 0 or all(name.startswith("column") or name == "" for name in field_definitions.keys()):
                # Create a flexible dict-based schema for tables with dynamic columns
                return create_model(
                    datapoint.name,
                    rows=(List[Dict[str, Optional[Union[str, int, float]]]], Field(description="Table rows with dynamic columns")),
                    column_headers=(Optional[List[str]], Field(default=None, description="Column headers if available")),
                    table_name=(Optional[str], Field(default=datapoint.name, description="Table name")),
                    row_count=(Optional[int], Field(default=None, description="Number of rows"))
                )
            else:
                # For tables with known columns, create a structured row model
                row_model = create_model(
                    f"{datapoint.name}_Row",
                    **field_definitions
                )
                # Return a model that contains a list of rows
                return create_model(
                    datapoint.name,
                    rows=(List[row_model], Field(description="Table rows")),
                    table_name=(Optional[str], Field(default=datapoint.name, description="Table name")),
                    row_count=(Optional[int], Field(default=None, description="Number of rows"))
                )
        elif datapoint.data_type == "key_value_pairs":
            # For key-value pairs, use the fields directly
            return create_model(
                datapoint.name,
                **field_definitions
            )
        elif datapoint.data_type == "list":
            # For lists, create a model with an items field
            item_type = list(field_definitions.values())[0][0] if field_definitions else str
            return create_model(
                datapoint.name,
                items=(List[item_type], Field(description="List items")),
                count=(int, Field(description="Number of items"))
            )
        else:
            # Default: use fields as-is
            return create_model(
                datapoint.name,
                **field_definitions
            )
    
    async def extract_single_datapoint(
        self,
        pdf_url: str,
        datapoint: DataPointIdentification,
        schema: Type[BaseModel]
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
Location hint: {datapoint.location_hint}

IMPORTANT EXTRACTION GUIDELINES:
1. For tables:
   - Use the actual column headers from the PDF as field names
   - Each row should have ALL columns populated with their values
   - If a cell is empty, use null or an empty string
   - For financial data with time periods, use descriptive field names like 'q1_2024_revenue' not just 'value'
   - Extract numbers as pure numeric values without currency symbols or commas

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
            # Use PDF processor for extraction with schema
            processor = ClaudePDFProcessor(
                provider=self.extraction_provider,
                model=self.extraction_model,
                temperature=self.temperature
            )
            
            result = await processor.analyze_pdf(
                url=pdf_url,
                task=extraction_task,
                response_format=schema
            )
            
            if result.success:
                return DataExtractionResult(
                    datapoint_name=datapoint.name,
                    success=True,
                    extracted_data=result.result
                )
            else:
                return DataExtractionResult(
                    datapoint_name=datapoint.name,
                    success=False,
                    error=result.error
                )
                
        except Exception as e:
            logger.error(f"Error extracting datapoint {datapoint.name}: {e}")
            return DataExtractionResult(
                datapoint_name=datapoint.name,
                success=False,
                error=str(e)
            )
    
    async def extract_all_data(
        self,
        pdf_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
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
        analysis = await self.analyze_pdf_structure(pdf_url, focus_areas)
        
        logger.info(f"Identified {len(analysis.identified_datapoints)} datapoints")
        
        # Filter datapoints if requested
        datapoints_to_extract = analysis.identified_datapoints
        if datapoint_filter:
            datapoints_to_extract = [
                dp for dp in datapoints_to_extract 
                if dp.name in datapoint_filter
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
                    pdf_url,
                    datapoint,
                    schemas[datapoint.name]
                )
                extraction_tasks.append(task)
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel_extractions)
        
        async def extract_with_limit(task):
            async with semaphore:
                return await task
        
        logger.info(f"Starting parallel extraction of {len(extraction_tasks)} datapoints")
        extraction_results = await asyncio.gather(
            *[extract_with_limit(task) for task in extraction_tasks],
            return_exceptions=True
        )
        
        # Process results
        final_results = []
        successful = 0
        failed = 0
        total_cost = 0.0
        
        for result in extraction_results:
            if isinstance(result, Exception):
                failed += 1
                final_results.append(
                    DataExtractionResult(
                        datapoint_name="unknown",
                        success=False,
                        error=str(result)
                    )
                )
            else:
                final_results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1
        
        end_time = asyncio.get_event_loop().time()
        
        return PDFDataExtractionResult(
            pdf_url=pdf_url,
            document_type=analysis.document_type,
            total_datapoints_identified=len(analysis.identified_datapoints),
            successful_extractions=successful,
            failed_extractions=failed,
            extraction_results=final_results,
            total_time_ms=int((end_time - start_time) * 1000),
            total_cost_cents=total_cost,
            metadata={
                "filtered_datapoints": len(datapoints_to_extract),
                "schemas_generated": len(schemas)
            }
        )
    
    async def extract_as_dict(
        self,
        pdf_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
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
                "extraction_summary": {
                    "total_identified": result.total_datapoints_identified,
                    "successful": result.successful_extractions,
                    "failed": result.failed_extractions,
                    "time_ms": result.total_time_ms,
                    "cost_cents": result.total_cost_cents
                }
            },
            "data": {}
        }
        
        for extraction in result.extraction_results:
            if extraction.success and extraction.extracted_data:
                extracted_data["data"][extraction.datapoint_name] = extraction.extracted_data
        
        return extracted_data


# Convenience function for simple usage
async def extract_pdf_data(
    pdf_url: str,
    focus_areas: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to extract all data from a PDF.
    
    Args:
        pdf_url: URL of the PDF to process
        focus_areas: Optional areas to focus on
        **kwargs: Additional arguments for PDFDataExtractor
        
    Returns:
        Dictionary with extracted data
    """
    extractor = PDFDataExtractor(**kwargs)
    return await extractor.extract_as_dict(pdf_url, focus_areas)