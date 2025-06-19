"""
HTML Data Extractor Agent Orchestrator

This module provides intelligent HTML analysis to:
1. Identify interesting datapoints that can be extracted as tabular data
2. Generate appropriate response_format schemas for each datapoint
3. Orchestrate parallel extraction of all identified datapoints
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field, create_model
from bs4 import BeautifulSoup
import re
import base64

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
    """Identified data point in HTML that can be extracted."""

    name: str = Field(
        description="Name of the datapoint (e.g., 'product_table', 'user_list')"
    )
    description: str = Field(description="Description of what this datapoint contains")
    data_type: str = Field(
        description="Type of data: 'table', 'list', 'key_value_pairs', 'image_data', 'structured_text'"
    )
    location_hint: str = Field(
        description="Hint about where in the HTML this data is located"
    )
    schema_fields: List[SchemaField] = Field(
        description="List of fields for extraction. For tables with multiple columns, use descriptive names."
    )


class HTMLAnalysisResponse(BaseModel):
    """Response from HTML structure analysis."""

    html_type: str = Field(
        description="Type of HTML content (e.g., 'webpage', 'table', 'form', 'dashboard', 'report')"
    )
    content_description: str = Field(
        description="Brief description of the HTML content"
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


class HTMLDataExtractionResult(BaseModel):
    """Complete result of HTML data extraction."""

    html_content_preview: str
    html_type: str
    total_datapoints_identified: int
    successful_extractions: int
    failed_extractions: int
    extraction_results: List[DataExtractionResult]
    total_time_ms: int
    total_cost_cents: float
    metadata: Dict[str, Any]


class HTMLDataExtractor:
    """
    Intelligent HTML data extractor that identifies and extracts
    structured data from HTML strings using parallel agent orchestration.
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
        Initialize HTML Data Extractor.

        Args:
            analysis_provider: Provider for HTML analysis
            analysis_model: Model for HTML analysis
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

    def _parse_html_content(self, html_string: str) -> Dict[str, Any]:
        """
        Parse HTML string and extract structured information.

        Args:
            html_string: Raw HTML string to parse

        Returns:
            Dictionary with parsed HTML information
        """
        soup = BeautifulSoup(html_string, 'lxml')
        
        parsed_info = {
            "tables": [],
            "images": [],
            "lists": [],
            "structured_divs": [],
            "text_content": soup.get_text(strip=True)[:1000],
            "has_forms": bool(soup.find_all('form')),
            "total_elements": len(soup.find_all())
        }

        for table in soup.find_all('table'):
            headers = []
            rows = []
            
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            for row in table.find_all('tr')[1:]:
                row_data = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                if row_data:
                    rows.append(row_data)
            
            if headers or rows:
                parsed_info["tables"].append({
                    "headers": headers,
                    "rows": rows[:5],
                    "total_rows": len(rows)
                })

        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src.startswith('data:image'):
                parsed_info["images"].append({
                    "alt": img.get('alt', ''),
                    "src_preview": src[:100] + "..." if len(src) > 100 else src,
                    "is_base64": True
                })

        for ul in soup.find_all(['ul', 'ol']):
            items = [li.get_text(strip=True) for li in ul.find_all('li')]
            if len(items) > 2:
                parsed_info["lists"].append({
                    "type": ul.name,
                    "items": items[:5],
                    "total_items": len(items)
                })

        divs_with_class = soup.find_all('div', class_=True)
        class_counts = {}
        for div in divs_with_class:
            classes = ' '.join(div.get('class', []))
            class_counts[classes] = class_counts.get(classes, 0) + 1
        
        for class_name, count in class_counts.items():
            if count > 2:
                sample_div = soup.find('div', class_=class_name.split())
                if sample_div:
                    parsed_info["structured_divs"].append({
                        "class": class_name,
                        "count": count,
                        "sample_text": sample_div.get_text(strip=True)[:200]
                    })

        return parsed_info

    def _extract_images_for_llm(self, html_string: str) -> List[Dict[str, str]]:
        """
        Extract base64 images from HTML for LLM analysis.

        Args:
            html_string: Raw HTML string

        Returns:
            List of image data dictionaries
        """
        soup = BeautifulSoup(html_string, 'lxml')
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src.startswith('data:image'):
                try:
                    match = re.match(r'data:image/([^;]+);base64,(.+)', src)
                    if match:
                        image_type, base64_data = match.groups()
                        images.append({
                            "alt": img.get('alt', ''),
                            "type": image_type,
                            "data_url": src,
                            "base64_data": base64_data
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse image data: {e}")
        
        return images

    async def analyze_html_structure(
        self, html_string: str, focus_areas: Optional[List[str]] = None
    ) -> tuple[HTMLAnalysisResponse, Dict[str, Any]]:
        """
        Analyze HTML to identify extractable datapoints.

        Args:
            html_string: HTML string to analyze
            focus_areas: Optional list of areas to focus on

        Returns:
            Tuple of (HTMLAnalysisResponse with identified datapoints, cost metadata dict)
        """
        parsed_info = self._parse_html_content(html_string)
        
        analysis_task = f"""Analyze this HTML content to identify all structured data that can be extracted and converted to tabular format suitable for SQL databases.

HTML Content Analysis:
- Total HTML elements: {parsed_info['total_elements']}
- Tables found: {len(parsed_info['tables'])}
- Lists found: {len(parsed_info['lists'])}
- Images found: {len(parsed_info['images'])}
- Structured divs: {len(parsed_info['structured_divs'])}
- Has forms: {parsed_info['has_forms']}

Sample content: {parsed_info['text_content']}

Focus on identifying:
1. HTML Tables - Extract with proper column headers and data rows
2. Structured Lists - Extract items that follow consistent patterns
3. Repeated div structures - Extract data from repeated layouts (cards, profiles, etc.)
4. Embedded images - Extract base64 images for visual data analysis
5. Key-value pairs - Extract structured information from definition lists or spans
6. Form data - Extract field names and default values

{f"Specifically focus on: {', '.join(focus_areas)}" if focus_areas else ""}

DATA TYPE RULES (use only these):
- Use 'table' for: HTML tables or any tabular data structure
- Use 'list' for: Ordered/unordered lists with consistent item patterns
- Use 'key_value_pairs' for: Definition lists, structured spans, or form fields
- Use 'image_data' for: Embedded base64 images that contain extractable data
- Use 'structured_text' for: Repeated div patterns or other structured content

For each identified datapoint:
- Provide a descriptive name using snake_case (e.g., 'product_catalog_table')
- Specify the appropriate data type from the rules above
- Indicate where it's located in the HTML structure
- Define schema_fields as a list where each field has:
  * name: snake_case field name (e.g., 'product_name', 'price')
  * type: data type ('string', 'int', 'float', etc.)
  * description: what the field contains
  * optional: true (always set to true for flexibility)

CRITICAL: 
- We want the RAW DATA from HTML structures, not descriptions
- Each datapoint should result in MULTIPLE ROWS when extracted
- For tables, extract the actual cell values
- For lists, extract individual items as separate rows
- For repeated structures, extract each instance as a row"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_task},
                    {"type": "text", "text": f"HTML Content:\n{html_string[:8000]}{'...' if len(html_string) > 8000 else ''}"},
                ],
            }
        ]

        result = await chat_async(
            provider=self.analysis_provider,
            model=self.analysis_model,
            messages=messages,
            temperature=self.temperature,
            response_format=HTMLAnalysisResponse,
        )

        if not result.content:
            raise Exception("HTML analysis failed: No response content")

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
        if datapoint.data_type == "table":
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
        elif datapoint.data_type == "list":
            return create_model(
                datapoint.name,
                items=(
                    List[Dict[str, Optional[Union[str, int, float]]]],
                    Field(description="List items with extracted fields"),
                ),
                item_count=(
                    Optional[int],
                    Field(default=None, description="Number of items"),
                ),
            )
        elif datapoint.data_type == "image_data":
            return create_model(
                datapoint.name,
                columns=(
                    List[str],
                    Field(description="Column names for the extracted data"),
                ),
                data=(
                    List[List[Optional[Union[str, int, float]]]],
                    Field(
                        description="Image data as array of arrays (each inner array is a row)"
                    ),
                ),
                image_description=(
                    Optional[str],
                    Field(default=None, description="Description of the image content"),
                ),
                row_count=(
                    Optional[int],
                    Field(default=None, description="Number of data points"),
                ),
            )
        elif datapoint.data_type == "key_value_pairs":
            field_definitions = {}
            for field_info in datapoint.schema_fields:
                field_name = field_info.name.replace(" ", "_").lower()
                field_description = field_info.description
                
                python_type = str
                if field_info.type.lower() in ["int", "integer"]:
                    python_type = int
                elif field_info.type.lower() in ["float", "decimal", "number"]:
                    python_type = float
                elif field_info.type.lower() in ["bool", "boolean"]:
                    python_type = bool
                
                if field_info.optional:
                    python_type = Optional[python_type]
                
                field_definitions[field_name] = (
                    python_type,
                    Field(description=field_description),
                )
            
            return create_model(datapoint.name, **field_definitions)
        else:
            return create_model(
                datapoint.name,
                columns=(
                    List[str],
                    Field(description="Column names for the extracted data"),
                ),
                data=(
                    List[List[Optional[Union[str, int, float]]]],
                    Field(
                        description="Structured data as array of arrays (each inner array is a row)"
                    ),
                ),
                content_type=(
                    Optional[str],
                    Field(default=datapoint.data_type, description="Type of content"),
                ),
                row_count=(
                    Optional[int],
                    Field(default=None, description="Number of data points"),
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
            analysis_metadata: Cost metadata from HTML analysis
            extraction_results: List of extraction results with individual costs

        Returns:
            Dictionary with aggregated totals
        """
        total_cost = analysis_metadata.get("cost_cents", 0.0)
        total_input_tokens = analysis_metadata.get("input_tokens", 0)
        total_output_tokens = analysis_metadata.get("output_tokens", 0)
        total_cached_tokens = analysis_metadata.get("cached_tokens", 0)

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
        html_string: str,
        datapoint: DataPointIdentification,
        schema: Type[BaseModel],
    ) -> DataExtractionResult:
        """
        Extract a single datapoint from the HTML.

        Args:
            html_string: HTML string content
            datapoint: Datapoint to extract
            schema: Pydantic schema for extraction

        Returns:
            DataExtractionResult
        """
        extraction_task = f"""Extract the following data from this HTML content:

Datapoint: {datapoint.name}
Description: {datapoint.description}
Type: {datapoint.data_type}
Location hint: {datapoint.location_hint}

IMPORTANT EXTRACTION GUIDELINES:
1. For HTML tables:
   - Extract data in COLUMNAR FORMAT to minimize tokens:
     * 'columns' field: array of column names from table headers
     * 'data' field: array of arrays, where each inner array is a row
   - The order of values in each row MUST match the order of columns
   - Example: columns: ["name", "price", "stock"], data: [["Product A", 29.99, 100], ["Product B", 39.99, 50]]
   - If a cell is empty, use null
   - Extract numbers as pure numeric values without currency symbols or commas

2. For lists:
   - Extract each list item as a separate object with identified fields
   - Look for patterns in list items to extract structured data

3. For embedded images (base64):
   - Analyze the image content and extract any visible data
   - Convert charts, graphs, or tables in images to structured data
   - Use the same columnar format as tables

4. For key-value pairs:
   - Use the actual labels/keys from the HTML
   - Extract values in their appropriate data type

5. For all data:
   - Preserve the structure and relationships in the original HTML
   - Make the data SQL-friendly with clear, unambiguous field names
   - Extract dates in ISO format (YYYY-MM-DD) when possible
   - For percentages, store as decimals (0.15 for 15%)

Please extract this data and format it according to the provided schema.
Be precise and include all available data that matches the schema."""

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_task},
                        {"type": "text", "text": f"HTML Content:\n{html_string}"},
                    ],
                }
            ]

            images = self._extract_images_for_llm(html_string)
            if images and datapoint.data_type == "image_data":
                for img in images[:3]:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": img["data_url"]}
                    })

            result = await chat_async(
                provider=self.extraction_provider,
                model=self.extraction_model,
                messages=messages,
                temperature=self.temperature,
                response_format=schema,
            )

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
        html_string: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> HTMLDataExtractionResult:
        """
        Extract all identified datapoints from HTML in parallel.

        Args:
            html_string: HTML string to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            HTMLDataExtractionResult with all extracted data
        """
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Analyzing HTML structure (length: {len(html_string)} chars)")
        analysis, analysis_cost_metadata = await self.analyze_html_structure(
            html_string, focus_areas
        )

        logger.info(f"Step 1 - HTML Analysis completed:")
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

        datapoints_to_extract = analysis.identified_datapoints
        if datapoint_filter:
            datapoints_to_extract = [
                dp for dp in datapoints_to_extract if dp.name in datapoint_filter
            ]

        schemas = {}
        for datapoint in datapoints_to_extract:
            try:
                schema = self._generate_pydantic_schema(datapoint)
                schemas[datapoint.name] = schema
                logger.info(f"Generated schema for {datapoint.name}")
            except Exception as e:
                logger.error(f"Failed to generate schema for {datapoint.name}: {e}")

        extraction_tasks = []
        for datapoint in datapoints_to_extract:
            if datapoint.name in schemas:
                task = self.extract_single_datapoint(
                    html_string, datapoint, schemas[datapoint.name]
                )
                extraction_tasks.append(task)

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

        logger.info(f"Step 2 - Individual extraction costs:")

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

        cost_metadata = self._aggregate_cost_and_token_metadata(
            analysis_cost_metadata, final_results
        )

        end_time = asyncio.get_event_loop().time()

        logger.info(f"HTML Data Extraction completed:")
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

        return HTMLDataExtractionResult(
            html_content_preview=html_string[:500] + "..." if len(html_string) > 500 else html_string,
            html_type=analysis.html_type,
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
        html_string: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all data and return as a dictionary for easy access.

        Args:
            html_string: HTML string to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract

        Returns:
            Dictionary with datapoint names as keys and extracted data as values
        """
        result = await self.extract_all_data(html_string, focus_areas, datapoint_filter)

        extracted_data = {
            "metadata": {
                "html_content_preview": result.html_content_preview,
                "html_type": result.html_type,
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
                data = extraction.extracted_data
                if hasattr(data, "model_dump"):
                    data = data.model_dump()
                extracted_data["data"][extraction.datapoint_name] = data

        return extracted_data


async def extract_html_data(
    html_string: str, focus_areas: Optional[List[str]] = None, **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to extract all data from HTML.

    Args:
        html_string: HTML string to process
        focus_areas: Optional areas to focus on
        **kwargs: Additional arguments for HTMLDataExtractor

    Returns:
        Dictionary with extracted data
    """
    extractor = HTMLDataExtractor(**kwargs)
    return await extractor.extract_as_dict(html_string, focus_areas)
