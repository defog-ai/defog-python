"""
HTML Data Extractor Agent Orchestrator

This module provides intelligent HTML analysis to:
1. Identify interesting datapoints that can be extracted as tabular data
2. Generate appropriate response_format schemas for each datapoint
3. Orchestrate parallel extraction of all identified datapoints
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field, create_model
import bleach
from bs4 import BeautifulSoup, NavigableString
import re

from .utils import chat_async
from .llm_providers import LLMProvider
from .image_data_extractor import ImageDataExtractor
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# Constants for security and performance
MAX_HTML_SIZE_MB = 10  # Maximum HTML size in megabytes
MAX_HTML_SIZE_BYTES = MAX_HTML_SIZE_MB * 1024 * 1024
ALLOWED_TAGS = [
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "div",
    "span",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "a",
    "img",
    "form",
    "input",
    "select",
    "option",
    "textarea",
    "nav",
    "section",
    "article",
    "aside",
    "header",
    "footer",
    "strong",
    "em",
    "b",
    "i",
    "u",
    "code",
    "pre",
    "script",  # Keep script tags to check for JSON-LD
]
ALLOWED_ATTRIBUTES = {
    "*": ["id", "class", "data-*", "aria-*"],
    "a": ["href", "title"],
    "img": ["src", "alt", "title"],
    "input": ["type", "name", "value", "placeholder"],
    "script": ["type"],  # For JSON-LD
}


class SchemaField(BaseModel):
    """Schema field definition for data extraction."""

    name: str = Field(description="Snake_case field name")
    type: str = Field(description="Data type (string, int, float, etc.)")
    description: str = Field(description="What the field contains")
    optional: bool = Field(default=True, description="Whether the field is optional")


class DataPointIdentification(BaseModel):
    """Identified data point in HTML that can be extracted."""

    name: str = Field(
        description="Name of the datapoint (e.g., 'product_inventory_table')"
    )
    description: str = Field(description="Description of what this datapoint contains")
    data_type: str = Field(
        description="Type of data: 'table', 'list', 'key_value_pairs', 'structured_text', 'image'"
    )
    location_hint: str = Field(
        description="Hint about where in the HTML this data is located (e.g., CSS selector, section name)"
    )
    schema_fields: List[SchemaField] = Field(
        description="List of fields for extraction. For tables with multiple columns, use descriptive names."
    )


class HTMLAnalysisResponse(BaseModel):
    """Response from HTML structure analysis."""

    page_type: str = Field(
        description="Type of page (e.g., 'e-commerce', 'article', 'dashboard', 'form', 'report')"
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

    html_content_hash: str
    page_type: str
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
        max_html_size_mb: int = MAX_HTML_SIZE_MB,
        enable_caching: bool = True,
        enable_image_extraction: bool = True,
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
            max_html_size_mb: Maximum HTML size in megabytes
            enable_caching: Whether to enable caching of analysis results
            enable_image_extraction: Whether to extract data from images in HTML
        """
        self.analysis_provider = analysis_provider
        self.analysis_model = analysis_model
        self.extraction_provider = extraction_provider
        self.extraction_model = extraction_model
        self.max_parallel_extractions = max_parallel_extractions
        self.temperature = temperature
        self.max_html_size_bytes = max_html_size_mb * 1024 * 1024
        self.enable_caching = enable_caching
        self._analysis_cache = {} if enable_caching else None
        self.enable_image_extraction = enable_image_extraction

        # Initialize image extractor if enabled
        if enable_image_extraction:
            self.image_extractor = ImageDataExtractor(
                analysis_provider=analysis_provider,
                analysis_model=analysis_model,
                extraction_provider=extraction_provider,
                extraction_model=extraction_model,
                max_parallel_extractions=max_parallel_extractions,
                temperature=temperature,
            )

    async def analyze_html_structure(
        self, html_content: str, focus_areas: Optional[List[str]] = None
    ) -> tuple[HTMLAnalysisResponse, Dict[str, Any]]:
        """
        Analyze HTML to identify extractable datapoints.

        Args:
            html_content: HTML string to analyze
            focus_areas: Optional list of areas to focus on

        Returns:
            Tuple of (HTMLAnalysisResponse with identified datapoints, cost metadata dict)
        """
        analysis_task = f"""Analyze HTML to identify extractable tabular data.

Find all structured data that can be stored in SQL tables:
- Tables (<table>)
- Lists with repeated patterns (<ul>, <ol>, <dl>)
- Repeated elements (product cards, profiles)
- Key-value pairs (metadata, specs)
- Structured text (FAQs, features)
- Forms and navigation menus
- JSON-LD/microdata
{"- Images (<img>) that contain data (charts, tables, diagrams)" if self.enable_image_extraction else ""}

{f"Focus on: {', '.join(focus_areas)}" if focus_areas else ""}

For each datapoint provide:
- name: snake_case (e.g., 'product_inventory')
- data_type: 'table', 'list', 'key_value_pairs', {"'structured_text', or 'image'" if self.enable_image_extraction else "or 'structured_text'"}
- location_hint: CSS selector or description{" (for images, include src attribute)" if self.enable_image_extraction else ""}
- schema_fields: [{{name, type, description, optional: true}}]{" (for images, describe expected data columns)" if self.enable_image_extraction else ""}

Extract RAW DATA values, not descriptions. Each datapoint should yield MULTIPLE ROWS."""

        # Prepare message with HTML content
        messages = [
            {
                "role": "user",
                "content": f"<html_content>\n{html_content}\n</html_content>\n\n{analysis_task}",
            }
        ]

        # Call LLM with structured response
        result = await chat_async(
            provider=self.analysis_provider,
            model=self.analysis_model,
            messages=messages,
            temperature=self.temperature,
            response_format=HTMLAnalysisResponse,
        )

        # chat_async returns the response directly
        if not result.content:
            raise Exception("HTML analysis failed: No response content")

        # Extract cost metadata from LLMResponse
        cost_metadata = {
            "cost_cents": result.cost_in_cents or 0.0,
            "input_tokens": result.input_tokens or 0,
            "output_tokens": result.output_tokens or 0,
            "cached_tokens": result.cached_input_tokens or 0,
        }

        return result.content, cost_metadata

    def _sanitize_and_preprocess_html(self, html_content: str) -> str:
        """
        Sanitize and preprocess HTML to improve security and performance.

        Args:
            html_content: Raw HTML string

        Returns:
            Sanitized and preprocessed HTML string
        """
        # First pass: Use bleach for security sanitization
        sanitized = bleach.clean(
            html_content,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            strip=True,
            strip_comments=False,  # Keep comments as they might contain data
        )

        # Second pass: Use BeautifulSoup for preprocessing
        soup = BeautifulSoup(sanitized, "html.parser")

        # Remove script tags except JSON-LD
        for script in soup.find_all("script"):
            if script.get("type") != "application/ld+json":
                script.decompose()

        # Remove style tags and inline styles to reduce noise
        for style in soup.find_all("style"):
            style.decompose()
        for tag in soup.find_all(style=True):
            del tag["style"]

        # Remove empty tags (except those that might be self-closing)
        for tag in soup.find_all():
            if tag.name not in ["img", "input", "br", "hr", "meta", "link"]:
                if not tag.get_text(strip=True) and not tag.find_all():
                    tag.decompose()

        # Normalize whitespace
        for element in soup.find_all(string=True):
            if isinstance(element, NavigableString):
                cleaned = re.sub(r"\s+", " ", element.string)
                element.replace_with(cleaned)

        return str(soup)

    def _extract_image_urls(
        self, html_content: str, base_url: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Extract image URLs from HTML content, resolving relative URLs if base_url is provided.

        Args:
            html_content: HTML string
            base_url: Optional base URL for resolving relative image paths

        Returns:
            Dictionary mapping original src to full URLs
        """
        soup = BeautifulSoup(html_content, "html.parser")
        image_urls = {}

        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                # Store both the src attribute and any alt/title text for context
                alt_text = img.get("alt", "")
                title_text = img.get("title", "")
                context = alt_text or title_text or ""

                # Resolve relative URLs if base_url is provided
                resolved_url = src
                if base_url and not urlparse(src).scheme:
                    # Handle relative URLs
                    resolved_url = urljoin(base_url, src)

                # Use original src as key for easy lookup in location hints
                image_urls[src] = {
                    "url": resolved_url,
                    "context": context,
                    "element": str(img),
                }

        return image_urls

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
        if datapoint.data_type == "table" or datapoint.data_type == "list":
            # For tables and lists, use columnar format for efficiency
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
                source_element=(
                    Optional[str],
                    Field(
                        default=None,
                        description="HTML element or selector where data was found",
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
        else:
            # For structured_text and other types, use a flexible format
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
                source_element=(
                    Optional[str],
                    Field(
                        default=None,
                        description="HTML element or selector where data was found",
                    ),
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
        html_content: str,
        datapoint: DataPointIdentification,
        schema: Type[BaseModel],
    ) -> DataExtractionResult:
        """
        Extract a single datapoint from the HTML.

        Args:
            html_content: HTML string
            datapoint: Datapoint to extract
            schema: Pydantic schema for extraction

        Returns:
            DataExtractionResult
        """
        extraction_task = f"""Extract: {datapoint.name} ({datapoint.data_type})
Location: {datapoint.location_hint}

For tables/lists: Use columnar format with 'columns' and 'data' arrays.
For key-value pairs: Extract fields with proper types (numbers without symbols).

Extract RAW VALUES only. Empty cells = null."""

        try:
            # Prepare message with HTML
            messages = [
                {
                    "role": "user",
                    "content": f"<html_content>\n{html_content}\n</html_content>\n\n{extraction_task}",
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

    async def _extract_image_data(
        self, image_url: str, datapoint: DataPointIdentification
    ) -> DataExtractionResult:
        """
        Extract data from an image using the image data extractor.

        Args:
            image_url: URL of the image
            datapoint: Datapoint identification with expected schema

        Returns:
            DataExtractionResult
        """
        try:
            # Use the image extractor to analyze and extract data
            result = await self.image_extractor.extract_all_data(
                image_url=image_url,
                focus_areas=[datapoint.description],
                datapoint_filter=[datapoint.name],
            )

            # Find the extraction result for our datapoint
            for extraction in result.extraction_results:
                if extraction.datapoint_name == datapoint.name and extraction.success:
                    return extraction

            # If we didn't find a matching extraction, try without filter
            result = await self.image_extractor.extract_all_data(
                image_url=image_url, focus_areas=[datapoint.description]
            )

            # Return the first successful extraction
            for extraction in result.extraction_results:
                if extraction.success:
                    # Update the datapoint name to match what was expected
                    extraction.datapoint_name = datapoint.name
                    return extraction

            # No successful extractions
            return DataExtractionResult(
                datapoint_name=datapoint.name,
                success=False,
                error="No data could be extracted from the image",
                cost_cents=result.total_cost_cents,
                input_tokens=result.metadata.get("total_input_tokens", 0),
                output_tokens=result.metadata.get("total_output_tokens", 0),
                cached_tokens=result.metadata.get("total_cached_tokens", 0),
            )

        except Exception as e:
            logger.error(f"Error extracting image data for {datapoint.name}: {e}")
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
        html_content: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
        base_url: Optional[str] = None,
    ) -> HTMLDataExtractionResult:
        """
        Extract all identified datapoints from HTML in parallel.

        Args:
            html_content: HTML string to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract
            base_url: Optional base URL for resolving relative image paths

        Returns:
            HTMLDataExtractionResult with all extracted data
        """
        start_time = asyncio.get_event_loop().time()

        # Validate HTML size
        html_size = len(html_content.encode("utf-8"))
        if html_size > self.max_html_size_bytes:
            raise ValueError(
                f"HTML content exceeds maximum size limit of {self.max_html_size_bytes / (1024 * 1024):.2f} MB. "
                f"Actual size: {html_size / (1024 * 1024):.2f} MB"
            )

        # Generate a secure hash for the HTML content
        content_hash = hashlib.sha256(html_content.encode()).hexdigest()[:16]

        # Sanitize and preprocess HTML
        sanitized_html = self._sanitize_and_preprocess_html(html_content)

        # Check cache if enabled
        cache_key = f"{content_hash}:{','.join(focus_areas or [])}:{','.join(datapoint_filter or [])}"
        if self.enable_caching and cache_key in self._analysis_cache:
            logger.info(f"Using cached analysis for hash: {content_hash}")
            cached_result = self._analysis_cache[cache_key]
            return cached_result

        # Step 1: Analyze HTML structure
        logger.info(f"Analyzing HTML structure (hash: {content_hash})")
        analysis, analysis_cost_metadata = await self.analyze_html_structure(
            sanitized_html, focus_areas
        )

        logger.info("Step 1 - HTML Analysis completed:")
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

        # Extract image URLs if we have image datapoints
        image_urls = {}
        if self.enable_image_extraction and any(
            dp.data_type == "image" for dp in datapoints_to_extract
        ):
            image_urls = self._extract_image_urls(sanitized_html, base_url)
            logger.info(f"Found {len(image_urls)} images in HTML")

        # Step 3: Extract data in parallel
        extraction_tasks = []
        image_extraction_tasks = []

        for datapoint in datapoints_to_extract:
            if datapoint.name in schemas:
                if datapoint.data_type == "image" and self.enable_image_extraction:
                    # Extract image URL from location hint
                    # The location hint should contain the src attribute
                    src_match = None
                    for src, img_info in image_urls.items():
                        if src in datapoint.location_hint:
                            src_match = src
                            break

                    if src_match:
                        # Create image extraction task
                        image_url = image_urls[src_match]["url"]
                        task = self._extract_image_data(image_url, datapoint)
                        image_extraction_tasks.append((datapoint, task))
                    else:
                        logger.warning(
                            f"Could not find image URL for datapoint {datapoint.name}"
                        )
                else:
                    # Regular HTML extraction
                    task = self.extract_single_datapoint(
                        sanitized_html, datapoint, schemas[datapoint.name]
                    )
                    extraction_tasks.append(task)

        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel_extractions)

        async def extract_with_limit(task):
            async with semaphore:
                return await task

        # Combine all extraction tasks
        all_tasks = extraction_tasks + [task for _, task in image_extraction_tasks]

        logger.info(
            f"Step 2 - Starting parallel extraction of {len(all_tasks)} datapoints "
            f"({len(extraction_tasks)} HTML, {len(image_extraction_tasks)} images)"
        )
        extraction_results = await asyncio.gather(
            *[extract_with_limit(task) for task in all_tasks],
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
        logger.info("HTML Data Extraction completed:")
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

        result = HTMLDataExtractionResult(
            html_content_hash=content_hash,
            page_type=analysis.page_type,
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
        html_content: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract all data and return as a dictionary for easy access.

        Args:
            html_content: HTML string to process
            focus_areas: Optional areas to focus analysis on
            datapoint_filter: Optional list of datapoint names to extract
            base_url: Optional base URL for resolving relative image paths

        Returns:
            Dictionary with datapoint names as keys and extracted data as values
        """
        result = await self.extract_all_data(
            html_content, focus_areas, datapoint_filter, base_url
        )

        extracted_data = {
            "metadata": {
                "html_content_hash": result.html_content_hash,
                "page_type": result.page_type,
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
async def extract_html_data(
    html_content: str, focus_areas: Optional[List[str]] = None, **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to extract all data from HTML.

    Args:
        html_content: HTML string to process
        focus_areas: Optional areas to focus on
        **kwargs: Additional arguments for HTMLDataExtractor

    Returns:
        Dictionary with extracted data
    """
    extractor = HTMLDataExtractor(**kwargs)
    return await extractor.extract_as_dict(html_content, focus_areas)
