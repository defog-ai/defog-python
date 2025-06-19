"""
Tests for HTMLDataExtractor
"""

import asyncio
import pytest
import os
from typing import Dict, Any

from defog.llm import HTMLDataExtractor, extract_html_data
from defog.llm.html_data_extractor import HTMLDataExtractionResult


# Sample HTML content for testing
SAMPLE_HTML = {
    "simple_table": """
    <html>
    <body>
        <h1>Product Inventory</h1>
        <table>
            <thead>
                <tr>
                    <th>Product Name</th>
                    <th>Price</th>
                    <th>Stock</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Widget A</td>
                    <td>$19.99</td>
                    <td>150</td>
                </tr>
                <tr>
                    <td>Widget B</td>
                    <td>$29.99</td>
                    <td>75</td>
                </tr>
                <tr>
                    <td>Widget C</td>
                    <td>$39.99</td>
                    <td>0</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """,
    "product_cards": """
    <html>
    <body>
        <div class="products">
            <div class="product-card" data-product-id="1">
                <h3>Premium Widget</h3>
                <p class="price">$49.99</p>
                <p class="description">High-quality widget for professionals</p>
                <span class="in-stock">In Stock</span>
            </div>
            <div class="product-card" data-product-id="2">
                <h3>Standard Widget</h3>
                <p class="price">$29.99</p>
                <p class="description">Everyday widget for general use</p>
                <span class="out-of-stock">Out of Stock</span>
            </div>
            <div class="product-card" data-product-id="3">
                <h3>Budget Widget</h3>
                <p class="price">$19.99</p>
                <p class="description">Affordable widget for basic needs</p>
                <span class="in-stock">In Stock</span>
            </div>
        </div>
    </body>
    </html>
    """,
    "mixed_content": """
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": "Widget Corp",
            "foundingDate": "2020",
            "numberOfEmployees": 50
        }
        </script>
    </head>
    <body>
        <div class="company-info">
            <h1>Widget Corp</h1>
            <dl>
                <dt>Founded</dt>
                <dd>2020</dd>
                <dt>Employees</dt>
                <dd>50</dd>
                <dt>Revenue</dt>
                <dd>$5M</dd>
            </dl>
        </div>
        
        <h2>Team Members</h2>
        <ul class="team-list">
            <li data-role="CEO">
                <span class="name">John Doe</span>
                <span class="email">john@widgetcorp.com</span>
            </li>
            <li data-role="CTO">
                <span class="name">Jane Smith</span>
                <span class="email">jane@widgetcorp.com</span>
            </li>
            <li data-role="CFO">
                <span class="name">Bob Johnson</span>
                <span class="email">bob@widgetcorp.com</span>
            </li>
        </ul>
        
        <table class="quarterly-revenue">
            <caption>Quarterly Revenue (in thousands)</caption>
            <thead>
                <tr>
                    <th>Quarter</th>
                    <th>Revenue</th>
                    <th>Growth %</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Q1 2024</td>
                    <td>$1,200</td>
                    <td>15%</td>
                </tr>
                <tr>
                    <td>Q2 2024</td>
                    <td>$1,380</td>
                    <td>15%</td>
                </tr>
                <tr>
                    <td>Q3 2024</td>
                    <td>$1,587</td>
                    <td>15%</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """,
}


class TestHTMLDataExtractorUnit:
    """Unit tests for HTMLDataExtractor"""

    def test_initialization(self):
        """Test extractor initialization with different configurations"""
        # Test with default settings
        extractor = HTMLDataExtractor()
        assert extractor.analysis_provider == "anthropic"
        assert extractor.analysis_model == "claude-sonnet-4-20250514"
        assert extractor.extraction_provider == "anthropic"
        assert extractor.extraction_model == "claude-sonnet-4-20250514"
        assert extractor.max_parallel_extractions == 5
        assert extractor.temperature == 0.0

        # Test with custom settings
        extractor = HTMLDataExtractor(
            analysis_provider="openai",
            analysis_model="gpt-4.1",
            extraction_provider="gemini",
            extraction_model="gemini-pro",
            max_parallel_extractions=3,
            temperature=0.5,
        )
        assert extractor.analysis_provider == "openai"
        assert extractor.analysis_model == "gpt-4.1"
        assert extractor.extraction_provider == "gemini"
        assert extractor.extraction_model == "gemini-pro"
        assert extractor.max_parallel_extractions == 3
        assert extractor.temperature == 0.5


@pytest.mark.asyncio
class TestHTMLDataExtractorE2E:
    """End-to-end tests for HTMLDataExtractor with real API calls
    
    Provider distribution across tests:
    - test_simple_table_extraction: Anthropic (claude-sonnet-4)
    - test_product_cards_extraction: OpenAI (gpt-4.1)
    - test_mixed_content_extraction: Gemini (gemini-2.5-pro)
    - test_datapoint_filtering: Mixed (Anthropic for analysis, OpenAI for extraction)
    - test_convenience_function: Gemini (gemini-2.5-pro)
    - test_error_handling: OpenAI (gpt-4.1)
    """

    async def test_simple_table_extraction(self):
        """Test extraction from a simple HTML table with Anthropic"""
        # Skip if no API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor(
            analysis_provider="anthropic",
            analysis_model="claude-sonnet-4-20250514",
            extraction_provider="anthropic",
            extraction_model="claude-sonnet-4-20250514",
        )

        result = await extractor.extract_all_data(SAMPLE_HTML["simple_table"])

        # Verify results
        assert isinstance(result, HTMLDataExtractionResult)
        assert result.page_type in ["report", "inventory", "product_list", "table"]
        assert result.total_datapoints_identified > 0
        assert result.successful_extractions > 0
        assert result.total_cost_cents > 0

        # Verify extraction data
        for extraction in result.extraction_results:
            if extraction.success:
                assert extraction.extracted_data is not None
                # Check if it's a table format
                data = extraction.extracted_data
                if hasattr(data, "model_dump"):
                    data = data.model_dump()
                
                # Should have columns and data for table extraction
                if "columns" in data and "data" in data:
                    assert len(data["columns"]) > 0
                    assert len(data["data"]) > 0
                    # Should have extracted 3 products
                    if "product" in extraction.datapoint_name.lower():
                        assert len(data["data"]) == 3

    async def test_product_cards_extraction(self):
        """Test extraction from product cards (list format) with OpenAI"""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        extractor = HTMLDataExtractor(
            analysis_provider="openai",
            analysis_model="gpt-4.1",
            extraction_provider="openai",
            extraction_model="gpt-4.1",
        )

        result = await extractor.extract_all_data(SAMPLE_HTML["product_cards"])

        # Verify results
        assert isinstance(result, HTMLDataExtractionResult)
        assert result.page_type in ["e-commerce", "product_list", "catalog"]
        assert result.total_datapoints_identified > 0
        assert result.successful_extractions > 0

        # Check extracted data
        dict_result = await extractor.extract_as_dict(SAMPLE_HTML["product_cards"])
        assert "data" in dict_result
        assert len(dict_result["data"]) > 0

        # Should identify product cards as a datapoint
        product_data_found = False
        for key, value in dict_result["data"].items():
            if "product" in key.lower():
                product_data_found = True
                # Should have extracted 3 products
                if "data" in value:
                    assert len(value["data"]) == 3
                elif "items" in value:
                    assert len(value["items"]) == 3

        assert product_data_found, "No product data was extracted"

    async def test_mixed_content_extraction(self):
        """Test extraction from mixed HTML content with Gemini"""
        # Skip if no API key
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        extractor = HTMLDataExtractor(
            analysis_provider="gemini",
            analysis_model="gemini-2.5-pro",
            extraction_provider="gemini",
            extraction_model="gemini-2.5-pro",
        )

        # Focus on specific areas
        result = await extractor.extract_all_data(
            SAMPLE_HTML["mixed_content"],
            focus_areas=["company information", "team members", "revenue data"],
        )

        # Print extraction details
        for extraction in result.extraction_results:
            print(f"\nDatapoint: {extraction.datapoint_name}")
            print(f"  Success: {extraction.success}")
            if extraction.error:
                print(f"  Error: {extraction.error}")
            if extraction.extracted_data:
                print(f"  Data type: {type(extraction.extracted_data)}")

        # Verify results
        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified >= 3  # At least 3 datapoints
        assert result.successful_extractions > 0

        # Extract as dictionary for easier verification
        dict_result = await extractor.extract_as_dict(SAMPLE_HTML["mixed_content"])
        
        for key, value in dict_result["data"].items():
            print(f"\nDatapoint '{key}':")
            print(f"  Type: {type(value)}")
            if isinstance(value, dict):
                print(f"  Keys: {list(value.keys())}")
        
        # Should have extracted multiple datapoints
        assert len(dict_result["data"]) >= 2

        # Check for specific data types
        datapoint_types = set()
        for key in dict_result["data"].keys():
            if "company" in key.lower() or "info" in key.lower():
                datapoint_types.add("company_info")
            elif "team" in key.lower() or "member" in key.lower():
                datapoint_types.add("team")
            elif "revenue" in key.lower() or "quarterly" in key.lower():
                datapoint_types.add("revenue")

        print(f"\n=== DEBUG: Identified datapoint types ===")
        print(f"Types found: {datapoint_types}")

        # Should have identified at least 2 different types of data
        assert len(datapoint_types) >= 2

    async def test_datapoint_filtering(self):
        """Test filtering specific datapoints with mixed providers"""
        # Skip if no API keys
        if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY or OPENAI_API_KEY not set")

        # Use Anthropic for analysis, OpenAI for extraction
        extractor = HTMLDataExtractor(
            analysis_provider="anthropic",
            analysis_model="claude-sonnet-4-20250514",
            extraction_provider="openai",
            extraction_model="gpt-4.1",
        )

        # First, analyze to get datapoint names
        analysis_result = await extractor.extract_all_data(
            SAMPLE_HTML["mixed_content"]
        )
        
        # Get first datapoint name
        if analysis_result.extraction_results:
            first_datapoint_name = analysis_result.extraction_results[0].datapoint_name
            
            # Extract only that specific datapoint
            filtered_result = await extractor.extract_all_data(
                SAMPLE_HTML["mixed_content"],
                datapoint_filter=[first_datapoint_name],
            )
            
            assert filtered_result.successful_extractions <= 1
            if filtered_result.extraction_results:
                assert all(
                    r.datapoint_name == first_datapoint_name
                    for r in filtered_result.extraction_results
                    if r.success
                )

    async def test_convenience_function(self):
        """Test the convenience function with Gemini"""
        # Skip if no API key
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        result = await extract_html_data(
            SAMPLE_HTML["simple_table"],
            analysis_provider="gemini",
            analysis_model="gemini-2.5-pro",
            extraction_provider="gemini",
            extraction_model="gemini-2.5-pro",
            temperature=0.1,
        )

        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data" in result
        assert len(result["data"]) > 0

    async def test_error_handling(self):
        """Test error handling with invalid HTML using OpenAI"""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        extractor = HTMLDataExtractor(
            analysis_provider="openai",
            analysis_model="gpt-4.1",
            extraction_provider="openai",
            extraction_model="gpt-4.1",
        )

        # Test with empty HTML
        result = await extractor.extract_all_data("")
        assert isinstance(result, HTMLDataExtractionResult)
        # Should still complete but might not find datapoints
        assert result.total_datapoints_identified >= 0

        # Test with invalid HTML structure
        invalid_html = "<html><body><table><tr><td>Broken</table>"
        result = await extractor.extract_all_data(invalid_html)
        assert isinstance(result, HTMLDataExtractionResult)
        # Should handle gracefully
        assert result.total_cost_cents >= 0


if __name__ == "__main__":
    # Run a simple test
    async def main():
        extractor = HTMLDataExtractor()
        result = await extractor.extract_as_dict(SAMPLE_HTML["simple_table"])
        print("Extraction Result:")
        print(f"Page Type: {result['metadata']['page_type']}")
        print(f"Extracted {len(result['data'])} datapoints")
        for name, data in result["data"].items():
            print(f"\n{name}:")
            print(data)

    asyncio.run(main())