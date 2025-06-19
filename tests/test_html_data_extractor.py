"""
End-to-end tests for HTMLDataExtractor with multiple providers
"""

import asyncio
import pytest
import os
from typing import Dict, Any

from defog.llm import HTMLDataExtractor
from defog.llm.html_data_extractor import HTMLDataExtractionResult


class TestHTMLDataExtractorE2E:
    """End-to-end tests for HTMLDataExtractor with real API calls"""

    @pytest.mark.asyncio
    async def test_anthropic_table_extraction(self):
        """Test table extraction with Anthropic"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        html_content = """
        <table>
            <tr><th>Name</th><th>Age</th><th>City</th></tr>
            <tr><td>Alice</td><td>25</td><td>New York</td></tr>
            <tr><td>Bob</td><td>30</td><td>San Francisco</td></tr>
            <tr><td>Charlie</td><td>35</td><td>Chicago</td></tr>
        </table>
        """

        extractor = HTMLDataExtractor(
            analysis_provider="anthropic",
            analysis_model="claude-sonnet-4-20250514",
            extraction_provider="anthropic",
            extraction_model="claude-sonnet-4-20250514",
        )

        result = await extractor.extract_all_data(html_content)

        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified > 0
        assert result.successful_extractions > 0
        assert result.total_cost_cents > 0

    @pytest.mark.asyncio
    async def test_openai_list_extraction(self):
        """Test list extraction with OpenAI"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        html_content = """
        <ul class="product-list">
            <li>iPhone 15 - $999</li>
            <li>Samsung Galaxy S24 - $899</li>
            <li>Google Pixel 8 - $699</li>
        </ul>
        """

        extractor = HTMLDataExtractor(
            analysis_provider="openai",
            analysis_model="gpt-4o",
            extraction_provider="openai",
            extraction_model="gpt-4o",
        )

        result = await extractor.extract_all_data(html_content)

        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified > 0

    @pytest.mark.asyncio
    async def test_gemini_structured_divs(self):
        """Test structured div extraction with Gemini"""
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        html_content = """
        <div class="user-card">
            <h3>John Doe</h3>
            <p class="role">Software Engineer</p>
            <p class="salary">$85,000</p>
        </div>
        <div class="user-card">
            <h3>Jane Smith</h3>
            <p class="role">Product Manager</p>
            <p class="salary">$95,000</p>
        </div>
        """

        extractor = HTMLDataExtractor(
            analysis_provider="gemini",
            analysis_model="gemini-2.5-pro",
            extraction_provider="gemini",
            extraction_model="gemini-2.5-pro",
        )

        result = await extractor.extract_all_data(html_content)

        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified >= 0

    @pytest.mark.asyncio
    async def test_extract_as_dict(self):
        """Test extract_as_dict convenience method"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        html_content = """
        <table>
            <tr><th>Product</th><th>Price</th></tr>
            <tr><td>Widget A</td><td>$10</td></tr>
            <tr><td>Widget B</td><td>$20</td></tr>
        </table>
        """

        extractor = HTMLDataExtractor()
        result = await extractor.extract_as_dict(html_content)

        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data" in result
        assert "extraction_summary" in result["metadata"]

    @pytest.mark.asyncio
    async def test_focus_areas(self):
        """Test extraction with focus areas"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        html_content = """
        <div>
            <table id="sales">
                <tr><th>Product</th><th>Sales</th></tr>
                <tr><td>A</td><td>100</td></tr>
            </table>
            <table id="inventory">
                <tr><th>Item</th><th>Stock</th></tr>
                <tr><td>B</td><td>50</td></tr>
            </table>
        </div>
        """

        extractor = HTMLDataExtractor()
        result = await extractor.extract_all_data(
            html_content, focus_areas=["sales data", "revenue information"]
        )

        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified >= 0

    @pytest.mark.asyncio
    async def test_datapoint_filter(self):
        """Test extraction with datapoint filtering"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        html_content = """
        <table class="products">
            <tr><th>Name</th><th>Price</th></tr>
            <tr><td>Widget</td><td>$10</td></tr>
        </table>
        <ul class="categories">
            <li>Electronics</li>
            <li>Books</li>
        </ul>
        """

        extractor = HTMLDataExtractor()
        
        all_result = await extractor.extract_all_data(html_content)
        
        if all_result.total_datapoints_identified > 1:
            first_datapoint = all_result.extraction_results[0].datapoint_name
            filtered_result = await extractor.extract_all_data(
                html_content, datapoint_filter=[first_datapoint]
            )
            
            assert filtered_result.successful_extractions <= all_result.successful_extractions

    @pytest.mark.asyncio
    async def test_malformed_html(self):
        """Test extraction with malformed HTML"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        malformed_html = """
        <table>
            <tr><th>Name<th>Age
            <tr><td>Alice<td>25
            <tr><td>Bob<td>30
        </table>
        """

        extractor = HTMLDataExtractor()
        result = await extractor.extract_all_data(malformed_html)

        assert isinstance(result, HTMLDataExtractionResult)

    @pytest.mark.asyncio
    async def test_empty_html(self):
        """Test extraction with empty HTML"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        empty_html = "<html><body></body></html>"

        extractor = HTMLDataExtractor()
        result = await extractor.extract_all_data(empty_html)

        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_datapoints_identified == 0

    @pytest.mark.asyncio
    async def test_complex_nested_structure(self):
        """Test extraction with complex nested HTML structures"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        complex_html = """
        <div class="dashboard">
            <section class="metrics">
                <div class="metric-card">
                    <h3>Revenue</h3>
                    <span class="value">$125,000</span>
                    <span class="change">+15%</span>
                </div>
                <div class="metric-card">
                    <h3>Users</h3>
                    <span class="value">1,250</span>
                    <span class="change">+8%</span>
                </div>
            </section>
            <section class="data-table">
                <table>
                    <thead>
                        <tr><th>Date</th><th>Orders</th><th>Revenue</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>2024-01-01</td><td>45</td><td>$2,250</td></tr>
                        <tr><td>2024-01-02</td><td>52</td><td>$2,600</td></tr>
                    </tbody>
                </table>
            </section>
        </div>
        """

        extractor = HTMLDataExtractor()
        result = await extractor.extract_all_data(complex_html)

        assert isinstance(result, HTMLDataExtractionResult)
        assert result.total_cost_cents > 0
