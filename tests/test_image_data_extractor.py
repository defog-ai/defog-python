"""
End-to-end tests for ImageDataExtractor with multiple providers
"""

import pytest
import os

from defog.llm import ImageDataExtractor
from defog.llm.image_data_extractor import ImageDataExtractionResult


# Test image URLs
TEST_IMAGES = {
    "chart": "https://data.europa.eu/apps/data-visualisation-guide/A%20deep%20dive%20into%20bar%20charts%20047791ead2e848bdb3d0afcd1bf2bd4a/electricity-bars-karim.png",
    "table": "https://support.microsoft.com/images/en-us/3dd2b79b-9160-403d-9967-af893d17b580",
    "multi_chart": "https://www.slideteam.net/media/catalog/product/cache/1280x720/m/u/multiple_charts_for_business_growth_presentation_images_Slide01.jpg",
}


class TestImageDataExtractorE2E:
    """End-to-end tests for ImageDataExtractor with real API calls"""

    @pytest.mark.asyncio
    async def test_anthropic_full_extraction(self):
        """Test full extraction with Anthropic provider"""
        # Skip if no API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = ImageDataExtractor(
            analysis_provider="anthropic",
            analysis_model="claude-sonnet-4-20250514",
            extraction_provider="anthropic",
            extraction_model="claude-sonnet-4-20250514",
        )

        result = await extractor.extract_all_data(TEST_IMAGES["chart"])

        # Verify results
        assert isinstance(result, ImageDataExtractionResult)
        assert result.image_type in ["chart", "bar_chart", "graph", "infographic"]
        assert result.total_datapoints_identified > 0
        assert result.successful_extractions > 0
        assert result.total_cost_cents > 0

        # Verify extraction data
        for extraction in result.extraction_results:
            if extraction.success:
                data = extraction.extracted_data
                if hasattr(data, "model_dump"):
                    data = data.model_dump()

                # Should have columns and data for tabular format
                assert "columns" in data or "data" in data

    @pytest.mark.asyncio
    async def test_openai_full_extraction(self):
        """Test full extraction with OpenAI provider"""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        extractor = ImageDataExtractor(
            analysis_provider="openai",
            analysis_model="gpt-4.1",
            extraction_provider="openai",
            extraction_model="gpt-4.1",
        )

        result = await extractor.extract_all_data(TEST_IMAGES["chart"])

        # Verify results
        assert isinstance(result, ImageDataExtractionResult)
        assert result.total_datapoints_identified > 0
        assert result.successful_extractions > 0
        assert result.total_cost_cents > 0

        # Check extraction format
        for extraction in result.extraction_results:
            if extraction.success:
                assert extraction.extracted_data is not None
                print(f"\nOpenAI extracted {extraction.datapoint_name}")

    @pytest.mark.asyncio
    async def test_gemini_full_extraction(self):
        """Test full extraction with Gemini provider"""
        # Skip if no API key
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        extractor = ImageDataExtractor(
            analysis_provider="gemini",
            analysis_model="gemini-2.5-flash",
            extraction_provider="gemini",
            extraction_model="gemini-2.5-flash",
        )

        result = await extractor.extract_all_data(TEST_IMAGES["chart"])

        # Verify results
        assert isinstance(result, ImageDataExtractionResult)
        assert result.total_datapoints_identified > 0
        assert result.successful_extractions > 0
        assert result.total_cost_cents >= 0  # Gemini might have free tier

        # Check extraction format
        for extraction in result.extraction_results:
            if extraction.success:
                assert extraction.extracted_data is not None
                print(f"\nGemini extracted {extraction.datapoint_name}")

    @pytest.mark.asyncio
    async def test_mixed_providers(self):
        """Test with different providers for analysis and extraction"""
        # Skip if missing keys
        if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Missing required API keys")

        # Use Anthropic for analysis, OpenAI for extraction
        extractor = ImageDataExtractor(
            analysis_provider="anthropic",
            analysis_model="claude-sonnet-4-20250514",
            extraction_provider="openai",
            extraction_model="gpt-4.1-mini",
        )

        result = await extractor.extract_all_data(TEST_IMAGES["chart"])

        assert result.successful_extractions > 0
        assert result.total_cost_cents > 0

        # Verify we got data
        extracted_dict = await extractor.extract_as_dict(TEST_IMAGES["chart"])
        assert len(extracted_dict["data"]) > 0

        print(
            f"\nMixed providers - Analysis cost: ${result.metadata['analysis_cost_cents'] / 100:.4f}"
        )
        print(f"Extraction cost: ${result.metadata['extraction_cost_cents'] / 100:.4f}")

    @pytest.mark.asyncio
    async def test_extract_as_dict_format(self):
        """Test the dictionary extraction format"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = ImageDataExtractor()
        result = await extractor.extract_as_dict(TEST_IMAGES["chart"])

        # Verify dictionary structure
        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data" in result

        # Check metadata fields
        metadata = result["metadata"]
        assert "image_url" in metadata
        assert "image_type" in metadata
        assert "content_description" in metadata
        assert "extraction_summary" in metadata

        # Check extraction summary
        summary = metadata["extraction_summary"]
        assert "total_identified" in summary
        assert "successful" in summary
        assert "failed" in summary
        assert "time_ms" in summary
        assert "cost_cents" in summary

        # Verify we got actual data
        assert len(result["data"]) > 0

    @pytest.mark.asyncio
    async def test_focus_areas(self):
        """Test extraction with focus areas"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = ImageDataExtractor()

        # Test with focus on specific areas
        result = await extractor.extract_all_data(
            TEST_IMAGES["chart"],
            focus_areas=["electricity generation", "country data", "bar values"],
        )

        assert result.successful_extractions > 0

    @pytest.mark.asyncio
    async def test_datapoint_filtering(self):
        """Test filtering specific datapoints"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        extractor = ImageDataExtractor()

        # First, analyze to see what datapoints are available
        analysis, _ = await extractor.analyze_image_structure(TEST_IMAGES["chart"])

        if len(analysis.identified_datapoints) > 0:
            # Extract only the first datapoint
            first_datapoint_name = analysis.identified_datapoints[0].name

            result = await extractor.extract_all_data(
                TEST_IMAGES["chart"], datapoint_filter=[first_datapoint_name]
            )

            # Should only extract the filtered datapoint
            assert len(result.extraction_results) == 1
            assert result.extraction_results[0].datapoint_name == first_datapoint_name
