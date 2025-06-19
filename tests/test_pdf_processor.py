"""
Tests for PDF processing functionality.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import base64

from defog.llm.pdf_processor import PDFAnalysisInput, ClaudePDFProcessor, analyze_pdf
from defog.llm.pdf_utils import PDFProcessor

from pydantic import BaseModel


class TestPDFProcessor:
    """Test PDF processing utilities."""

    @pytest.fixture
    def mock_pdf_content(self):
        """Mock PDF content as bytes."""
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n174\n%%EOF"

    @pytest.fixture
    def pdf_processor(self):
        """Create PDFProcessor instance."""
        return PDFProcessor()

    def test_encode_pdf_for_claude(self, pdf_processor, mock_pdf_content):
        """Test PDF encoding for Claude API."""
        encoded = pdf_processor.encode_pdf_for_claude(mock_pdf_content)

        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == mock_pdf_content

    @patch("fitz.open")
    def test_get_pdf_metadata(self, mock_fitz_open, pdf_processor, mock_pdf_content):
        """Test PDF metadata extraction."""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 5  # 5 pages
        mock_doc.needs_pass = False
        mock_doc.metadata = {"title": "Test PDF", "author": "Test Author"}
        mock_fitz_open.return_value = mock_doc

        metadata = pdf_processor.get_pdf_metadata(mock_pdf_content)

        assert metadata["page_count"] == 5
        assert metadata["size_bytes"] == len(mock_pdf_content)
        assert not metadata["is_encrypted"]
        assert metadata["metadata"]["title"] == "Test PDF"

        mock_doc.close.assert_called_once()

    def test_should_split_pdf(self, pdf_processor):
        """Test PDF splitting logic."""
        # Small PDF should not be split
        small_metadata = {"page_count": 50, "size_bytes": 10 * 1024 * 1024}  # 10MB
        assert not pdf_processor.should_split_pdf(small_metadata)

        # Large page count should be split
        large_pages_metadata = {"page_count": 150, "size_bytes": 10 * 1024 * 1024}
        assert pdf_processor.should_split_pdf(large_pages_metadata)

        # Large size should be split
        large_size_metadata = {"page_count": 50, "size_bytes": 40 * 1024 * 1024}  # 40MB
        assert pdf_processor.should_split_pdf(large_size_metadata)

    @patch("fitz.open")
    def test_split_pdf_by_pages(self, mock_fitz_open, pdf_processor, mock_pdf_content):
        """Test PDF splitting by pages."""
        # Mock original document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 150  # 150 pages

        # Mock chunk documents
        mock_chunk_doc = MagicMock()
        mock_chunk_doc.tobytes.return_value = b"chunk_content"

        mock_fitz_open.side_effect = [mock_doc, mock_chunk_doc, mock_chunk_doc]

        metadata = {"page_count": 150, "size_bytes": 20 * 1024 * 1024}  # 20MB

        chunks = pdf_processor.split_pdf_by_pages(mock_pdf_content, metadata)

        assert len(chunks) == 2  # Should create 2 chunks for 150 pages
        assert all(chunk == b"chunk_content" for chunk in chunks)

        mock_doc.close.assert_called_once()
        assert mock_chunk_doc.close.call_count == 2


class TestClaudePDFProcessor:
    """Test Claude PDF processor."""

    @pytest.fixture
    def claude_processor(self):
        """Create ClaudePDFProcessor instance."""
        return ClaudePDFProcessor()

    def test_create_pdf_message(self, claude_processor):
        """Test PDF message creation for Claude API."""
        pdf_chunks = ["base64_encoded_pdf"]
        task = "Analyze this PDF"

        messages = asyncio.run(
            claude_processor._create_pdf_message(pdf_chunks, task, 0, 1)
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # Check cache control
        system_content = messages[0]["content"][0]
        assert system_content["cache_control"]["type"] == "ephemeral"

        user_content = messages[1]["content"]
        pdf_content = next(item for item in user_content if item["type"] == "document")
        assert pdf_content["cache_control"]["type"] == "ephemeral"
        assert pdf_content["source"]["data"] == "base64_encoded_pdf"

    @patch("defog.llm.pdf_processor.chat_async")
    @pytest.mark.asyncio
    async def test_process_single_chunk_success(
        self, mock_chat_async, claude_processor
    ):
        """Test successful single chunk processing."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.content = "Analysis result"
        mock_response.input_tokens = 1000
        mock_response.output_tokens = 500
        mock_response.cost_in_cents = 10
        mock_response.cached_input_tokens = 800
        mock_chat_async.return_value = mock_response

        pdf_chunks = ["base64_chunk"]
        task = "Analyze PDF"

        result = await claude_processor._process_single_chunk(pdf_chunks, task, 0, 1)

        assert result["success"] is True
        assert result["content"] == "Analysis result"
        assert result["chunk_index"] == 0
        assert result["input_tokens"] == 1000
        assert result["cached_tokens"] == 800

        mock_chat_async.assert_called_once()

    @patch("defog.llm.pdf_processor.chat_async")
    @pytest.mark.asyncio
    async def test_process_single_chunk_failure(
        self, mock_chat_async, claude_processor
    ):
        """Test failed single chunk processing."""
        # Mock API failure
        mock_chat_async.side_effect = Exception("API Error")

        pdf_chunks = ["base64_chunk"]
        task = "Analyze PDF"

        result = await claude_processor._process_single_chunk(pdf_chunks, task, 0, 1)

        assert result["success"] is False
        assert "API Error" in result["error"]
        assert result["chunk_index"] == 0

    def test_combine_chunk_results(self, claude_processor):
        """Test combining results from multiple chunks."""
        chunk_results = [
            {"success": True, "content": "Analysis of chunk 1", "chunk_index": 0},
            {"success": True, "content": "Analysis of chunk 2", "chunk_index": 1},
            {"success": False, "error": "Failed to process", "chunk_index": 2},
        ]

        metadata = {"page_count": 150}

        combined = claude_processor._combine_chunk_results(chunk_results, metadata)

        assert "PDF Analysis Summary (2 chunks processed)" in combined
        assert "Analysis of chunk 1" in combined
        assert "Analysis of chunk 2" in combined
        assert "1 chunks failed to process" in combined


class TestPDFAnalysisTool:
    """Test the main PDF analysis tool function."""

    @patch("defog.llm.pdf_processor.download_and_process_pdf")
    @patch("defog.llm.pdf_processor._default_processor.analyze_pdf")
    @pytest.mark.asyncio
    async def test_analyze_pdf_success(self, mock_analyze_pdf, mock_download):
        """Test successful PDF analysis."""
        # Mock download and processing
        mock_download.return_value = (["base64_chunk"], {"page_count": 10})

        # Mock analysis result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "Analysis complete"
        mock_result.metadata = {"total_cost_in_cents": 15}
        mock_result.error = None
        mock_result.chunks_processed = 1
        mock_analyze_pdf.return_value = mock_result

        input_data = PDFAnalysisInput(
            url="https://example.com/test.pdf",
            task="Analyze this PDF",
            response_format=BaseModel,
        )

        result = await analyze_pdf(input_data)

        assert result["success"] is True
        assert result["result"] == "Analysis complete"
        assert result["chunks_processed"] == 1
        assert result["error"] is None

    @patch("defog.llm.pdf_processor.download_and_process_pdf")
    @pytest.mark.asyncio
    async def test_analyze_pdf_download_failure(self, mock_download):
        """Test PDF analysis with download failure."""
        # Mock download failure
        mock_download.side_effect = Exception("Download failed")

        input_data = PDFAnalysisInput(
            url="https://example.com/invalid.pdf", task="Analyze this PDF"
        )

        result = await analyze_pdf(input_data)

        assert result["success"] is False
        assert "Download failed" in result["error"]
        assert result["chunks_processed"] == 0


class TestPDFAnalysisInput:
    """Test the PDFAnalysisInput model."""

    def test_valid_input(self):
        """Test valid input creation."""
        input_data = PDFAnalysisInput(
            url="https://example.com/test.pdf",
            task="Summarize the document",
            response_format=BaseModel,
        )

        assert input_data.url == "https://example.com/test.pdf"
        assert input_data.task == "Summarize the document"
        assert input_data.response_format == BaseModel

    def test_input_without_response_format(self):
        """Test input creation without response format."""
        input_data = PDFAnalysisInput(
            url="https://example.com/test.pdf", task="Analyze the document"
        )

        assert input_data.url == "https://example.com/test.pdf"
        assert input_data.task == "Analyze the document"
        assert input_data.response_format is None


if __name__ == "__main__":
    pytest.main([__file__])
