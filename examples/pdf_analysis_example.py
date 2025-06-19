"""
Example of PDF analysis using Claude's PDF Support Tool.

This example demonstrates:
1. PDF analysis with input caching (5-minute cache)
2. Smart chunking for large PDFs (>100 pages or >32MB)
3. Structured response formatting
4. Integration with the orchestrator system
"""

import asyncio
import logging
from pydantic import BaseModel, Field

from defog.llm.pdf_processor import analyze_pdf, PDFAnalysisInput

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Example structured response models
class DocumentSummary(BaseModel):
    """Structured summary of a document."""

    title: str = Field(description="Document title")
    main_topics: list[str] = Field(description="List of main topics covered")
    key_findings: list[str] = Field(description="Key findings or conclusions")


async def simple_pdf_analysis():
    """Simple example of PDF analysis without orchestrator."""

    print("=== Simple PDF Analysis Example ===")

    # Example with a research paper PDF
    pdf_input = PDFAnalysisInput(
        url="https://arxiv.org/pdf/2301.07041.pdf",  # Example: Verifiable Fully Homomorphic Encryption
        task="Summarize this research paper, focusing on the main contributions, methodology, and key findings.",
        response_format=DocumentSummary,
    )

    try:
        result = await analyze_pdf(pdf_input)

        if result["success"]:
            print("✅ Analysis completed successfully!")
            print(f"📄 Chunks processed: {result['chunks_processed']}")
            print(
                f"💰 Cost: ${result['metadata'].get('total_cost_in_cents', 0) / 100:.4f}"
            )
            print(
                f"📊 Tokens: {result['metadata'].get('total_input_tokens', 0)} input, {result['metadata'].get('total_output_tokens', 0)} output"
            )
            print(f"⚡ Cached tokens: {result['metadata'].get('cached_tokens', 0)}")
            print("\n--- Analysis Result ---")
            print(result["result"])
        else:
            print(f"❌ Analysis failed: {result['error']}")

    except Exception as e:
        logger.error(f"Error in simple PDF analysis: {e}", exc_info=True)


async def test_large_pdf():
    """Test PDF processing with a larger document that might require chunking."""

    print("\n=== Large PDF Processing Test ===")

    # Example with a potentially large PDF (academic paper or report)
    pdf_input = PDFAnalysisInput(
        url="https://www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf",  # Example: Claude 4 System Card (120 pages)
        task="""Provide a comprehensive analysis of this document including:
        1. Executive summary
        2. Key technical innovations
        3. Experimental results and evaluation
        4. Limitations and future work
        5. Impact on the field""",
        response_format=DocumentSummary,
    )

    try:
        result = await analyze_pdf(pdf_input)

        if result["success"]:
            print("✅ Large PDF analysis completed!")
            print("📄 PDF metadata:")
            metadata = result["metadata"]
            print(f"   - Pages: {metadata.get('page_count', 'unknown')}")
            print(f"   - Size: {metadata.get('size_mb', 0):.2f} MB")
            print(f"   - Split: {metadata.get('split', False)}")
            print(f"   - Chunks: {metadata.get('chunk_count', 1)}")
            print(f"   - Processed chunks: {result['chunks_processed']}")
            print(f"💰 Total cost: ${metadata.get('total_cost_in_cents', 0) / 100:.4f}")
            print(
                f"📊 Total tokens: {metadata.get('total_input_tokens', 0)} input, {metadata.get('total_output_tokens', 0)} output"
            )
            print(f"⚡ Cached tokens: {metadata.get('cached_tokens', 0)}")

            print("\n--- Analysis Result ---")
            print(
                result["result"][:2000] + "..."
                if len(str(result["result"])) > 2000
                else result["result"]
            )
        else:
            print(f"❌ Large PDF analysis failed: {result['error']}")

    except Exception as e:
        logger.error(f"Error in large PDF analysis: {e}")


async def test_caching_behavior():
    """Test the caching behavior by analyzing the same PDF twice."""

    print("\n=== PDF Caching Test ===")

    pdf_input = PDFAnalysisInput(
        url="https://arxiv.org/pdf/2301.07041.pdf",
        task="Provide a brief summary of this paper's main contribution.",
        response_format=DocumentSummary,
    )

    print("🔄 First analysis (no cache)...")
    result1 = await analyze_pdf(pdf_input)

    if result1["success"]:
        print("✅ First analysis completed")
        print(f"⚡ Cached tokens: {result1['metadata'].get('cached_tokens', 0)}")
        print(
            f"💰 Cost: ${result1['metadata'].get('total_cost_in_cents', 0) / 100:.4f}"
        )

    print("\n🔄 Second analysis (should use cache)...")
    result2 = await analyze_pdf(pdf_input)

    if result2["success"]:
        print("✅ Second analysis completed")
        print(f"⚡ Cached tokens: {result2['metadata'].get('cached_tokens', 0)}")
        print(
            f"💰 Cost: ${result2['metadata'].get('total_cost_in_cents', 0) / 100:.4f}"
        )

        # Compare caching
        if result2["metadata"].get("cached_tokens", 0) > result1["metadata"].get(
            "cached_tokens", 0
        ):
            print("🎉 Cache is working! More tokens were cached in the second request.")
        else:
            print("⚠️  Cache behavior unclear - tokens might not be cached as expected.")


async def main():
    """Run all PDF analysis examples."""

    print("🚀 Starting Claude PDF Analysis Examples")
    print("=" * 60)

    # Run examples
    await simple_pdf_analysis()
    await test_large_pdf()
    await test_caching_behavior()

    print("\n" + "=" * 60)
    print("✅ All PDF analysis examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
