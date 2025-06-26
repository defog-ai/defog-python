"""
Example of using the PDF Data Extractor to identify and extract
structured data from PDFs in parallel.

This example demonstrates:
- Using different LLM providers (Anthropic and OpenAI)
- Extracting data from various PDF types
- Cost analysis and token usage tracking
"""

import asyncio
from defog.llm.pdf_data_extractor import extract_pdf_data
import json
import logging
from urllib.parse import urlparse
import re
import time

logging.basicConfig(level=logging.INFO)


def generate_safe_filename(url: str) -> str:
    """
    Generate a safe filename from a URL for saving extracted data.

    Args:
        url: The URL to convert to a filename

    Returns:
        A safe filename string
    """
    try:
        parsed = urlparse(url)
        # Use the path component, removing leading slash and .pdf extension
        path_part = parsed.path.lstrip("/").replace(".pdf", "")
        # If path is empty, use the netloc (domain)
        if not path_part:
            path_part = parsed.netloc
        # Replace invalid filename characters with underscores
        safe_name = re.sub(r"[^\w\-_.]", "_", path_part)
        # Remove multiple consecutive underscores
        safe_name = re.sub(r"_+", "_", safe_name)
        # Remove leading/trailing underscores
        safe_name = safe_name.strip("_")
        # Ensure we have a non-empty filename
        if not safe_name:
            safe_name = "extracted_data"

        timestamp = int(time.time())
        return f"extracted_data_{safe_name}_{timestamp}.json"
    except Exception:
        # Fallback to a simple timestamp-based name if URL parsing fails
        timestamp = int(time.time())
        return f"extracted_data_{timestamp}.json"


async def extract_pdf_example(url, provider="anthropic", model=None):
    """Extract data from a PDF using the extract_pdf_data function."""
    print(f"\n=== Data Extraction from {url} using {provider} ===\n")

    # Use the convenience function
    result = await extract_pdf_data(
        pdf_url=url,
        provider=provider,
        model=model,
        focus_areas=["financial data", "tables", "key metrics"],
        max_parallel_extractions=10,
    )

    # Extract metadata for display
    metadata = result.get("metadata", {})

    print(f"Provider: {provider}")
    print(f"Model: {model or 'default for provider'}")
    print(f"Document Type: {metadata.get('document_type', 'Unknown')}")
    print(
        f"Total datapoints identified: {metadata.get('total_datapoints_identified', 0)}"
    )
    print(f"Successful extractions: {metadata.get('successful_extractions', 0)}")
    print(f"Failed extractions: {metadata.get('failed_extractions', 0)}")
    print(f"Total time: {metadata.get('total_time_ms', 0) / 1000:.2f} seconds")

    print("\n--- Cost Analysis ---")
    print(f"Total cost: ${metadata.get('total_cost_cents', 0) / 100:.4f}")
    print(
        f"Analysis cost (Step 1): ${metadata.get('analysis_cost_cents', 0.0) / 100:.4f}"
    )
    print(
        f"Extraction cost (Step 2+): ${metadata.get('extraction_cost_cents', 0.0) / 100:.4f}"
    )

    print("\n--- Token Usage ---")
    print(f"Total input tokens: {metadata.get('total_input_tokens', 0):,}")
    print(f"Total output tokens: {metadata.get('total_output_tokens', 0):,}")
    print(f"Total cached tokens: {metadata.get('total_cached_tokens', 0):,}")
    print(
        f"Total tokens: {metadata.get('total_input_tokens', 0) + metadata.get('total_output_tokens', 0):,}"
    )

    print("\n--- Extracted Datapoints ---")
    data = result.get("data", {})
    for datapoint_name, datapoint_data in data.items():
        print(f"\n✅ {datapoint_name}")
        # Show a preview of the data structure
        if isinstance(datapoint_data, dict):
            print(
                f"   Keys: {list(datapoint_data.keys())[:5]}{'...' if len(datapoint_data.keys()) > 5 else ''}"
            )
        elif isinstance(datapoint_data, list) and len(datapoint_data) > 0:
            print(f"   {len(datapoint_data)} items extracted")
        else:
            print(f"   Type: {type(datapoint_data).__name__}")

    # Save the extracted data to a JSON file
    filename = generate_safe_filename(url)
    with open(filename, "w") as f:
        # The result is already JSON-serializable
        json.dump(result, f, indent=2)
    print(f"\n💾 Data saved to: {filename}")

    return result


async def main():
    """Run all examples."""
    print("🚀 PDF Data Extractor Examples")
    print("=" * 60)

    # Example PDF URLs
    apple_financial_pdf = "https://www.apple.com/newsroom/pdfs/fy2025-q2/FY25_Q2_Consolidated_Financial_Statements.pdf"

    try:
        # Example 1: Extract with Anthropic (default)
        print("\n📄 Example 1: Extracting Apple Financial Report with Anthropic")
        await extract_pdf_example(
            apple_financial_pdf, provider="anthropic", model="claude-sonnet-4-20250514"
        )

        # Example 2: Extract with OpenAI
        print("\n\n📄 Example 2: Extracting Apple Financial Report with OpenAI")
        await extract_pdf_example(
            apple_financial_pdf, provider="openai", model="o4-mini"
        )

        # Uncomment to try other PDFs:
        # Qwen 2.5 research paper (26 pages)
        # await extract_pdf_example("https://arxiv.org/pdf/2412.15115", provider="openai")

        # AI 2027 report
        # await extract_pdf_example("https://ai-2027.com/ai-2027.pdf", provider="anthropic")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
