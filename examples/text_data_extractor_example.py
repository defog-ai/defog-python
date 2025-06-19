"""
Example of using TextDataExtractor to extract structured data from text documents
"""

import asyncio
import json
import logging
from defog.llm.text_data_extractor import TextDataExtractor

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    # Read the example text file
    logger.info("Reading example.txt")
    with open("example.txt", "r") as f:
        text_content = f.read()

    logger.info(f"Loaded text content: {len(text_content)} characters")

    # Initialize the extractor
    logger.info("Initializing TextDataExtractor")
    extractor = TextDataExtractor(
        analysis_provider="anthropic",
        analysis_model="claude-sonnet-4-20250514",
        extraction_provider="anthropic",
        extraction_model="claude-sonnet-4-20250514",
        max_parallel_extractions=5,
        temperature=0.0,
    )

    print("🔍 Analyzing text document...")
    print("-" * 60)

    logger.info("Starting extraction process")

    # Extract all data with focus on Q&A and economic indicators
    result = await extractor.extract_as_dict(
        text_content,
        focus_areas=["economic indicators", "policy decisions"],
    )

    # Print metadata
    print(f"📄 Document Type: {result['metadata']['document_type']}")
    print(f"📝 Description: {result['metadata']['content_description']}")
    print("\n📊 Extraction Summary:")
    summary = result["metadata"]["extraction_summary"]
    print(f"  • Total datapoints identified: {summary['total_identified']}")
    print(f"  • Successful extractions: {summary['successful']}")
    print(f"  • Failed extractions: {summary['failed']}")
    print(f"  • Time taken: {summary['time_ms']}ms")
    print(f"  • Total cost: ${summary['cost_cents'] / 100:.4f}")

    # Print extracted data
    print("\n🎯 Extracted Data:")
    print("-" * 60)

    for datapoint_name, data in result["data"].items():
        print(f"\n📌 {datapoint_name}:")

        # Pretty print the data based on its structure
        if isinstance(data, dict):
            if "exchanges" in data:  # Q&A pairs
                print(f"  Found {len(data.get('exchanges', []))} Q&A exchanges")
                for i, exchange in enumerate(
                    data.get("exchanges", [])[:3]
                ):  # Show first 3
                    print(f"\n  Exchange {i + 1}:")
                    print(f"    Questioner: {exchange.get('questioner', 'N/A')}")
                    print(f"    Question: {exchange.get('question', 'N/A')[:100]}...")
                    print(f"    Answer: {exchange.get('answer', 'N/A')[:100]}...")
                if len(data.get("exchanges", [])) > 3:
                    print(
                        f"  ... and {len(data.get('exchanges', [])) - 3} more exchanges"
                    )

            elif "columns" in data and "data" in data:  # Table format
                print(f"  Columns: {', '.join(data['columns'])}")
                print(f"  Rows: {len(data['data'])}")
                # Show first few rows
                for i, row in enumerate(data["data"][:3]):
                    print(f"  Row {i + 1}: {row}")
                if len(data["data"]) > 3:
                    print(f"  ... and {len(data['data']) - 3} more rows")

            elif "items" in data:  # List format
                print(f"  Found {data.get('item_count', len(data['items']))} items")
                for i, item in enumerate(data["items"][:5]):  # Show first 5
                    print(f"  • {item}")
                if len(data["items"]) > 5:
                    print(f"  ... and {len(data['items']) - 5} more items")

            else:  # Key-value pairs
                for key, value in list(data.items())[:5]:  # Show first 5
                    print(f"  • {key}: {value}")
                if len(data) > 5:
                    print(f"  ... and {len(data) - 5} more fields")

    # Save full results to JSON
    output_file = "text_extraction_results.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
