"""
Example of using the PDF Data Extractor to identify and extract
structured data from PDFs in parallel.
"""

import asyncio
from defog.llm.pdf_data_extractor import PDFDataExtractor


async def example_financial_report():
    """Extract data from a financial report PDF."""
    print("=== Financial Report Data Extraction ===\n")
    
    # Example with a financial report
    pdf_url = "https://www.apple.com/newsroom/pdfs/fy2025-q2/FY25_Q2_Consolidated_Financial_Statements.pdf"
    
    # Create extractor
    extractor = PDFDataExtractor()
    
    # Extract all financial data
    print("Analyzing PDF structure and identifying extractable data...")
    result = await extractor.extract_all_data(
        pdf_url=pdf_url
    )
    
    print(f"\nDocument Type: {result.document_type}")
    print(f"Total datapoints identified: {result.total_datapoints_identified}")
    print(f"Successful extractions: {result.successful_extractions}")
    print(f"Failed extractions: {result.failed_extractions}")
    print(f"Total time: {result.total_time_ms / 1000:.2f} seconds")
    
    print("\n--- Extracted Datapoints ---")
    for extraction in result.extraction_results:
        if extraction.success:
            print(f"\n✅ {extraction.datapoint_name}:")
            print(extraction.extracted_data.model_dump_json(indent=2))
        else:
            print(f"\n❌ {extraction.datapoint_name}: {extraction.error}")


async def example_research_paper():
    """Extract structured data from a research paper."""
    print("\n\n=== Research Paper Data Extraction ===\n")
    
    pdf_url = "https://arxiv.org/pdf/2412.15115"

    extractor = PDFDataExtractor()
    
    # Use convenience function for simpler usage
    print("Extracting data from research paper...")
    result = await extractor.extract_all_data(
        pdf_url=pdf_url
    )
    
    print(f"\nDocument Type: {result.document_type}")
    print(f"Total datapoints identified: {result.total_datapoints_identified}")
    print(f"Successful extractions: {result.successful_extractions}")
    print(f"Failed extractions: {result.failed_extractions}")
    print(f"Total time: {result.total_time_ms / 1000:.2f} seconds")
    
    print("\n--- Extracted Datapoints ---")
    for extraction in result.extraction_results:
        if extraction.success:
            print(f"\n✅ {extraction.datapoint_name}:")
            print(extraction.extracted_data.model_dump_json(indent=2))
        else:
            print(f"\n❌ {extraction.datapoint_name}: {extraction.error}")


async def main():
    """Run all examples."""
    print("🚀 PDF Data Extractor Examples")
    print("=" * 60)
    
    try:
        # Run examples
        # await example_financial_report()
        await example_research_paper()
        # await example_with_filtering()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())