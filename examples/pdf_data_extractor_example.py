"""
Example of using the PDF Data Extractor to identify and extract
structured data from PDFs in parallel.
"""

import asyncio
import json
from defog.llm.pdf_data_extractor import PDFDataExtractor, extract_pdf_data


async def example_financial_report():
    """Extract data from a financial report PDF."""
    print("=== Financial Report Data Extraction ===\n")
    
    # Example with a financial report
    pdf_url = "https://www.apple.com/newsroom/pdfs/fy2025-q2/FY25_Q2_Consolidated_Financial_Statements.pdf"
    
    # Create extractor
    extractor = PDFDataExtractor(
        analysis_model="claude-sonnet-4-20250514",
        extraction_model="claude-sonnet-4-20250514",
        max_parallel_extractions=5
    )
    
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
    
    # Use convenience function for simpler usage
    print("Extracting data from research paper...")
    data = await extract_pdf_data(
        pdf_url=pdf_url
    )
    
    print(f"\nDocument Type: {data['metadata']['document_type']}")
    print(f"Extraction Summary: {json.dumps(data['metadata']['extraction_summary'], indent=2)}")
    
    print("\n--- Extracted Data ---")
    for datapoint_name, datapoint_data in data['data'].items():
        print(f"\n📊 {datapoint_name}:")
        print(datapoint_data)


async def example_with_filtering():
    """Extract only specific datapoints from a PDF."""
    print("\n\n=== Selective Data Extraction ===\n")
    
    pdf_url = "https://www.apple.com/newsroom/pdfs/fy2024-q1-unaudited-financial-statements.pdf"
    
    extractor = PDFDataExtractor()
    
    # First, analyze to see what's available
    print("Step 1: Analyzing PDF to identify available datapoints...")
    analysis = await extractor.analyze_pdf_structure(pdf_url)
    
    print(f"\nFound {len(analysis.identified_datapoints)} datapoints:")
    for dp in analysis.identified_datapoints:
        print(f"  - {dp.name}: {dp.data_type} ({dp.description[:50]}...)")
    
    # Extract only specific datapoints
    print("\nStep 2: Extracting only revenue-related datapoints...")
    
    # Filter to only revenue-related datapoints
    revenue_datapoints = [
        dp.name for dp in analysis.identified_datapoints 
        if 'revenue' in dp.name.lower() or 'income' in dp.name.lower()
    ]
    
    if revenue_datapoints:
        result = await extractor.extract_all_data(
            pdf_url=pdf_url,
            datapoint_filter=revenue_datapoints
        )
        
        print(f"\nExtracted {result.successful_extractions} revenue-related datapoints")
        
        # Get as dictionary for easy access
        data_dict = await extractor.extract_as_dict(
            pdf_url=pdf_url,
            datapoint_filter=revenue_datapoints
        )
        
        for name, data in data_dict['data'].items():
            print(f"\n💰 {name}:")
            print(json.dumps(data, indent=2)[:300] + "...")


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