"""
Example script demonstrating HTML data extraction capabilities
"""

import asyncio
import logging
from defog.llm import HTMLDataExtractor, extract_html_data

# Set up logging to see extraction progress
logging.basicConfig(level=logging.INFO)

# Example HTML content - e-commerce product page
ECOMMERCE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Electronics Store - Best Deals</title>
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "Store",
        "name": "TechMart Electronics",
        "priceRange": "$$",
        "aggregateRating": {
            "@type": "AggregateRating",
            "ratingValue": "4.5",
            "reviewCount": "1289"
        }
    }
    </script>
</head>
<body>
    <nav class="breadcrumb">
        <a href="/">Home</a> > 
        <a href="/electronics">Electronics</a> > 
        <a href="/electronics/laptops">Laptops</a>
    </nav>

    <h1>Premium Laptops - Holiday Sale</h1>
    
    <div class="filters">
        <h3>Filter by Brand</h3>
        <ul>
            <li data-brand="apple" data-count="15">Apple (15)</li>
            <li data-brand="dell" data-count="23">Dell (23)</li>
            <li data-brand="hp" data-count="18">HP (18)</li>
            <li data-brand="lenovo" data-count="27">Lenovo (27)</li>
        </ul>
    </div>

    <div class="product-grid">
        <div class="product-card" data-sku="LAP-001">
            <img src="/images/macbook-pro.jpg" alt="MacBook Pro">
            <h2>MacBook Pro 16"</h2>
            <div class="specs">
                <span class="cpu">M3 Pro</span>
                <span class="ram">18GB RAM</span>
                <span class="storage">512GB SSD</span>
            </div>
            <p class="price">
                <span class="original-price">$2,499</span>
                <span class="sale-price">$2,199</span>
                <span class="discount">-12%</span>
            </p>
            <div class="rating" data-rating="4.8">★★★★★ (4.8/5)</div>
            <p class="availability in-stock">In Stock - Ships Today</p>
        </div>

        <div class="product-card" data-sku="LAP-002">
            <img src="/images/dell-xps.jpg" alt="Dell XPS 15">
            <h2>Dell XPS 15</h2>
            <div class="specs">
                <span class="cpu">Intel i7-13700H</span>
                <span class="ram">16GB RAM</span>
                <span class="storage">1TB SSD</span>
            </div>
            <p class="price">
                <span class="original-price">$1,899</span>
                <span class="sale-price">$1,599</span>
                <span class="discount">-16%</span>
            </p>
            <div class="rating" data-rating="4.6">★★★★★ (4.6/5)</div>
            <p class="availability in-stock">In Stock - Ships Tomorrow</p>
        </div>

        <div class="product-card" data-sku="LAP-003">
            <img src="/images/hp-spectre.jpg" alt="HP Spectre x360">
            <h2>HP Spectre x360</h2>
            <div class="specs">
                <span class="cpu">Intel i7-1355U</span>
                <span class="ram">16GB RAM</span>
                <span class="storage">512GB SSD</span>
            </div>
            <p class="price">
                <span class="original-price">$1,449</span>
                <span class="sale-price">$1,249</span>
                <span class="discount">-14%</span>
            </p>
            <div class="rating" data-rating="4.5">★★★★☆ (4.5/5)</div>
            <p class="availability low-stock">Low Stock - Only 3 Left</p>
        </div>
    </div>

    <section class="comparison-table">
        <h2>Quick Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Processor</th>
                    <th>Display</th>
                    <th>Battery Life</th>
                    <th>Weight</th>
                    <th>Price</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>MacBook Pro 16"</td>
                    <td>M3 Pro</td>
                    <td>16.2" Retina</td>
                    <td>22 hours</td>
                    <td>4.8 lbs</td>
                    <td>$2,199</td>
                </tr>
                <tr>
                    <td>Dell XPS 15</td>
                    <td>Intel i7-13700H</td>
                    <td>15.6" OLED</td>
                    <td>13 hours</td>
                    <td>4.2 lbs</td>
                    <td>$1,599</td>
                </tr>
                <tr>
                    <td>HP Spectre x360</td>
                    <td>Intel i7-1355U</td>
                    <td>13.5" OLED Touch</td>
                    <td>15 hours</td>
                    <td>3.0 lbs</td>
                    <td>$1,249</td>
                </tr>
            </tbody>
        </table>
    </section>

    <footer>
        <div class="store-info">
            <h3>Store Information</h3>
            <dl>
                <dt>Phone</dt>
                <dd>1-800-TECH-MART</dd>
                <dt>Email</dt>
                <dd>support@techmart.com</dd>
                <dt>Hours</dt>
                <dd>Mon-Fri: 9AM-8PM, Sat-Sun: 10AM-6PM</dd>
            </dl>
        </div>
    </footer>
</body>
</html>
"""

# Example HTML content - Financial Report
FINANCIAL_REPORT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Q3 2024 Financial Report - TechCorp Inc.</title>
</head>
<body>
    <header>
        <h1>TechCorp Inc. - Quarterly Financial Report</h1>
        <p>Q3 2024 (July 1 - September 30, 2024)</p>
    </header>

    <section class="highlights">
        <h2>Financial Highlights</h2>
        <div class="metrics-grid">
            <div class="metric">
                <span class="label">Total Revenue</span>
                <span class="value" data-value="458700000">$458.7M</span>
                <span class="change positive">+18.3% YoY</span>
            </div>
            <div class="metric">
                <span class="label">Gross Profit</span>
                <span class="value" data-value="206415000">$206.4M</span>
                <span class="change positive">+22.1% YoY</span>
            </div>
            <div class="metric">
                <span class="label">Operating Income</span>
                <span class="value" data-value="91740000">$91.7M</span>
                <span class="change positive">+15.8% YoY</span>
            </div>
            <div class="metric">
                <span class="label">Net Income</span>
                <span class="value" data-value="73392000">$73.4M</span>
                <span class="change positive">+12.5% YoY</span>
            </div>
        </div>
    </section>

    <section class="revenue-breakdown">
        <h2>Revenue by Segment</h2>
        <table class="financial-table">
            <thead>
                <tr>
                    <th>Segment</th>
                    <th>Q3 2024</th>
                    <th>Q3 2023</th>
                    <th>Change</th>
                    <th>% of Total</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Cloud Services</td>
                    <td>$201.8M</td>
                    <td>$158.2M</td>
                    <td>+27.5%</td>
                    <td>44.0%</td>
                </tr>
                <tr>
                    <td>Software Licenses</td>
                    <td>$137.6M</td>
                    <td>$125.4M</td>
                    <td>+9.7%</td>
                    <td>30.0%</td>
                </tr>
                <tr>
                    <td>Professional Services</td>
                    <td>$82.6M</td>
                    <td>$71.3M</td>
                    <td>+15.9%</td>
                    <td>18.0%</td>
                </tr>
                <tr>
                    <td>Hardware</td>
                    <td>$36.7M</td>
                    <td>$32.8M</td>
                    <td>+11.9%</td>
                    <td>8.0%</td>
                </tr>
            </tbody>
        </table>
    </section>

    <section class="geographic">
        <h2>Geographic Distribution</h2>
        <ul class="region-list">
            <li data-region="north-america" data-revenue="229350000">
                <span class="region-name">North America</span>
                <span class="revenue">$229.4M</span>
                <span class="percentage">50%</span>
            </li>
            <li data-region="europe" data-revenue="137610000">
                <span class="region-name">Europe</span>
                <span class="revenue">$137.6M</span>
                <span class="percentage">30%</span>
            </li>
            <li data-region="asia-pacific" data-revenue="68805000">
                <span class="region-name">Asia Pacific</span>
                <span class="revenue">$68.8M</span>
                <span class="percentage">15%</span>
            </li>
            <li data-region="other" data-revenue="22935000">
                <span class="region-name">Other</span>
                <span class="revenue">$22.9M</span>
                <span class="percentage">5%</span>
            </li>
        </ul>
    </section>
</body>
</html>
"""


async def example_basic_extraction():
    """Basic example of extracting all data from HTML"""
    print("\n=== Basic HTML Data Extraction ===")
    
    # Create extractor
    extractor = HTMLDataExtractor()
    
    # Extract all data
    result = await extractor.extract_as_dict(ECOMMERCE_HTML)
    
    print(f"\nPage Type: {result['metadata']['page_type']}")
    print(f"Identified {result['metadata']['extraction_summary']['total_identified']} datapoints")
    print(f"Successfully extracted: {result['metadata']['extraction_summary']['successful']}")
    print(f"Total cost: ${result['metadata']['extraction_summary']['cost_cents'] / 100:.4f}")
    
    print("\nExtracted Data:")
    for name, data in result['data'].items():
        print(f"\n{name}:")
        if isinstance(data, dict):
            if 'columns' in data and 'data' in data:
                print(f"  Columns: {data['columns']}")
                print(f"  Rows: {len(data['data'])}")
                if data['data']:
                    print(f"  Sample row: {data['data'][0]}")
            else:
                for key, value in list(data.items())[:3]:
                    print(f"  {key}: {value}")
        else:
            print(f"  {data}")


async def example_focused_extraction():
    """Example of extracting specific types of data"""
    print("\n=== Focused Extraction ===")
    
    extractor = HTMLDataExtractor()
    
    # Focus on specific areas
    result = await extractor.extract_all_data(
        FINANCIAL_REPORT_HTML,
        focus_areas=["revenue breakdown", "financial metrics", "geographic data"]
    )
    
    print(f"\nExtraction Results:")
    print(f"Total datapoints identified: {result.total_datapoints_identified}")
    print(f"Successful extractions: {result.successful_extractions}")
    print(f"Failed extractions: {result.failed_extractions}")
    
    # Get dictionary format for easier access
    dict_result = await extractor.extract_as_dict(FINANCIAL_REPORT_HTML)
    
    print("\nExtracted Financial Data:")
    for name, data in dict_result['data'].items():
        if 'revenue' in name.lower() or 'financial' in name.lower():
            print(f"\n{name}:")
            if isinstance(data, dict) and 'data' in data:
                print(f"  Found {len(data['data'])} rows of financial data")


async def example_filtered_extraction():
    """Example of extracting only specific datapoints"""
    print("\n=== Filtered Extraction ===")
    
    extractor = HTMLDataExtractor()
    
    # First, analyze to see what datapoints are available
    initial_result = await extractor.extract_all_data(ECOMMERCE_HTML)
    
    print("\nAvailable datapoints:")
    for extraction in initial_result.extraction_results:
        if extraction.success:
            print(f"  - {extraction.datapoint_name}")
    
    # Extract only specific datapoints
    if initial_result.extraction_results:
        # Get names of product-related datapoints
        product_datapoints = [
            e.datapoint_name for e in initial_result.extraction_results 
            if 'product' in e.datapoint_name.lower() and e.success
        ]
        
        if product_datapoints:
            print(f"\nExtracting only: {product_datapoints}")
            
            filtered_result = await extractor.extract_all_data(
                ECOMMERCE_HTML,
                datapoint_filter=product_datapoints[:1]  # Just the first one
            )
            
            print(f"Filtered extraction completed:")
            print(f"  Extracted {filtered_result.successful_extractions} datapoints")


async def example_convenience_function():
    """Example using the convenience function"""
    print("\n=== Convenience Function Example ===")
    
    # Simple one-liner extraction
    result = await extract_html_data(
        FINANCIAL_REPORT_HTML,
        focus_areas=["revenue", "financial metrics"],
        temperature=0.1
    )
    
    print(f"\nExtracted {len(result['data'])} datapoints:")
    for name in result['data'].keys():
        print(f"  - {name}")


async def main():
    """Run all examples"""
    print("HTML Data Extractor Examples")
    print("=" * 50)
    
    await example_basic_extraction()
    await example_focused_extraction()
    await example_filtered_extraction()
    await example_convenience_function()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())