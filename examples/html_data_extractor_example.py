"""
Example of using the HTML Data Extractor to identify and extract
structured data from HTML strings in parallel.
"""

import asyncio
from defog.llm.html_data_extractor import HTMLDataExtractor
import json
import logging

logging.basicConfig(level=logging.INFO)


async def extract_from_html(html_string, description="HTML content"):
    """Extract data from HTML string."""
    print(f"=== Data Extraction from {description} ===\n")

    extractor = HTMLDataExtractor(
        analysis_model="claude-sonnet-4-20250514",
        extraction_model="claude-sonnet-4-20250514",
        max_parallel_extractions=10,
    )

    print("Analyzing HTML structure and extracting data...")
    result = await extractor.extract_all_data(html_string=html_string)

    print(f"\nHTML Type: {result.html_type}")
    print(f"Total datapoints identified: {result.total_datapoints_identified}")
    print(f"Successful extractions: {result.successful_extractions}")
    print(f"Failed extractions: {result.failed_extractions}")
    print(f"Total time: {result.total_time_ms / 1000:.2f} seconds")

    print(f"\n--- Cost Analysis ---")
    print(f"Total cost: ${result.total_cost_cents / 100:.4f}")
    print(
        f"Analysis cost (Step 1): ${result.metadata.get('analysis_cost_cents', 0.0) / 100:.4f}"
    )
    print(
        f"Extraction cost (Step 2+): ${result.metadata.get('extraction_cost_cents', 0.0) / 100:.4f}"
    )

    print(f"\n--- Token Usage ---")
    print(f"Total input tokens: {result.metadata.get('total_input_tokens', 0):,}")
    print(f"Total output tokens: {result.metadata.get('total_output_tokens', 0):,}")
    print(f"Total cached tokens: {result.metadata.get('total_cached_tokens', 0):,}")
    print(
        f"Total tokens: {result.metadata.get('total_input_tokens', 0) + result.metadata.get('total_output_tokens', 0):,}"
    )

    print("\n--- Extracted Datapoints ---")
    for extraction in result.extraction_results:
        if extraction.success:
            print(f"\n✅ {extraction.datapoint_name}:")
            print(
                f"   Cost: ${extraction.cost_cents / 100:.4f} | Tokens: {extraction.input_tokens + extraction.output_tokens:,} (in:{extraction.input_tokens:,}, out:{extraction.output_tokens:,}, cached:{extraction.cached_tokens:,})"
            )
            if hasattr(extraction.extracted_data, 'model_dump'):
                data = extraction.extracted_data.model_dump()
                print(f"   Data preview: {str(data)[:200]}...")
        else:
            print(f"\n❌ {extraction.datapoint_name}: {extraction.error}")
            if extraction.cost_cents > 0:
                print(
                    f"   Cost: ${extraction.cost_cents / 100:.4f} | Tokens: {extraction.input_tokens + extraction.output_tokens:,}"
                )

    filename = f"extracted_html_data_{description.replace(' ', '_').lower()}.json"
    with open(filename, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"\nData saved to {filename}")


async def main():
    """Run all examples."""
    print("🚀 HTML Data Extractor Examples")
    print("=" * 60)

    try:
        sample_table_html = """
        <html>
        <head><title>Sales Report</title></head>
        <body>
            <h1>Q4 2024 Sales Report</h1>
            <table border="1">
                <tr>
                    <th>Product</th>
                    <th>Q1 Sales</th>
                    <th>Q2 Sales</th>
                    <th>Q3 Sales</th>
                    <th>Q4 Sales</th>
                    <th>Total</th>
                </tr>
                <tr>
                    <td>Widget A</td>
                    <td>$15,000</td>
                    <td>$18,000</td>
                    <td>$22,000</td>
                    <td>$25,000</td>
                    <td>$80,000</td>
                </tr>
                <tr>
                    <td>Widget B</td>
                    <td>$12,000</td>
                    <td>$14,000</td>
                    <td>$16,000</td>
                    <td>$18,000</td>
                    <td>$60,000</td>
                </tr>
                <tr>
                    <td>Widget C</td>
                    <td>$8,000</td>
                    <td>$9,000</td>
                    <td>$11,000</td>
                    <td>$13,000</td>
                    <td>$41,000</td>
                </tr>
            </table>
            
            <h2>Top Performers</h2>
            <ul>
                <li>John Smith - $45,000 in sales</li>
                <li>Sarah Johnson - $38,000 in sales</li>
                <li>Mike Davis - $32,000 in sales</li>
                <li>Lisa Wilson - $28,000 in sales</li>
            </ul>
            
            <h2>Key Metrics</h2>
            <div class="metric">
                <span class="label">Total Revenue:</span>
                <span class="value">$181,000</span>
            </div>
            <div class="metric">
                <span class="label">Growth Rate:</span>
                <span class="value">15.2%</span>
            </div>
            <div class="metric">
                <span class="label">Customer Satisfaction:</span>
                <span class="value">4.8/5</span>
            </div>
        </body>
        </html>
        """

        await extract_from_html(sample_table_html, "Sales Report Table")

        sample_product_html = """
        <div class="product-grid">
            <div class="product-card">
                <h3>Laptop Pro 15</h3>
                <p class="price">$1,299.99</p>
                <p class="rating">4.5 stars</p>
                <p class="stock">In Stock (23 units)</p>
                <p class="category">Electronics</p>
            </div>
            <div class="product-card">
                <h3>Wireless Headphones</h3>
                <p class="price">$199.99</p>
                <p class="rating">4.2 stars</p>
                <p class="stock">Low Stock (5 units)</p>
                <p class="category">Audio</p>
            </div>
            <div class="product-card">
                <h3>Smart Watch</h3>
                <p class="price">$299.99</p>
                <p class="rating">4.7 stars</p>
                <p class="stock">In Stock (15 units)</p>
                <p class="category">Wearables</p>
            </div>
            <div class="product-card">
                <h3>Gaming Mouse</h3>
                <p class="price">$79.99</p>
                <p class="rating">4.3 stars</p>
                <p class="stock">Out of Stock</p>
                <p class="category">Gaming</p>
            </div>
        </div>
        """

        await extract_from_html(sample_product_html, "Product Catalog")

        sample_form_html = """
        <form id="user-registration">
            <h2>User Registration</h2>
            <div class="form-group">
                <label for="firstName">First Name:</label>
                <input type="text" id="firstName" name="firstName" value="John" required>
            </div>
            <div class="form-group">
                <label for="lastName">Last Name:</label>
                <input type="text" id="lastName" name="lastName" value="Doe" required>
            </div>
            <div class="form-group">
                <label for="email">Email:</label>
                <input type="email" id="email" name="email" value="john.doe@example.com" required>
            </div>
            <div class="form-group">
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" value="30" min="18" max="100">
            </div>
            <div class="form-group">
                <label for="country">Country:</label>
                <select id="country" name="country">
                    <option value="us" selected>United States</option>
                    <option value="ca">Canada</option>
                    <option value="uk">United Kingdom</option>
                </select>
            </div>
            <div class="form-group">
                <label for="newsletter">Subscribe to newsletter:</label>
                <input type="checkbox" id="newsletter" name="newsletter" checked>
            </div>
        </form>
        """

        await extract_from_html(sample_form_html, "Registration Form")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
