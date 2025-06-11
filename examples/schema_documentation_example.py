"""
Example demonstrating LLM-powered database schema self-documentation.

This example shows how to use the new self-documentation feature to automatically
generate table and column descriptions using LLMs and store them as database comments.
"""

import asyncio
import tempfile
import os
from defog import Defog
from defog.schema_documenter import (
    SchemaDocumenter,
    DocumentationConfig,
    document_and_apply_schema,
)


def create_sample_database():
    """Create a sample DuckDB database for demonstration."""
    import duckdb
    
    # Create temporary database file
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.duckdb')
    db_file.close()
    
    conn = duckdb.connect(db_file.name)
    
    # Create sample tables with realistic data
    conn.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            email VARCHAR,
            first_name VARCHAR,
            last_name VARCHAR,
            registration_date DATE,
            status VARCHAR,
            total_orders INTEGER,
            lifetime_value DECIMAL(10,2)
        )
    """)
    
    conn.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            status VARCHAR,
            total_amount DECIMAL(10,2),
            shipping_address VARCHAR,
            payment_method VARCHAR
        )
    """)
    
    conn.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name VARCHAR,
            category VARCHAR,
            price DECIMAL(10,2),
            in_stock BOOLEAN,
            supplier_id INTEGER,
            created_at TIMESTAMP
        )
    """)
    
    # Insert sample data
    conn.execute("""
        INSERT INTO customers VALUES
        (1, 'alice@example.com', 'Alice', 'Johnson', '2023-01-15', 'active', 5, 1250.75),
        (2, 'bob@gmail.com', 'Bob', 'Smith', '2023-02-20', 'active', 3, 890.50),
        (3, 'charlie@yahoo.com', 'Charlie', 'Brown', '2023-03-10', 'inactive', 1, 45.99),
        (4, 'diana@hotmail.com', 'Diana', 'Williams', '2023-04-05', 'active', 8, 2100.00),
        (5, 'eve@example.org', 'Eve', 'Davis', '2023-05-12', 'suspended', 0, 0.00)
    """)
    
    conn.execute("""
        INSERT INTO orders VALUES
        (101, 1, '2023-06-01', 'completed', 125.50, '123 Main St, Anytown', 'credit_card'),
        (102, 2, '2023-06-02', 'completed', 89.99, '456 Oak Ave, Somewhere', 'paypal'),
        (103, 1, '2023-06-03', 'pending', 250.00, '123 Main St, Anytown', 'credit_card'),
        (104, 4, '2023-06-04', 'shipped', 75.25, '789 Pine St, Elsewhere', 'debit_card'),
        (105, 3, '2023-06-05', 'cancelled', 45.99, '321 Elm Dr, Nowhere', 'cash')
    """)
    
    conn.execute("""
        INSERT INTO products VALUES
        (1001, 'Wireless Headphones', 'Electronics', 99.99, true, 501, '2023-01-01 10:00:00'),
        (1002, 'Coffee Mug', 'Home & Garden', 12.50, true, 502, '2023-01-15 14:30:00'),
        (1003, 'Running Shoes', 'Sports', 129.99, false, 503, '2023-02-01 09:15:00'),
        (1004, 'Desk Lamp', 'Office', 45.00, true, 504, '2023-02-15 16:45:00'),
        (1005, 'Smartphone Case', 'Electronics', 19.99, true, 501, '2023-03-01 11:20:00')
    """)
    
    conn.close()
    return db_file.name


async def example_basic_documentation():
    """Example 1: Basic schema documentation with default settings."""
    print("=" * 60)
    print("Example 1: Basic Schema Documentation")
    print("=" * 60)
    
    # Create sample database
    db_path = create_sample_database()
    
    try:
        # Create documenter with default config
        documenter = SchemaDocumenter(
            db_type="duckdb",
            db_creds={"database": db_path}
        )
        
        # Generate documentation for all tables
        print("Generating documentation for all tables...")
        documentation = await documenter.document_schema()
        
        # Display results
        for table_name, doc in documentation.items():
            print(f"\nTable: {table_name}")
            print(f"Description: {doc.table_description}")
            print(f"Confidence: {doc.confidence_score:.2f}")
            print("Column Descriptions:")
            for col_name, col_desc in doc.column_descriptions.items():
                print(f"  - {col_name}: {col_desc}")
        
        # Apply documentation to database
        print("\nApplying documentation to database...")
        results = await documenter.apply_documentation(documentation)
        
        success_count = sum(results.values())
        print(f"Successfully documented {success_count}/{len(results)} tables")
        
    finally:
        # Cleanup
        os.unlink(db_path)


async def example_custom_configuration():
    """Example 2: Custom documentation configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Create sample database
    db_path = create_sample_database()
    
    try:
        # Custom configuration
        config = DocumentationConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.05,  # More deterministic
            sample_size=50,    # Fewer samples for faster processing
            max_categorical_values=10,
            include_data_patterns=True,
            include_sample_values=True,
            min_confidence_score=0.8,  # Higher confidence threshold
            max_concurrent_requests=2,  # Reduce concurrency
            request_delay=0.5,  # Add delay between requests
        )
        
        # Create documenter with custom config
        documenter = SchemaDocumenter(
            db_type="duckdb",
            db_creds={"database": db_path},
            config=config
        )
        
        # Document specific tables only
        tables_to_document = ["customers", "orders"]
        print(f"Generating documentation for specific tables: {tables_to_document}")
        
        documentation = await documenter.document_schema(tables=tables_to_document)
        
        # Show detailed analysis metadata
        for table_name, doc in documentation.items():
            print(f"\nTable: {table_name}")
            print(f"Description: {doc.table_description}")
            print(f"Confidence: {doc.confidence_score:.2f}")
            print(f"Metadata: {doc.analysis_metadata}")
            
            if doc.confidence_score >= config.min_confidence_score:
                print("✓ High confidence - will be applied")
            else:
                print("✗ Low confidence - will be skipped")
        
        # Apply with dry run first
        print("\nDry run validation...")
        dry_run_results = await documenter.apply_documentation(documentation, dry_run=True)
        print(f"Dry run results: {dry_run_results}")
        
        # Apply for real
        print("\nApplying documentation...")
        apply_results = await documenter.apply_documentation(documentation, dry_run=False)
        print(f"Application results: {apply_results}")
        
    finally:
        # Cleanup
        os.unlink(db_path)


async def example_convenience_function():
    """Example 3: Using the convenience function."""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Function")
    print("=" * 60)
    
    # Create sample database
    db_path = create_sample_database()
    
    try:
        # Custom config for convenience function
        config = DocumentationConfig(
            sample_size=30,
            temperature=0.2,
            min_confidence_score=0.6,
        )
        
        # Use convenience function for one-shot documentation
        print("Running complete documentation process...")
        result = await document_and_apply_schema(
            db_type="duckdb",
            db_creds={"database": db_path},
            tables=["products"],  # Document only products table
            config=config,
            dry_run=False
        )
        
        # Display summary
        print(f"\nSummary:")
        print(f"Tables analyzed: {result['summary']['tables_analyzed']}")
        print(f"Tables applied: {result['summary']['tables_applied']}")
        print(f"Dry run: {result['summary']['dry_run']}")
        
        # Show documentation details
        for table_name, doc in result['documentation'].items():
            print(f"\nTable: {table_name}")
            print(f"Description: {doc.table_description}")
            for col_name, col_desc in doc.column_descriptions.items():
                print(f"  - {col_name}: {col_desc}")
        
    finally:
        # Cleanup
        os.unlink(db_path)


def example_integration_with_defog():
    """Example 4: Integration with Defog class."""
    print("\n" + "=" * 60)
    print("Example 4: Integration with Defog Class")
    print("=" * 60)
    
    # Create sample database
    db_path = create_sample_database()
    
    try:
        # Create Defog instance
        defog = Defog(
            api_key="your-api-key-here",  # Replace with actual API key
            db_type="duckdb",
            db_creds={"database": db_path}
        )
        
        # Use new self_document parameter
        print("Generating schema with LLM documentation...")
        
        # Configuration for documentation
        doc_config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.1,
            "sample_size": 50,
            "min_confidence_score": 0.7,
        }
        
        # This will generate schema AND apply LLM documentation
        schema = defog.generate_db_schema(
            tables=[],  # All tables
            upload=False,  # Don't upload to Defog servers
            self_document=True,  # Enable LLM documentation
            documentation_config=doc_config,
        )
        
        print("Schema generated with LLM documentation applied!")
        print("Database comments have been updated with AI-generated descriptions.")
        
        # Now generate schema again to see the applied comments
        print("\nReading back the schema with applied comments...")
        schema_with_comments = defog.generate_db_schema(
            tables=[],
            upload=False,
            self_document=False,  # Don't re-document
        )
        
        print("Schema with comments:")
        for table_name, columns in schema_with_comments.items():
            print(f"\nTable: {table_name}")
            for column in columns:
                desc = column.get('column_description', 'No description')
                print(f"  - {column['column_name']} ({column['data_type']}): {desc}")
        
    except Exception as e:
        print(f"Note: This example requires a valid API key. Error: {e}")
        print("Use examples 1-3 for standalone documentation without API key.")
    
    finally:
        # Cleanup
        os.unlink(db_path)


async def main():
    """Run all examples."""
    print("LLM-Powered Database Schema Self-Documentation Examples")
    print("=" * 60)
    print("This example demonstrates the new self-documentation feature that uses")
    print("LLMs to automatically generate meaningful table and column descriptions")
    print("by analyzing your database structure and data patterns.")
    print()
    
    # Run async examples
    await example_basic_documentation()
    await example_custom_configuration()
    await example_convenience_function()
    
    # Run sync example
    example_integration_with_defog()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("The self-documentation feature helps improve SQL query generation")
    print("by providing rich context about your database schema through")
    print("automatically generated comments stored directly in the database.")


if __name__ == "__main__":
    asyncio.run(main())