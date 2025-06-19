"""
Example usage of the LLM-powered schema documentation feature.

This example demonstrates how to use the schema documentation functionality
to automatically generate table and column descriptions using LLMs.
"""

import asyncio
from defog.schema_documenter import SchemaDocumenter, DocumentationConfig
from defog import Defog


def main():
    """Main example function."""
    print("=" * 60)
    print("LLM-Powered Database Schema Documentation Example")
    print("=" * 60)

    # Example 1: Using with existing Defog workflow
    print("\n1. Integration with existing Defog workflow:")
    print("-" * 50)

    # Initialize Defog (example with Postgres)
    postgres_creds = {
        "host": "localhost",
        "port": 5432,
        "database": "your_database",
        "user": "your_user",
        "password": "your_password",
    }

    try:
        defog = Defog(
            db_type="postgres",
            db_creds=postgres_creds,
            # api_key parameter is deprecated and no longer needed
        )

        # Generate schema with LLM documentation
        print("Generating schema with LLM-powered documentation...")
        schema = defog.generate_db_schema(
            tables=[],  # Empty list = all tables
            # upload parameter is deprecated - schemas are now saved locally
            self_document=True,  # Enable LLM documentation
            documentation_config={
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "min_confidence_score": 0.8,
                "sample_size": 50,
                "dry_run": False,  # Set to True for testing without actual changes
            },
        )

        print("Schema generation completed!")
        print(f"Found {len(schema)} tables with documentation.")

        # Display sample output
        for table_name, table_info in list(schema.items())[:2]:  # Show first 2 tables
            print(f"\nTable: {table_name}")
            if isinstance(table_info, dict) and "table_description" in table_info:
                print(f"  Description: {table_info['table_description']}")
                print(f"  Columns: {len(table_info.get('columns', []))}")

    except Exception as e:
        print(f"Note: This example requires a real database connection: {e}")

    # Example 2: Standalone usage with advanced configuration
    print("\n2. Advanced standalone usage:")
    print("-" * 50)

    # Create advanced configuration
    config = DocumentationConfig(
        provider="anthropic",  # or "openai", "gemini"
        model="claude-sonnet-4-20250514",
        sample_size=100,
        min_confidence_score=0.7,
        include_data_patterns=True,
        dry_run=True,  # Safe mode - don't actually modify database
        concurrent_requests=3,
        rate_limit_delay=1.0,
    )

    try:
        # Initialize documenter
        documenter = SchemaDocumenter("postgres", postgres_creds, config)

        # Run async documentation
        async def run_documentation():
            print("Running async schema documentation...")

            # Document specific tables
            documentation = await documenter.document_schema(
                tables=["users", "orders", "products"]
            )

            print(f"Documentation generated for {len(documentation)} tables:")

            for table_name, doc in documentation.items():
                print(f"\n  Table: {table_name}")
                if doc.table_description:
                    print(f"    Description: {doc.table_description}")
                    print(f"    Confidence: {doc.table_confidence:.2f}")

                if doc.existing_table_comment:
                    print(f"    Existing comment: {doc.existing_table_comment}")

                for col_name, desc in doc.column_descriptions.items():
                    confidence = doc.column_confidences.get(col_name, 0.0)
                    print(
                        f"    Column {col_name}: {desc} (confidence: {confidence:.2f})"
                    )

                for col_name, existing in doc.existing_column_comments.items():
                    print(f"    Existing comment for {col_name}: {existing}")

            # Apply documentation to database (in dry_run mode, this just logs)
            if not config.dry_run:
                print("\nApplying documentation to database...")
                results = await documenter.apply_documentation(documentation)
                print("Application results:", results)
            else:
                print("\nDry run mode - documentation would be applied as shown above.")

            return documentation

        # Run the async function
        documentation = asyncio.run(run_documentation())

    except Exception as e:
        print(f"Note: This example requires a real database connection: {e}")

    # Example 3: DuckDB usage
    print("\n3. DuckDB usage:")
    print("-" * 50)

    duckdb_creds = {"database": ":memory:"}  # or path to .duckdb file

    try:
        # For demonstration purposes, let's show what the call would look like
        print("DuckDB schema documentation would work like this:")
        print(
            """
        defog_duckdb = Defog(
            db_type="duckdb",
            db_creds={"database": "/path/to/your/database.duckdb"}
        )
        
        schema = defog_duckdb.generate_db_schema(
            tables=[],
            self_document=True,
            documentation_config={
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514"
            }
        )
        """
        )

    except Exception as e:
        print(f"DuckDB example: {e}")

    # Example 4: Configuration options
    print("\n4. Configuration options:")
    print("-" * 50)

    print(
        """
    Available configuration options for DocumentationConfig:
    
    provider: "anthropic", "openai", or "gemini"
        - Choose your preferred LLM provider
    
    model: Model name (e.g., "claude-sonnet-4-20250514", "gpt-4.1", "gemini-2.5-pro")
        - Specific model to use for documentation generation
    
    sample_size: int (default: 100)
        - Number of rows to sample for pattern analysis
    
    min_confidence_score: float (default: 0.7)
        - Minimum confidence required to apply generated descriptions
    
    include_data_patterns: bool (default: True)
        - Whether to analyze data patterns (emails, URLs, etc.)
    
    dry_run: bool (default: False)
        - If True, shows what would be done without making changes
    
    concurrent_requests: int (default: 3)
        - Number of concurrent LLM requests
    
    rate_limit_delay: float (default: 1.0)
        - Delay between requests in seconds
    """
    )

    # Example 5: Error handling and best practices
    print("\n5. Best practices:")
    print("-" * 50)

    print(
        """
    Best practices for using schema documentation:
    
    1. Start with dry_run=True to preview changes
    2. Use appropriate confidence thresholds (0.7-0.9)
    3. Review generated descriptions before applying
    4. Consider rate limiting for large databases
    5. Use specific table lists for focused documentation
    6. Backup your database before applying comments
    7. Test with sample data first
    
    Preservation of existing comments:
    - The system automatically detects existing table/column comments
    - Existing comments are never overwritten
    - Only tables/columns without comments get new descriptions
    - This ensures your manual documentation is preserved
    """
    )


if __name__ == "__main__":
    main()
