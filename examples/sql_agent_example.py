#!/usr/bin/env python3
"""
SQL Agent Example - SQLite Edition

This example demonstrates how to use the SQL tools with SQLite for local database operations,
including natural language to SQL query conversion and automatic table relevance
filtering for large databases.

Features demonstrated:
- sql_answer_tool: Convert natural language questions to SQL and execute them
- identify_relevant_tables_tool: Find relevant tables in large databases
- SQLite database setup with sample e-commerce data
- Error handling and fallback scenarios
"""

import asyncio
import os
import sqlite3
import tempfile
import logging
from typing import Dict, Any

from defog.llm.sql import sql_answer_tool
from defog.llm.llm_providers import LLMProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_database():
    """Create a sample SQLite database with e-commerce data."""

    # Create temporary database file
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_file.close()
    db_path = db_file.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            city TEXT,
            country TEXT,
            registration_date DATE
        )
    """)

    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            cost REAL,
            stock_quantity INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            total_amount REAL,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price REAL,
            FOREIGN KEY (order_id) REFERENCES orders (order_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)

    # Insert sample data
    # Customers
    customers_data = [
        (1, "John Smith", "john@email.com", "New York", "USA", "2023-01-15"),
        (2, "Emma Wilson", "emma@email.com", "London", "UK", "2023-02-20"),
        (3, "Michael Brown", "michael@email.com", "Toronto", "Canada", "2023-03-10"),
        (4, "Sarah Davis", "sarah@email.com", "Sydney", "Australia", "2023-03-25"),
        (5, "David Johnson", "david@email.com", "Berlin", "Germany", "2023-04-05"),
    ]
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers_data
    )

    # Products
    products_data = [
        (1, "Laptop Pro", "Electronics", 1299.99, 899.99, 50),
        (2, "Wireless Mouse", "Electronics", 29.99, 15.99, 200),
        (3, "Office Chair", "Furniture", 249.99, 149.99, 30),
        (4, "Coffee Mug", "Kitchen", 12.99, 5.99, 100),
        (5, "Smartphone", "Electronics", 699.99, 499.99, 75),
        (6, "Desk Lamp", "Furniture", 89.99, 49.99, 45),
        (7, "Water Bottle", "Kitchen", 19.99, 8.99, 150),
    ]
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", products_data)

    # Orders
    orders_data = [
        (1, 1, "2023-05-01", 1329.98, "completed"),
        (2, 2, "2023-05-02", 279.98, "completed"),
        (3, 3, "2023-05-03", 699.99, "shipped"),
        (4, 1, "2023-05-04", 32.98, "completed"),
        (5, 4, "2023-05-05", 339.98, "processing"),
        (6, 5, "2023-05-06", 109.98, "completed"),
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders_data)

    # Order items
    order_items_data = [
        (1, 1, 1, 1, 1299.99),  # Order 1: Laptop
        (2, 1, 2, 1, 29.99),  # Order 1: Mouse
        (3, 2, 3, 1, 249.99),  # Order 2: Chair
        (4, 2, 2, 1, 29.99),  # Order 2: Mouse
        (5, 3, 5, 1, 699.99),  # Order 3: Smartphone
        (6, 4, 4, 1, 12.99),  # Order 4: Mug
        (7, 4, 7, 1, 19.99),  # Order 4: Water Bottle
        (8, 5, 6, 2, 89.99),  # Order 5: 2 Desk Lamps
        (9, 5, 3, 1, 249.99),  # Order 5: Chair
        (10, 6, 2, 2, 29.99),  # Order 6: 2 Mice
        (11, 6, 7, 3, 19.99),  # Order 6: 3 Water Bottles
    ]
    cursor.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", order_items_data
    )

    conn.commit()
    conn.close()

    logger.info(f"Created sample database at: {db_path}")
    return db_path


class SQLAgent:
    """A SQL agent that can answer natural language questions about SQLite databases."""

    def __init__(
        self,
        db_path: str,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the SQL agent with SQLite database.

        Args:
            db_path: Path to SQLite database file
            provider: LLM provider to use
            model: Model name
        """
        self.db_type = "sqlite"
        self.db_creds = {"database": db_path}
        self.provider = provider
        self.model = model

    async def ask_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Ask a natural language question about the database.

        Args:
            question: Natural language question
            **kwargs: Optional parameters like glossary, hard_filters, etc.

        Returns:
            Dictionary with query results and metadata
        """
        logger.info(f"Processing question: {question}")

        result = await sql_answer_tool(
            question=question,
            db_type=self.db_type,
            db_creds=self.db_creds,
            model=self.model,
            provider=self.provider,
            **kwargs,
        )

        if result["success"]:
            logger.info(f"Query successful! Returned {len(result['results'])} rows")
        else:
            logger.error(f"Query failed: {result['error']}")

        return result


async def main():
    """Example usage of the SQL agent with SQLite."""

    print("🤖 SQL Agent Example - SQLite Edition")
    print("=" * 60)

    # Create sample SQLite database
    print("📊 Creating sample e-commerce database...")
    db_path = create_sample_database()

    try:
        # Create SQL agent
        agent = SQLAgent(
            db_path=db_path,
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
        )

        print("✅ Database created successfully!")
        print(f"📁 Database location: {db_path}")

        # Example questions for e-commerce database
        questions = [
            "What are the total sales by product category?",
            "Who are the top 3 customers by total purchase amount?",
            "What is the average order value by country?",
            "Which products have the highest profit margin?",
        ]

        # Example 1: Basic question answering
        print("\n📊 Example 1: Basic Question Answering")
        print("-" * 40)

        # Test basic database functionality first
        print("\n🔍 Testing Database Schema Extraction...")
        try:
            from defog import Defog

            defog_client = Defog(
                api_key="test", db_type="sqlite", db_creds={"database": db_path}
            )
            schema = defog_client.generate_db_schema([], upload=False, scan=False)
            print(f"✅ Found {len(schema)} tables: {list(schema.keys())}")

            # Show sample table structure
            if "customers" in schema:
                print("📋 Sample table structure (customers):")
                for col in schema["customers"][:3]:
                    print(f"   {col['column_name']}: {col['data_type']}")
        except Exception as e:
            print(f"❌ Schema extraction failed: {str(e)}")

        # Only run AI-powered examples if API keys are available
        if any(
            os.getenv(key)
            for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
        ):
            for i, question in enumerate(questions[:2], 1):
                print(f"\nQuestion {i}: {question}")
                try:
                    result = await agent.ask_question(question)

                    if result["success"]:
                        print(f"✅ Query: {result['query']}")
                        print(f"📈 Results: {len(result['results'])} rows returned")

                        # Show first few results
                        if result["results"] and len(result["results"]) > 0:
                            print("📄 Sample results:")
                            for row in result["results"][:3]:
                                print(f"   {row}")
                            if len(result["results"]) > 3:
                                print(
                                    f"   ... and {len(result['results']) - 3} more rows"
                                )

                    else:
                        print(f"❌ Error: {result['error']}")

                except Exception as e:
                    print(f"💥 Exception: {str(e)}")
        else:
            print("\n⚠️  Skipping AI-powered examples (no API keys configured)")
            print(
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY to test full functionality"
            )

        # AI-powered examples (only if API keys available)
        if any(
            os.getenv(key)
            for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
        ):
            # Example 2: Using business glossary
            print("\n📚 Example 2: Using Business Glossary")
            print("-" * 35)

            glossary = """
            Profit Margin: The difference between product price and cost, divided by price, expressed as a percentage
            Total Sales: The sum of all order item quantities multiplied by their unit prices
            Average Order Value: The total order amount divided by the number of orders
            """

            question = "What are the profit margins for all products, ordered by highest margin?"
            print(f"\nQuestion: {question}")

            try:
                result = await agent.ask_question(
                    question=question,
                    glossary=glossary,
                )

                if result["success"]:
                    print(f"✅ Query with glossary: {result['query']}")
                    print(f"📈 Results: {len(result['results'])} rows")

                    # Show first few results
                    if result["results"] and len(result["results"]) > 0:
                        print("📄 Sample results:")
                        for row in result["results"][:3]:
                            print(f"   {row}")
                else:
                    print(f"❌ Error: {result['error']}")

            except Exception as e:
                print(f"💥 Exception: {str(e)}")

            # Example 3: Conversational Context
            print("\n💬 Example 3: Conversational Context")
            print("-" * 38)

            conversation_context = []

            # Simulate a conversation with context
            questions_with_context = [
                "What are the top 3 selling products by quantity?",
                "What about by revenue instead?",
                "Show me the customers who bought these top products",
            ]

            for i, question in enumerate(questions_with_context, 1):
                print(f"\nTurn {i}: {question}")

                try:
                    result = await agent.ask_question(
                        question=question, previous_context=conversation_context
                    )

                    if result["success"]:
                        print(f"✅ Query: {result['query']}")
                        print(f"📈 Results: {len(result['results'])} rows")

                        # Show first few results
                        if result["results"] and len(result["results"]) > 0:
                            print("📄 Sample results:")
                            for row in result["results"][:2]:
                                print(f"   {row}")

                        # Update conversation context for next turn
                        conversation_context.append(
                            {"role": "user", "content": question}
                        )
                        conversation_context.append(
                            {"role": "assistant", "content": result["query"]}
                        )

                    else:
                        print(f"❌ Error: {result['error']}")

                except Exception as e:
                    print(f"💥 Exception: {str(e)}")
        else:
            print("\n⚠️  Skipping advanced AI examples (no API keys configured)")
            print(
                "The basic SQLite functionality works! Set API keys to test natural language queries."
            )

        print("\n✨ Example completed!")
        print("\nKey Features Demonstrated:")
        print("• SQLite database creation and setup")
        print("• Natural language to SQL conversion")
        print("• Business glossary integration")
        print("• Conversational context")
        print("• Comprehensive error handling")

    finally:
        # Clean up database file
        try:
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up database file: {db_path}")
        except Exception:
            pass


def setup_environment():
    """Set up environment variables and validate configuration."""

    required_env_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API key for Claude models",
        "OPENAI_API_KEY": "OpenAI API key for GPT models",
        "GEMINI_API_KEY": "Google API key for Gemini models",
    }

    print("🔧 Environment Setup")
    print("-" * 20)

    missing_vars = []
    for var, description in required_env_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  • {var}: {description}")
        else:
            print(f"✅ {var}: Set")

    if missing_vars:
        print("\n⚠️  Missing environment variables:")
        for var in missing_vars:
            print(var)
        print("\nPlease set at least one API key to run the example.")

    print("\n📋 SQLite Configuration")
    print("✅ No additional database setup required!")
    print("This example creates a temporary SQLite database with sample data.")
    print("SQLite is included with Python, so no external database is needed.")
    print()


if __name__ == "__main__":
    # Set up environment and run example
    setup_environment()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        print(f"\n💥 Fatal error: {str(e)}")
        print("Check your database connections and API keys.")
