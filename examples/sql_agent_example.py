#!/usr/bin/env python3
"""
SQL Agent Example

This example demonstrates how to use the SQL tools for local database operations,
including natural language to SQL query conversion and automatic table relevance
filtering for large databases.

Features demonstrated:
- sql_answer_tool: Convert natural language questions to SQL and execute them
- identify_relevant_tables_tool: Find relevant tables in large databases
- Automatic table filtering for databases with >1000 columns and >5 tables
- Error handling and fallback scenarios
"""

import asyncio
import os
import logging
from typing import Dict, Any

from defog.llm.sql import sql_answer_tool, identify_relevant_tables_tool
from defog.llm.llm_providers import LLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLAgent:
    """A SQL agent that can answer natural language questions about databases."""
    
    def __init__(self, db_type: str, db_creds: Dict[str, Any], 
                 provider: LLMProvider = LLMProvider.ANTHROPIC,
                 model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the SQL agent.
        
        Args:
            db_type: Database type (postgres, mysql, bigquery, etc.)
            db_creds: Database connection credentials
            provider: LLM provider to use
            model: Model name
        """
        self.db_type = db_type
        self.db_creds = db_creds
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
            **kwargs
        )
        
        if result["success"]:
            logger.info(f"Query successful! Returned {len(result["results"])} rows")
        else:
            logger.error(f"Query failed: {result['error']}")
            
        return result

async def main():
    """Example usage of the SQL agent."""
    
    # Example 1: PostgreSQL connection
    postgres_creds = {
        "host": "localhost",
        "port": 5432,
        "database": "cricket",
        "user": "postgres",
        "password": "postgres"
    }
    
    # Create SQL agent
    agent = SQLAgent(
        db_type="postgres",
        db_creds=postgres_creds,
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514"
    )
    
    print("🤖 SQL Agent Example - Natural Language Database Queries")
    print("=" * 60)
    
    # Example questions for an e-commerce database
    questions = [
        "What are the total runs scored by each of the top 10 batsmen?",
        "What are the top 10 bowlers by wickets taken?",
        "What are the top 10 bowlers by economy rate (min 100 balls)?",
    ]
    
    # Example 1: Basic question answering
    print("\n📊 Example 1: Basic Question Answering")
    print("-" * 40)
    
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
                        print(f"   ... and {len(result['results']) - 3} more rows")
                        
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    # Example 2: Using business glossary and filters
    print("\n📚 Example 2: Using Business Glossary and Hard Filters")
    print("-" * 50)
    
    glossary = """
    Average: The total number of runs scored divided by the number of times out
    Strike Rate: The total number of runs scored divided by the number of balls faced
    Economy Rate: The total number of runs conceded divided by the number of balls bowled, multiplied by 6
    """
    
    question = "Who are the top 10 batsmen by average (min 200 runs)?"
    print(f"\nQuestion: {question}")
    
    try:
        result = await agent.ask_question(
            question=question,
            glossary=glossary,
        )
        
        if result["success"]:
            print(f"✅ Query with filters: {result['query']}")
            print(f"📈 Results: {len(result['results'])} rows")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"💥 Exception: {str(e)}")
    
    # Example 4: Conversation context
    print("\n💬 Example 4: Conversational Context")
    print("-" * 38)

    conversation_context = []
    
    # Simulate a conversation with context
    questions_with_context = [
        "Who are the top 10 batsmen by average (min 200 runs)?",
        "How about the next 10?",
        "Who are the top 10 bowlers by economy rate (min 100 balls)?"
    ]
    
    for i, question in enumerate(questions_with_context, 1):
        print(f"\nTurn {i}: {question}")
        
        try:
            result = await agent.ask_question(
                question=question,
                previous_context=conversation_context
            )
            
            if result["success"]:
                print(f"✅ Query: {result['query']}")
                print(f"📈 Results: {len(result['results'])} rows")
                
                # Update conversation context for next turn
                conversation_context.append({"role": "user", "content": question})
                conversation_context.append({"role": "assistant", "content": result['query']})
                
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    print("\n✨ Example completed!")
    print("\nKey Features Demonstrated:")
    print("• Natural language to SQL conversion")
    print("• Automatic table filtering for large databases")
    print("• Business glossary and hard filters")
    print("• Table relevance analysis")
    print("• Conversational context")
    print("• Comprehensive error handling")


def setup_environment():
    """Set up environment variables and validate configuration."""
    
    required_env_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API key for Claude models",
        "OPENAI_API_KEY": "OpenAI API key for GPT models", 
        "GEMINI_API_KEY": "Google API key for Gemini models"
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
        print("\nPlease set these environment variables to run the full example.")
        print("You can still run parts of the example with the providers you have configured.")
    
    print("\n📋 Database Configuration")
    print("Update the database credentials in this file to match your setup:")
    print("  • postgres_creds: Your PostgreSQL connection details")
    print("  • mysql_creds: Your MySQL connection details") 
    print("  • bigquery_creds: Your BigQuery project and credentials")
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