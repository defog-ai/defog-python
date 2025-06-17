# Database Operations

This document covers all database-related functionality in the defog-python library.

## Supported Database Types

The library supports the following database types:

- **PostgreSQL** (`postgres`)
- **MySQL** (`mysql`)
- **BigQuery** (`bigquery`)
- **Snowflake** (`snowflake`)
- **Databricks** (`databricks`)
- **SQL Server** (`sqlserver`)
- **Redshift** (`redshift`)
- **SQLite** (`sqlite`)
- **DuckDB** (`duckdb`)
- **MongoDB** (`mongo`) - *Undocumented in main README*
- **Elasticsearch** (`elastic`) - *Undocumented in main README*

## Query Execution

### Basic Query Execution

```python
from defog.query import execute_query, async_execute_query

# Synchronous execution
colnames, results = execute_query(
    query="SELECT * FROM customers LIMIT 10",
    db_type="postgres",
    db_creds={
        "host": "localhost",
        "port": 5432,
        "database": "mydb",
        "user": "postgres",
        "password": "password"
    }
)

# Asynchronous execution (recommended)
colnames, results = await async_execute_query(
    query="SELECT COUNT(*) FROM orders",
    db_type="mysql",
    db_creds={...}
)
```

### Advanced Query Parameters

The `run_query()` method supports many undocumented parameters:

```python
from defog import Defog

df = Defog(api_key="your-api-key")

# Advanced query with all parameters
result = await df.run_query(
    question="Show top customers",
    
    # Pruning options
    subtable_pruning=True,            # Enable intelligent table filtering
    glossary_pruning=True,            # Enable glossary-based filtering
    prune_max_tokens=5000,            # Token limit for pruning
    prune_bm25_num_columns=30,        # BM25 column pruning parameter
    prune_glossary_max_tokens=2000,   # Glossary pruning token limit
    prune_glossary_num_cos_sim_units=5,  # Cosine similarity units
    prune_glossary_bm25_units=10,     # BM25 units for glossary
    
    # Metadata options
    table_metadata=custom_metadata,    # Pass metadata directly
    cache_metadata=True,              # Enable metadata caching
    
    # Other options
    return_error=True,                # Return errors instead of raising
    schema_whitelist=["public", "sales"],  # Limit to specific schemas
)
```

## Metadata Extraction

### Local Metadata Extraction

Extract metadata directly from databases without API calls:

```python
from defog.local_metadata_extractor import (
    extract_metadata_from_db,
    extract_metadata_from_db_async
)

# Synchronous extraction
metadata = extract_metadata_from_db(
    db_type="postgres",
    db_creds=db_creds
)

# Asynchronous extraction
metadata = await extract_metadata_from_db_async(
    db_type="mysql",
    db_creds=db_creds
)
```

### Metadata Caching

Improve performance with intelligent caching:

```python
from defog.metadata_cache import MetadataCache

# Initialize cache with custom TTL
cache = MetadataCache(ttl_seconds=7200)  # 2 hour TTL

# Generate cache key
cache_key = cache.generate_cache_key(db_type, db_creds)

# Set metadata
cache.set(cache_key, metadata)

# Get metadata
cached_metadata = cache.get(cache_key)

# Invalidate specific cache
cache.invalidate(cache_key)

# Clear all cache
cache.clear_all()
```

## Schema Documentation

### Automatic Schema Documentation

Generate intelligent documentation for your database schemas:

```python
from defog.llm.schema_documenter import SchemaDocumenter

# Initialize documenter
documenter = SchemaDocumenter(
    model="claude-3-5-sonnet",
    provider="anthropic",
    db_type="postgres",
    db_creds=db_creds
)

# Document all tables
all_docs = await documenter.document_all_tables()

# Document specific tables
specific_docs = await documenter.document_tables(
    tables=["customers", "orders"],
    include_existing_comments=True,  # Preserve DB comments
    batch_size=5                     # Tables per batch
)

# Get documentation with confidence scores
for table_doc in specific_docs:
    print(f"Table: {table_doc['table_name']}")
    print(f"Description: {table_doc['table_description']}")
    print(f"Confidence: {table_doc['confidence']}")
    
    for col in table_doc['columns']:
        print(f"  Column: {col['column_name']}")
        print(f"  Type: {col['data_type']}")
        print(f"  Description: {col['description']}")
        print(f"  Confidence: {col['confidence']}")
        if col.get('is_categorical'):
            print(f"  Categories: {col['categorical_values']}")
```

### Schema Documentation Features

The schema documenter includes:

- **Automatic pattern detection**: Emails, URLs, UUIDs, phone numbers
- **Categorical column identification**: Detects columns with limited unique values
- **Confidence scoring**: Each description includes a confidence score
- **Database comment integration**: Preserves existing database documentation
- **Batch processing**: Efficient handling of large schemas

## SQL Agent

### Natural Language to SQL

Convert questions to SQL queries:

```python
from defog.llm.sql import sql_answer_tool
from defog.llm.llm_providers import LLMProvider

result = await sql_answer_tool(
    question="What are the top 10 customers by revenue?",
    db_type="postgres",
    db_creds=db_creds,
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    
    # Optional parameters
    glossary="Revenue: Total sales amount per customer",
    previous_context=[],              # Conversation history
    whitelist_tables=["customers", "orders"],  # Limit tables
    domain_filter="retail",           # Domain-specific filters
    is_delegated_tool_call=False,     # For agent orchestration
    verbose=True                      # Detailed logging
)

if result["success"]:
    print(f"SQL: {result['query']}")
    print(f"Results: {result['results']}")
    print(f"Explanation: {result.get('explanation', '')}")
```

### Table Relevance Identification

For large databases, identify relevant tables:

```python
from defog.llm.sql import identify_relevant_tables_tool

relevance_result = await identify_relevant_tables_tool(
    question="Show customer purchase history",
    db_type="postgres",
    db_creds=db_creds,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    max_tables=10,                    # Maximum tables to return
    whitelist_tables=None,            # Optional table filter
    glossary=None                     # Optional business context
)

print(f"Relevant tables: {relevance_result['tables']}")
print(f"Reasoning: {relevance_result['reasoning']}")
```

## Database Utilities

### Categorical Column Detection

Identify categorical columns in your data:

```python
from defog.util import (
    identify_categorical_columns,
    async_identify_categorical_columns
)

# Synchronous detection
categorical_cols = identify_categorical_columns(
    df=dataframe,
    threshold=0.1  # 10% uniqueness threshold
)

# Asynchronous detection  
categorical_cols = await async_identify_categorical_columns(
    df=dataframe,
    threshold=0.1
)
```

### DDL Generation

Generate CREATE TABLE statements:

```python
from defog.admin_methods import create_table_ddl, create_ddl_from_metadata

# From column definitions
ddl = create_table_ddl(
    table_name="customers",
    column_names=["id", "name", "email"],
    column_types=["integer", "varchar(100)", "varchar(255)"]
)

# From metadata
ddl_statements = create_ddl_from_metadata(metadata)
```

## MongoDB and Elasticsearch Support

### MongoDB Queries

```python
# MongoDB connection
mongo_creds = {
    "connection_string": "mongodb://localhost:27017/",
    "database": "mydb"
}

# Execute aggregation pipeline
colnames, results = await async_execute_query(
    query='[{"$match": {"status": "active"}}, {"$limit": 10}]',
    db_type="mongo",
    db_creds=mongo_creds
)
```

### Elasticsearch Queries

```python
# Elasticsearch connection
elastic_creds = {
    "host": "localhost",
    "port": 9200,
    "user": "elastic",
    "password": "password",
    "use_ssl": True
}

# Execute search query
colnames, results = await async_execute_query(
    query='{"query": {"match": {"title": "python"}}}',
    db_type="elastic",
    db_creds=elastic_creds
)
```

## Best Practices

1. **Use async methods** when possible for better performance
2. **Enable metadata caching** for repeated queries on the same database
3. **Use table whitelisting** for large databases to improve performance
4. **Provide glossaries** for domain-specific terminology
5. **Batch operations** when documenting multiple tables
6. **Monitor costs** using the cost tracking features