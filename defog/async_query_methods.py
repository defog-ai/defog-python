from defog.query import async_execute_query
from datetime import datetime
from defog.llm.sql_generator import generate_sql_query_local
from defog.llm.llm_providers import LLMProvider
from defog.metadata_cache import get_global_cache
from defog.local_metadata_extractor import extract_metadata_from_db_async
from typing import Union, Optional


async def get_query(
    self,
    question: str,
    hard_filters: str = "",
    previous_context: list = [],
    glossary: str = "",
    debug: bool = False,
    dev: bool = False,
    temp: bool = False,
    profile: bool = False,
    ignore_cache: bool = False,
    model: str = "",
    use_golden_queries: bool = True,
    subtable_pruning: bool = False,
    glossary_pruning: bool = False,
    prune_max_tokens: int = 2000,
    prune_bm25_num_columns: int = 10,
    prune_glossary_max_tokens: int = 1000,
    prune_glossary_num_cos_sim_units: int = 10,
    prune_glossary_bm25_units: int = 10,
    use_llm_directly: bool = True,  # Deprecated parameter, always True now
    llm_provider: Optional[Union[LLMProvider, str]] = None,
    llm_model: Optional[str] = None,
    table_metadata: Optional[dict] = None,
    cache_metadata: bool = False,
):
    """
    Asynchronously generates SQL query using local LLM and returns the response.
    :param question: The question to be asked.
    :param llm_provider: LLM provider to use for generation (default: Anthropic)
    :param llm_model: Model name for generation (default: claude-sonnet-4-20250514)
    :param table_metadata: Database schema for generation
    :param cache_metadata: Whether to cache metadata (default: True)
    :return: The generated SQL query and metadata.
    """
    # Always use local LLM generation
    if table_metadata is None:
        # Try to get from cache first
        cache = get_global_cache()
        table_metadata = cache.get(self.api_key, self.db_type, dev)

        if table_metadata is None:
            # Not in cache, extract metadata directly from the database
            try:
                table_metadata = await extract_metadata_from_db_async(
                    db_type=self.db_type,
                    db_creds=self.db_creds,
                    cache=cache if cache_metadata else None,
                    api_key=self.api_key,
                )
            except Exception as e:
                return {
                    "ran_successfully": False,
                    "error_message": f"Failed to extract database metadata: {str(e)}. Please provide table_metadata parameter or check database connection.",
                }

    # Set defaults for provider and model if not specified
    if llm_provider is None:
        llm_provider = LLMProvider.ANTHROPIC
    if llm_model is None:
        llm_model = "claude-sonnet-4-20250514"

    t_start = datetime.now()
    result = await generate_sql_query_local(
        question=question,
        table_metadata=table_metadata,
        db_type=self.db_type,
        provider=llm_provider,
        model=llm_model,
        glossary=glossary,
        hard_filters=hard_filters,
        previous_context=previous_context,
        temperature=0.0,
        config=getattr(self, "llm_config", None),
    )
    t_end = datetime.now()

    if profile:
        result["time_taken"] = (t_end - t_start).total_seconds()

    return result


async def run_query(
    self,
    question: str,
    hard_filters: str = "",
    previous_context: list = [],
    glossary: str = "",
    query: dict = None,
    retries: int = 3,
    dev: bool = False,
    temp: bool = False,
    profile: bool = False,
    ignore_cache: bool = False,
    model: str = "",
    use_golden_queries: bool = True,
    subtable_pruning: bool = False,
    glossary_pruning: bool = False,
    prune_max_tokens: int = 2000,
    prune_bm25_num_columns: int = 10,
    prune_glossary_max_tokens: int = 1000,
    prune_glossary_num_cos_sim_units: int = 10,
    prune_glossary_bm25_units: int = 10,
    use_llm_directly: bool = True,  # Deprecated parameter, always True now
    llm_provider: Optional[Union[LLMProvider, str]] = None,
    llm_model: Optional[str] = None,
    table_metadata: Optional[dict] = None,
    cache_metadata: bool = False,
):
    """
    Asynchronously sends the question to the defog servers, executes the generated SQL,
    and returns the response.
    :param question: The question to be asked.
    :return: The response from the defog server.
    """
    if query is None:
        print(f"Generating the query for your question: {question}...")
        query = await self.get_query(
            question,
            hard_filters,
            previous_context,
            glossary=glossary,
            dev=dev,
            temp=temp,
            profile=profile,
            model=model,
            ignore_cache=ignore_cache,
            use_golden_queries=use_golden_queries,
            subtable_pruning=subtable_pruning,
            glossary_pruning=glossary_pruning,
            prune_max_tokens=prune_max_tokens,
            prune_bm25_num_columns=prune_bm25_num_columns,
            prune_glossary_max_tokens=prune_glossary_max_tokens,
            prune_glossary_num_cos_sim_units=prune_glossary_num_cos_sim_units,
            prune_glossary_bm25_units=prune_glossary_bm25_units,
            use_llm_directly=use_llm_directly,
            llm_provider=llm_provider,
            llm_model=llm_model,
            table_metadata=table_metadata,
            cache_metadata=cache_metadata,
        )
    if query["ran_successfully"]:
        try:
            print("Query generated, now running it on your database...")
            tstart = datetime.now()
            colnames, result = await async_execute_query(
                query=query["query_generated"],
                api_key=self.api_key,
                db_type=self.db_type,
                db_creds=self.db_creds,
                question=question,
                hard_filters=hard_filters,
                retries=retries,
                dev=dev,
                temp=temp,
            )
            tend = datetime.now()
            time_taken = (tend - tstart).total_seconds()
            resp = {
                "columns": colnames,
                "data": result,
                "query_generated": query["query_generated"],
                "ran_successfully": True,
                "reason_for_query": query.get("reason_for_query"),
                "previous_context": query.get("previous_context"),
            }
            if profile:
                resp["execution_time_taken"] = time_taken
                resp["generation_time_taken"] = query.get("time_taken")
            return resp
        except Exception as e:
            return {
                "ran_successfully": False,
                "error_message": str(e),
                "query_generated": query["query_generated"],
            }
    else:
        return {"ran_successfully": False, "error_message": query["error_message"]}
