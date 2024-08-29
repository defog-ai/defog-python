import requests
from defog.query import execute_query
from datetime import datetime


def get_query(
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
):
    """
    Sends the query to the defog servers, and return the response.
    :param question: The question to be asked.
    :return: The response from the defog server.
    """
    try:
        data = {
            "question": question,
            "api_key": self.api_key,
            "previous_context": previous_context,
            "db_type": self.db_type if self.db_type != "databricks" else "postgres",
            "glossary": glossary,
            "hard_filters": hard_filters,
            "dev": dev,
            "temp": temp,
            "ignore_cache": ignore_cache,
            "model": model,
            "use_golden_queries": use_golden_queries,
            "subtable_pruning": subtable_pruning,
            "glossary_pruning": glossary_pruning,
            "prune_max_tokens": prune_max_tokens,
            "prune_bm25_num_columns": prune_bm25_num_columns,
            "prune_glossary_max_tokens": prune_glossary_max_tokens,
            "prune_glossary_num_cos_sim_units": prune_glossary_num_cos_sim_units,
            "prune_glossary_bm25_units": prune_glossary_bm25_units,
        }

        t_start = datetime.now()
        r = requests.post(
            self.generate_query_url,
            json=data,
            timeout=300,
        )
        resp = r.json()
        t_end = datetime.now()
        time_taken = (t_end - t_start).total_seconds()
        query_generated = resp.get("sql", resp.get("query_generated"))
        ran_successfully = resp.get("ran_successfully")
        error_message = resp.get("error_message")
        query_db = self.db_type
        resp = {
            "query_generated": query_generated,
            "ran_successfully": ran_successfully,
            "error_message": error_message,
            "query_db": query_db,
            "previous_context": resp.get("previous_context"),
            "reason_for_query": resp.get("reason_for_query"),
        }
        if profile:
            resp["time_taken"] = time_taken

        return resp
    except Exception as e:
        if debug:
            print(e)
        return {
            "ran_successfully": False,
            "error_message": "Sorry :( Our server is at capacity right now and we are unable to process your query. Please try again in a few minutes?",
        }


def run_query(
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
):
    """
    Sends the question to the defog servers, executes the generated SQL,
    and returns the response.
    :param question: The question to be asked.
    :return: The response from the defog server.
    """
    if query is None:
        print(f"Generating the query for your question: {question}...")
        query = self.get_query(
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
        )
    if query["ran_successfully"]:
        try:
            print("Query generated, now running it on your database...")
            tstart = datetime.now()
            colnames, result, executed_query = execute_query(
                query=query["query_generated"],
                api_key=self.api_key,
                db_type=self.db_type,
                db_creds=self.db_creds,
                question=question,
                hard_filters=hard_filters,
                retries=retries,
                dev=dev,
                temp=temp,
                base_url=self.base_url,
            )
            tend = datetime.now()
            time_taken = (tend - tstart).total_seconds()
            resp = {
                "columns": colnames,
                "data": result,
                "query_generated": executed_query,
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
