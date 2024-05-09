import requests
from defog.query import execute_query
from datetime import datetime


def get_query(
    self,
    question: str,
    hard_filters: str = "",
    previous_context: list = [],
    schema: dict = {},
    glossary: str = "",
    language: str = None,
    debug: bool = False,
    dev: bool = False,
    profile: bool = False,
    ignore_cache: bool = False,
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
            "ignore_cache": ignore_cache,
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
    schema: dict = {},
    glossary: str = "",
    mode: str = "chat",
    language: str = None,
    query: dict = None,
    retries: int = 3,
    dev: bool = False,
    profile: bool = False,
    ignore_cache: bool = False,
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
            schema=schema,
            glossary=glossary,
            language=language,
            dev=dev,
            profile=profile,
            ignore_cache=ignore_cache,
        )
    if query["ran_successfully"]:
        try:
            print("Query generated, now running it on your database...")
            tstart = datetime.now()
            colnames, result, executed_query = execute_query(
                query["query_generated"],
                self.api_key,
                self.db_type,
                self.db_creds,
                question,
                hard_filters,
                retries,
                dev=dev,
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
