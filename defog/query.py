import json
import re
import requests
from defog.util import write_logs
import os


# execute query for given db_type and return column names and data
def execute_query_once(db_type: str, db_creds, query: str):
    """
    Executes the query once and returns the column names and results.
    """
    if db_type == "postgres":
        try:
            import psycopg2
        except:
            raise Exception("psycopg2 not installed.")
        conn = psycopg2.connect(**db_creds)
        cur = conn.cursor()
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()
        conn.close()
        return colnames, results
    elif db_type == "redshift":
        try:
            import psycopg2
        except:
            raise Exception("redshift_connector not installed.")

        if "schema" not in db_creds:
            schema = "public"
            conn = psycopg2.connect(**db_creds)
        else:
            schema = db_creds["schema"]
            del db_creds["schema"]
            conn = psycopg2.connect(**db_creds)

        cur = conn.cursor()

        if schema is not None and schema != "public":
            cur.execute(f"SET search_path TO {schema}")

        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]

        # if there are any column names that are the same, we need to deduplicate them
        colnames = [
            f"{col}_{i}" if colnames.count(col) > 1 else col
            for i, col in enumerate(colnames)
        ]

        results = cur.fetchall()
        cur.close()
        conn.close()
        return colnames, results
    elif db_type == "mysql":
        try:
            import mysql.connector
        except:
            raise Exception("mysql.connector not installed.")
        conn = mysql.connector.connect(**db_creds)
        cur = conn.cursor()
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()
        conn.close()
        return colnames, results
    elif db_type == "bigquery":
        try:
            from google.cloud import bigquery
        except:
            raise Exception("google.cloud.bigquery not installed.")

        json_key = db_creds["json_key_path"]
        client = bigquery.Client.from_service_account_json(json_key)
        query_job = client.query(query)
        results = query_job.result()
        colnames = [i.name for i in results.schema]
        rows = []
        for row in results:
            rows.append([row[i] for i in range(len(row))])
        return colnames, rows
    elif db_type == "snowflake":
        try:
            import snowflake.connector
        except:
            raise Exception("snowflake.connector not installed.")
        conn = snowflake.connector.connect(
            user=db_creds["user"],
            password=db_creds["password"],
            account=db_creds["account"],
        )
        cur = conn.cursor()
        cur.execute(f"USE WAREHOUSE {db_creds['warehouse']}")  # set the warehouse
        if "database" in db_creds:
            cur.execute(f"USE DATABASE {db_creds['database']}")
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()
        conn.close()
        return colnames, results
    elif db_type == "databricks":
        try:
            from databricks import sql
        except:
            raise Exception("databricks-sql-connector not installed.")
        conn = sql.connect(**db_creds)
        with conn.cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
        return colnames, results
    elif db_type == "sqlserver":
        try:
            import pyodbc
        except:
            raise Exception("pyodbc not installed.")

        if db_creds["database"] != "":
            connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={db_creds['server']};DATABASE={db_creds['database']};UID={db_creds['user']};PWD={db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
        else:
            connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={db_creds['server']};UID={db_creds['user']};PWD={db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
        conn = pyodbc.connect(connection_string)
        cur = conn.cursor()
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        results = [list(row) for row in results]
        cur.close()
        conn.close()
        return colnames, results
    else:
        raise Exception(f"Database type {db_type} not yet supported.")


def execute_query(
    query: str,
    api_key: str,
    db_type: str,
    db_creds,
    question: str = "",
    hard_filters: str = "",
    retries: int = 3,
    schema: dict = None,
    dev: bool = False,
    temp: bool = False,
    base_url: str = None,
):
    """
    Execute the query and retry with adaptive learning if there is an error.
    Raises an Exception if there are no retries left, or if the error is a connection error.
    """
    err_msg = None
    # if base_url is not explicitly defined, check if DEFOG_BASE_URL is set in the environment
    # if not, then use "https://api.defog.ai" as the default
    if base_url is None:
        base_url = os.environ.get("DEFOG_BASE_URL", "https://api.defog.ai")

    try:
        return execute_query_once(db_type, db_creds, query) + (query,)
    except Exception as e:
        err_msg = str(e)
        if is_connection_error(err_msg):
            raise Exception(
                f"There was a connection issue to your database:\n{err_msg}\n\nPlease check your database credentials and try again."
            )
        # log this error to our feedback system first (this is a 1-way side-effect)
        try:
            requests.post(
                f"{base_url}/feedback",
                json={
                    "api_key": api_key,
                    "feedback": "bad",
                    "text": err_msg,
                    "db_type": db_type,
                    "question": question,
                    "query": query,
                    "dev": dev,
                    "temp": temp,
                },
                timeout=1,
            )
        except:
            pass
        # log locally
        write_logs(str(e))
        # retry with adaptive learning
        while retries > 0:
            write_logs(f"Retries left: {retries}")
            try:
                retry = {
                    "api_key": api_key,
                    "previous_query": query,
                    "error": err_msg,
                    "db_type": db_type,
                    "hard_filters": hard_filters,
                    "question": question,
                    "dev": dev,
                    "temp": temp,
                }
                if schema is not None:
                    retry["schema"] = schema
                write_logs(json.dumps(retry))
                r = requests.post(
                    f"{base_url}/retry_query_after_error",
                    json=retry,
                )
                response = r.json()
                new_query = response["new_query"]
                write_logs(f"New query: \n{new_query}")
                return execute_query_once(db_type, db_creds, new_query) + (new_query,)
            except Exception as e:
                err_msg = str(e)
                print(
                    "There was an error when running the previous query. Retrying with adaptive learning..."
                )
                write_logs(str(e))
                retries -= 1
        raise Exception(err_msg)


def is_connection_error(err_msg: str) -> bool:
    return (
        isinstance(err_msg, str)
        and re.search(r"connection.*failed", err_msg) is not None
    )
