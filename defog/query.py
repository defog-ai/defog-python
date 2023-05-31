import json

import requests


# execute query for given db_type and return column names and data
def execute_query_once(db_type: str, db_creds, query: str):
    """
    Executes the query once and returns the column names and results.
    """
    if db_type == "postgres" or db_type == "redshift":
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
    elif db_type == "mongo":
        try:
            from pymongo import MongoClient
        except:
            raise Exception("pymongo not installed.")
        client = MongoClient(db_creds["connection_string"])
        db = client.get_database()  # used in the query string passed to eval
        results = eval(f"{query}")
        results = [i for i in results]
        if len(results) > 0:
            colnames = results[0].keys()
        else:
            colnames = []
        return colnames, results
    elif db_type == "bigquery":
        try:
            from google.cloud import bigquery
        except:
            raise Exception("google.cloud.bigquery not installed.")

        json_key = db_creds['json_key_path']
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
        cur.execute(query["query_generated"])
        colnames = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()
        conn.close()
        return colnames, results
    elif db_type == "sqlserver":
        try:
            import pyodbc
        except:
            raise Exception("pyodbc not installed.")
        conn = pyodbc.connect(db_creds)
        cur = conn.cursor()
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        results = cur.fetchall()
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
    question: str,
    hard_filters: str,
    retries: int,
):
    err_msg = None
    try:
        return execute_query_once(db_type, db_creds, query) + (query,)
    except Exception as e:
        err_msg = str(e)
        print(
            f"""There was an error {err_msg} when running the previous query:
{query}
Retrying with adaptive learning..."""
        )
        while retries > 0:
            print("Retries left: ", retries)
            try:
                retry = {
                    "api_key": api_key,
                    "previous_query": query,
                    "error": err_msg,
                    "db_type": db_type,
                    "hard_filters": hard_filters,
                    "question": question,
                }
                print(json.dumps(retry))
                r = requests.post(
                    "https://api.defog.ai/retry_query_after_error",
                    json=retry,
                )
                response = r.json()
                new_query = response["new_query"]
                return execute_query_once(db_type, db_creds, new_query) + (new_query,)
            except Exception as e:
                err_msg = str(e)
                print(
                    f"""There was an error {err_msg} when retrying the previous query:
{query}
Retrying with adaptive learning..."""
                )
                retries -= 1
        raise Exception("Maximum retries exceeded.")
