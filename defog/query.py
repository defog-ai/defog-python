import json
import requests
from defog.util import write_logs


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
        cur.execute(query)
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
    elif db_type == "elastic":
        host = db_creds["host"]
        if host.endswith("/"):
            host = host[:-1]
        url = host + "/_sql"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"ApiKey {db_creds['api_key']}",
        }
        r = requests.post(
            url, headers=headers, data=json.dumps({"query": query}).replace(";", "")
        )
        if r.status_code != 200:
            raise Exception(f"Error executing query: {r.json()['error']}")
        else:
            results = r.json()["rows"]
            colnames = [i["name"] for i in r.json()["columns"]]
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
):
    err_msg = None
    try:
        return execute_query_once(db_type, db_creds, query) + (query,)
    except Exception as e:
        err_msg = str(e)
        print(
            "There was an error when running the previous query. Retrying with adaptive learning..."
        )

        # log this error to our feedback system
        try:
            r = requests.post(
                "https://api.defog.ai/feedback",
                json={
                    "api_key": api_key,
                    "feedback": "bad",
                    "text": err_msg,
                    "db_type": db_type,
                    "question": question,
                    "query": query,
                },
                timeout=1,
            )
        except:
            pass

        write_logs(str(e))
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
                }
                if schema is not None:
                    retry["schema"] = schema
                write_logs(json.dumps(retry))
                r = requests.post(
                    "https://api.defog.ai/retry_query_after_error",
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
