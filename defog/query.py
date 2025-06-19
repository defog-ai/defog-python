import re
import asyncio


# execute query for given db_type and return column names and data
def execute_query_once(db_type: str, db_creds, query: str):
    """
    Executes the query once and returns the column names and results.
    """
    if db_type == "postgres":
        try:
            import psycopg2
        except ImportError:
            raise Exception("psycopg2 not installed.")
        with psycopg2.connect(**db_creds) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                colnames = [desc.name for desc in cur.description]
                rows = cur.fetchall()
        return colnames, rows

    elif db_type == "redshift":
        try:
            import psycopg2
        except ImportError:
            raise Exception("redshift_connector not installed.")

        if "schema" not in db_creds:
            schema = "public"
            conn_args = db_creds
        else:
            schema = db_creds["schema"]
            db_creds = db_creds.copy()
            del db_creds["schema"]
            conn_args = db_creds

        with psycopg2.connect(**conn_args) as conn:
            with conn.cursor() as cur:
                if schema is not None and schema != "public":
                    cur.execute(f"SET search_path TO {schema}")

                cur.execute(query)
                colnames = [desc.name for desc in cur.description]

                # if there are any column names that are the same, we need to deduplicate them
                colnames = [
                    f"{col}_{i}" if colnames.count(col) > 1 else col
                    for i, col in enumerate(colnames)
                ]
                rows = cur.fetchall()
        return colnames, rows

    elif db_type == "mysql":
        try:
            import mysql.connector
        except ImportError:
            raise Exception("mysql.connector not installed.")
        with mysql.connector.connect(**db_creds) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        return colnames, rows

    elif db_type == "bigquery":
        try:
            from google.cloud import bigquery
        except ImportError:
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
        except ImportError:
            raise Exception("snowflake.connector not installed.")
        with snowflake.connector.connect(
            user=db_creds["user"],
            password=db_creds["password"],
            account=db_creds["account"],
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"USE WAREHOUSE {db_creds['warehouse']}"
                )  # set the warehouse
                if "database" in db_creds:
                    cur.execute(f"USE DATABASE {db_creds['database']}")
                cur.execute(query)
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        return colnames, rows

    elif db_type == "databricks":
        try:
            from databricks import sql
        except ImportError:
            raise Exception("databricks-sql-connector not installed.")
        with sql.connect(**db_creds) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                colnames = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
        return colnames, rows

    elif db_type == "sqlserver":
        try:
            import pyodbc
        except ImportError:
            raise Exception("pyodbc not installed.")

        if "database" in db_creds and db_creds["database"] != "":
            connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={db_creds['server']};DATABASE={db_creds['database']};UID={db_creds['user']};PWD={db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
        else:
            connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={db_creds['server']};UID={db_creds['user']};PWD={db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
        with pyodbc.connect(connection_string) as conn:
            cur = conn.cursor()
            cur.execute(query)
            colnames = [desc[0] for desc in cur.description]
            results = cur.fetchall()
            rows = [list(row) for row in results]
            cur.close()
        return colnames, rows

    elif db_type == "sqlite":
        try:
            import sqlite3
        except ImportError as e:
            raise ImportError(
                "sqlite3 module not available. This should be included with Python by default."
            ) from e

        database_path = db_creds.get("database", ":memory:")
        if database_path != ":memory:" and not isinstance(database_path, str):
            raise ValueError("Database path must be a string or ':memory:'")
        with sqlite3.connect(database_path) as conn:
            cur = conn.cursor()
            cur.execute(query)
            colnames = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
        return colnames, rows

    elif db_type == "duckdb":
        try:
            import duckdb
        except ImportError as e:
            raise ImportError(
                "duckdb not installed. Please install it with `pip install duckdb`."
            ) from e

        database_path = db_creds.get("database", ":memory:")
        if database_path != ":memory:" and not isinstance(database_path, str):
            raise ValueError("Database path must be a string or ':memory:'")

        # DuckDB supports both file-based and in-memory databases
        with duckdb.connect(database_path, read_only=True) as conn:
            # Execute the query and fetch results
            result = conn.execute(query)
            colnames = (
                [desc[0] for desc in result.description] if result.description else []
            )
            rows = result.fetchall()
        return colnames, rows

    else:
        raise Exception(f"Database type {db_type} not yet supported.")


async def async_execute_query_once(db_type: str, db_creds, query: str):
    """
    Asynchronously  executes the query once and returns the column names and results.
    """
    if db_type == "postgres":
        try:
            import asyncpg
        except ImportError:
            raise Exception("asyncpg not installed.")

        conn = await asyncpg.connect(**db_creds)
        try:
            results = await conn.fetch(query)
        finally:
            await conn.close()

        if results and len(results) > 0:
            colnames = list(results[0].keys())
        else:
            colnames = []

        # get the results in a list of lists format
        rows = [list(row.values()) for row in results]
        return colnames, rows

    elif db_type == "redshift":
        try:
            import asyncpg
        except ImportError:
            raise Exception("asyncpg not installed.")

        if "schema" not in db_creds:
            schema = "public"
            conn_args = db_creds
        else:
            schema = db_creds["schema"]
            db_creds = db_creds.copy()
            del db_creds["schema"]
            conn_args = db_creds

        conn = await asyncpg.connect(**conn_args)
        try:
            if schema is not None and schema != "public":
                await conn.execute(f"SET search_path TO {schema}")

            results = await conn.fetch(query)
            if results and len(results) > 0:
                colnames = list(results[0].keys())
            else:
                colnames = []

            # deduplicate the column names
            colnames = [
                f"{col}_{i}" if colnames.count(col) > 1 else col
                for i, col in enumerate(colnames)
            ]
            rows = [list(row.values()) for row in results]
        finally:
            await conn.close()
        return colnames, rows

    elif db_type == "mysql":
        try:
            import aiomysql
        except ImportError:
            raise Exception("aiomysql not installed.")
        db_creds = db_creds.copy()
        db_creds["db"] = db_creds["database"]
        del db_creds["database"]
        async with aiomysql.connect(**db_creds) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                colnames = [desc[0] for desc in cur.description]
                rows = await cur.fetchall()
        return colnames, rows

    elif db_type == "bigquery":
        try:
            from google.cloud import bigquery
        except ImportError:
            raise Exception("google.cloud.bigquery not installed.")
        # using asynico.to_thread since google-cloud-bigquery is synchronous
        json_key = db_creds["json_key_path"]
        client = await asyncio.to_thread(
            bigquery.Client.from_service_account_json, json_key
        )
        query_job = await asyncio.to_thread(client.query, query)
        results = await asyncio.to_thread(query_job.result)
        colnames = [i.name for i in results.schema]
        rows = []
        for row in results:
            rows.append([row[i] for i in range(len(row))])
        return colnames, rows

    elif db_type == "snowflake":
        try:
            import snowflake.connector
        except ImportError:
            raise Exception("snowflake.connector not installed.")
        with snowflake.connector.connect(
            user=db_creds["user"],
            password=db_creds["password"],
            account=db_creds["account"],
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"USE WAREHOUSE {db_creds['warehouse']}"
                )  # set the warehouse

                if "database" in db_creds:
                    cur.execute(
                        f"USE DATABASE {db_creds['database']}"
                    )  # set the database

                cur.execute_async(query)
                query_id = cur.sfqid
                while conn.is_still_running(conn.get_query_status(query_id)):
                    await asyncio.sleep(1)

                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        return colnames, rows

    elif db_type == "databricks":
        try:
            from databricks import sql
        except ImportError:
            raise Exception("databricks-sql-connector not installed.")
        conn = await asyncio.to_thread(sql.connect, **db_creds)
        try:
            cursor = await asyncio.to_thread(conn.cursor)
            try:
                await asyncio.to_thread(cursor.execute, query)
                colnames = [desc[0] for desc in cursor.description]
                rows = await asyncio.to_thread(cursor.fetchall)
            finally:
                await asyncio.to_thread(cursor.close)
        finally:
            await asyncio.to_thread(conn.close)
        return colnames, rows

    elif db_type == "sqlserver":
        try:
            import aioodbc
        except ImportError:
            raise Exception("aioodbc not installed.")

        if "database" in db_creds and db_creds["database"] != "":
            connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={db_creds['server']};DATABASE={db_creds['database']};UID={db_creds['user']};PWD={db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
        else:
            connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={db_creds['server']};UID={db_creds['user']};PWD={db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
        async with aioodbc.connect(dsn=connection_string) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                colnames = [desc[0] for desc in cur.description]
                results = await cur.fetchall()
                rows = [list(row) for row in results]
        return colnames, rows

    elif db_type == "sqlite":
        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError(
                "aiosqlite module not available. Please install with 'pip install aiosqlite' for async SQLite support."
            ) from e

        database_path = db_creds.get("database", ":memory:")
        if database_path != ":memory:" and not isinstance(database_path, str):
            raise ValueError("Database path must be a string or ':memory:'")
        async with aiosqlite.connect(database_path) as conn:
            cur = await conn.cursor()
            await cur.execute(query)
            colnames = [desc[0] for desc in cur.description] if cur.description else []
            rows = await cur.fetchall()
            await cur.close()
        return colnames, rows

    elif db_type == "duckdb":
        try:
            import duckdb
        except ImportError as e:
            raise ImportError(
                "duckdb not installed. Please install it with `pip install duckdb`."
            ) from e

        database_path = db_creds.get("database", ":memory:")
        if database_path != ":memory:" and not isinstance(database_path, str):
            raise ValueError("Database path must be a string or ':memory:'")

        # DuckDB doesn't have native async support, so we use asyncio.to_thread
        def _execute_duckdb():
            with duckdb.connect(database_path, read_only=True) as conn:
                result = conn.execute(query)
                colnames = (
                    [desc[0] for desc in result.description]
                    if result.description
                    else []
                )
                rows = result.fetchall()
            return colnames, rows

        colnames, rows = await asyncio.to_thread(_execute_duckdb)
        return colnames, rows

    else:
        raise Exception(f"Database type {db_type} not yet supported.")


def execute_query(
    query: str,
    db_type: str,
    db_creds,
    api_key: str = None,
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

    try:
        return execute_query_once(db_type, db_creds, query)
    except Exception as e:
        err_msg = str(e)
        if is_connection_error(err_msg):
            raise Exception(
                f"There was a connection issue to your database:\n{err_msg}\n\nPlease check your database credentials and try again."
            )
        else:
            raise Exception(err_msg)


async def async_execute_query(
    query: str,
    db_type: str,
    db_creds,
    api_key: str = None,
    question: str = "",
    hard_filters: str = "",
    retries: int = 3,
    schema: dict = None,
    dev: bool = False,
    temp: bool = False,
    base_url: str = None,
):
    """
    Execute the query asynchronously and retry with adaptive learning if there is an error.
    Raises an Exception if there are no retries left, or if the error is a connection error.
    """
    err_msg = None

    try:
        return await async_execute_query_once(
            db_type=db_type, db_creds=db_creds, query=query
        )
    except Exception as e:
        err_msg = str(e)
        if is_connection_error(err_msg):
            raise Exception(
                f"There was a connection issue to your database:\n{err_msg}\n\nPlease check your database credentials and try again."
            )

        raise Exception(err_msg)


def is_connection_error(err_msg: str) -> bool:
    return (
        isinstance(err_msg, str)
        and re.search(r"connection.*failed", err_msg) is not None
    )
