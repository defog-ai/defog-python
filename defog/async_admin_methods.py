import json
from typing import Dict, List, Optional
from defog.util import make_async_post_request
import pandas as pd
import asyncio


async def update_db_schema(self, path_to_csv, dev=False, temp=False):
    """
    Update the DB schema via a CSV
    """
    schema_df = pd.read_csv(path_to_csv).fillna("")
    # check columns
    if not all(
        col in schema_df.columns
        for col in ["table_name", "column_name", "data_type", "column_description"]
    ):
        raise ValueError(
            "The CSV must contain the following columns: table_name, column_name, data_type, column_description"
        )
    schema = {}
    for table_name in schema_df["table_name"].unique():
        schema[table_name] = schema_df[schema_df["table_name"] == table_name][
            ["column_name", "data_type", "column_description"]
        ].to_dict(orient="records")

    payload = {
        "api_key": self.api_key,
        "table_metadata": schema,
        "db_type": self.db_type,
        "dev": dev,
        "temp": temp,
    }

    resp = await make_async_post_request(
        url=f"{self.base_url}/update_metadata", payload=payload
    )
    return resp


async def update_glossary(
    self,
    glossary: str = "",
    customized_glossary: dict = None,
    glossary_compulsory: str = "",
    glossary_prunable_units: List[str] = [],
    dev: bool = False,
):
    """
    Updates the glossary on the defog servers.
    :param glossary: The glossary to be used.
    """
    data = {
        "api_key": self.api_key,
        "glossary": glossary,
        "dev": dev,
        "glossary_compulsory": glossary_compulsory,
        "glossary_prunable_units": glossary_prunable_units,
    }
    if customized_glossary:
        data["customized_glossary"] = customized_glossary
    resp = await make_async_post_request(
        url=f"{self.base_url}/update_glossary", payload=data
    )
    return resp


async def delete_glossary(self, user_type=None, dev=False):
    """
    Deletes the glossary on the defog servers.
    """
    data = {
        "api_key": self.api_key,
        "dev": dev,
    }
    if user_type:
        data["key"] = user_type
    r = await make_async_post_request(
        url=f"{self.base_url}/delete_glossary",
        payload=data,
        return_response_object=True,
    )
    if r.status_code == 200:
        print("Glossary deleted successfully.")
    else:
        error_message = r.json().get("message", "")
        print(f"Glossary deletion failed.\nError message: {error_message}")


async def get_glossary(self, mode="general", dev=False):
    """
    Gets the glossary on the defog servers.
    """
    resp = await make_async_post_request(
        url=f"{self.base_url}/get_glossary",
        payload={"api_key": self.api_key, "dev": dev},
    )
    if mode == "general":
        return resp["glossary"]
    elif mode == "customized":
        return resp["customized_glossary"]


async def get_metadata(self, format="markdown", export_path=None, dev=False):
    """
    Gets the metadata on the defog servers.
    """
    resp = await make_async_post_request(
        url=f"{self.base_url}/get_metadata",
        payload={"api_key": self.api_key, "dev": dev},
    )
    items = []
    for table in resp["table_metadata"]:
        for item in resp["table_metadata"][table]:
            item["table_name"] = table
            items.append(item)
    if format == "markdown":
        return pd.DataFrame(items)[
            ["table_name", "column_name", "data_type", "column_description"]
        ].to_markdown(index=False)
    elif format == "csv":
        if export_path is None:
            export_path = "metadata.csv"
        pd.DataFrame(items)[
            ["table_name", "column_name", "data_type", "column_description"]
        ].to_csv(export_path, index=False)
        print(f"Metadata exported to {export_path}")
        return True
    elif format == "json":
        return resp["table_metadata"]


async def get_feedback(self, n_rows: int = 50, start_from: int = 0):
    """
    Gets the feedback on the defog servers.
    """
    resp = await make_async_post_request(
        url=f"{self.base_url}/get_feedback", payload={"api_key": self.api_key}
    )
    df = pd.DataFrame(resp["data"], columns=resp["columns"])
    df["created_at"] = df["created_at"].apply(lambda x: x[:10])
    for col in ["query_generated", "feedback_text"]:
        df[col] = df[col].fillna("")
        df[col] = df[col].apply(lambda x: x.replace("\n", "\\n"))
    return df.iloc[start_from:].head(n_rows).to_markdown(index=False)


async def get_quota(self) -> Optional[Dict]:
    """
    Get the quota usage for the API key.
    """
    api_key = self.api_key
    r = await make_async_post_request(
        url=f"{self.base_url}/check_api_usage",
        payload={"api_key": api_key},
        return_response_object=True,
    )
    # get status code and return None if not 200
    if r.status_code != 200:
        return None
    return r.json()


async def update_golden_queries(
    self,
    golden_queries: List[Dict] = None,
    golden_queries_path: str = None,
    scrub: bool = True,
    dev: bool = False,
):
    """
    Updates the golden queries on the defog servers.
    :param golden_queries: The golden queries to be used.
    :param golden_queries_path: The path to the golden queries CSV.
    :param scrub: Whether to scrub the golden queries.
    """
    if golden_queries is None and golden_queries_path is None:
        raise ValueError("Please provide either golden_queries or golden_queries_path.")

    if golden_queries is None:
        golden_queries = (
            pd.read_csv(golden_queries_path).fillna("").to_dict(orient="records")
        )

    resp = await make_async_post_request(
        url=f"{self.base_url}/update_golden_queries",
        payload={
            "api_key": self.api_key,
            "golden_queries": golden_queries,
            "scrub": scrub,
            "dev": dev,
        },
    )
    print(
        "Golden queries have been received by the system, and will be processed shortly..."
    )
    print(
        "Once that is done, you should be able to see improved results for your questions."
    )
    return resp


async def delete_golden_queries(
    self,
    golden_queries: dict = None,
    golden_queries_path: str = None,
    all: bool = False,
    dev: bool = False,
):
    """
    Updates the golden queries on the defog servers.
    :param golden_queries: The golden queries to be used.
    :param golden_queries_path: The path to the golden queries CSV.
    :param scrub: Whether to scrub the golden queries.
    """
    if golden_queries is None and golden_queries_path is None and not all:
        raise ValueError(
            "Please provide either golden_queries or golden_queries_path, or set all=True."
        )

    if all:
        resp = await make_async_post_request(
            url=f"{self.base_url}/delete_golden_queries",
            payload={"api_key": self.api_key, "all": True, "dev": dev},
        )
        print("All golden queries have now been deleted.")
    else:
        if golden_queries is None:
            golden_queries = (
                pd.read_csv(golden_queries_path).fillna("").to_dict(orient="records")
            )
        resp = await make_async_post_request(
            url=f"{self.base_url}/update_golden_queries",
            payload={"api_key": self.api_key, "golden_queries": golden_queries},
        )
    return resp


async def get_golden_queries(
    self, format: str = "csv", export_path: str = None, dev: bool = False
):
    """
    Gets the golden queries on the defog servers.
    """
    resp = await make_async_post_request(
        url=f"{self.base_url}/get_golden_queries",
        payload={"api_key": self.api_key, "dev": dev},
    )
    golden_queries = resp["golden_queries"]
    if format == "csv":
        if export_path is None:
            export_path = "golden_queries.csv"
        pd.DataFrame(golden_queries).to_csv(export_path, index=False)
        print(f"{len(golden_queries)} golden queries exported to {export_path}")
        return golden_queries
    elif format == "json":
        if export_path is None:
            export_path = "golden_queries.json"
        # Writing JSON asynchronously
        async with open(export_path, "w") as f:
            await f.write(json.dumps(resp, indent=4))
        print(f"{len(golden_queries)} golden queries exported to {export_path}")
        return golden_queries
    else:
        raise ValueError("format must be either 'csv' or 'json'.")


async def create_table_ddl(
    table_name: str, columns: List[Dict[str, str]], add_exists=True
) -> str:
    """
    Return a DDL statement for creating a table from a list of columns
    `columns` is a list of dictionaries with the following keys:
    - column_name: str
    - data_type: str
    - column_description: str
    """
    md_create = ""
    if add_exists:
        md_create += f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    else:
        md_create += f"CREATE TABLE {table_name} (\n"
    for i, column in enumerate(columns):
        col_name = column["column_name"]
        # if column name has spaces and hasn't been wrapped in double quotes, wrap it in double quotes
        if " " in col_name and not col_name.startswith('"'):
            col_name = f'"{col_name}"'
        dtype = column["data_type"]
        if i < len(columns) - 1:
            md_create += f"  {col_name} {dtype},\n"
        else:
            # avoid the trailing comma for the last line
            md_create += f"  {col_name} {dtype}\n"
    md_create += ");\n"
    return md_create


async def create_ddl_from_metadata(
    metadata: Dict[str, List[Dict[str, str]]], add_exists=True
) -> str:
    """
    Return a DDL statement for creating tables from metadata
    `metadata` is a dictionary with table names as keys and lists of dictionaries as values.
    Each dictionary in the list has the following keys:
    - column_name: str
    - data_type: str
    - column_description: str
    """
    md_create = ""
    for table_name, columns in metadata.items():
        if "." in table_name:
            # table_name = table_name.split(".", 1)[1]
            schema_name = table_name.split(".")[0]

            md_create += f"CREATE SCHEMA IF NOT EXISTS {schema_name};\n"
        md_create += create_table_ddl(table_name, columns, add_exists=add_exists)
    return md_create


async def create_empty_tables(self, dev: bool = False):
    """
    Create empty tables based on metadata
    """
    metadata = self.get_metadata(format="json", dev=dev)
    if self.db_type == "sqlserver":
        ddl = create_ddl_from_metadata(metadata, add_exists=False)
    else:
        ddl = create_ddl_from_metadata(metadata)

    try:
        if self.db_type == "postgres" or self.db_type == "redshift":
            import asyncpg

            conn = await asyncpg.connect(**self.db_creds)
            await conn.execute(ddl)
            await conn.close()
            return True

        elif self.db_type == "mysql":
            import aiomysql

            db_creds = self.db_creds.copy()
            db_creds["db"] = db_creds["database"]
            del db_creds["database"]
            conn = await aiomysql.connect(**db_creds)
            async with conn.cursor() as cur:
                for statement in ddl.split(";"):
                    await cur.execute(statement)
            await conn.commit()
            await conn.ensure_closed()
            return True

        elif self.db_type == "databricks":
            from databricks import sql

            conn = await asyncio.to_thread(sql.connect, **self.db_creds)
            await asyncio.to_thread(conn.execute, ddl)
            await asyncio.to_thread(conn.commit)
            await asyncio.to_thread(conn.close)
            return True

        elif self.db_type == "snowflake":
            import snowflake.connector

            conn = snowflake.connector.connect(
                user=self.db_creds["user"],
                password=self.db_creds["password"],
                account=self.db_creds["account"],
            )
            cur = conn.cursor()
            cur.execute(
                f"USE WAREHOUSE {self.db_creds['warehouse']}"
            )  # set the warehouse
            for statement in ddl.split(";"):
                cur.execute_async(statement)
                query_id = cur.sfqid
                while conn.is_still_running(conn.get_query_status(query_id)):
                    await asyncio.sleep(1)
            cur.close()
            conn.close()
            return True

        elif self.db_type == "bigquery":
            from google.cloud import bigquery

            client = await asyncio.to_thread(
                bigquery.Client.from_service_account_json,
                self.db_creds["json_key_path"],
            )
            for statement in ddl.split(";"):
                await asyncio.to_thread(client.query, statement)
            return True

        elif self.db_type == "sqlserver":
            import aioodbc

            if self.db_creds["database"] != "":
                connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};DATABASE={self.db_creds['database']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
            else:
                connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
            conn = await aioodbc.connect(dsn=connection_string)
            cur = await conn.cursor()
            for statement in ddl.split(";"):
                await cur.execute(statement)
            await conn.commit()
            await cur.close()
            await conn.close()
            return True

        else:
            raise ValueError(f"Unsupported DB type: {self.db_type}")
    except Exception as e:
        print(f"Error: {e}")
        return False
