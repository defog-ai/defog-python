from defog.util import async_identify_categorical_columns, make_async_post_request
import asyncio
from io import StringIO
import pandas as pd
import json
from typing import List


async def generate_postgres_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
    schemas: List[str] = ["public"],
) -> str:
    # when upload is True, we send the schema to the defog servers and generate a CSV
    # when its false, we return the schema as a dict
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg not installed. Please install it with `pip install psycopg2-binary`."
        )

    conn = await asyncpg.connect(**self.db_creds)
    schemas = tuple(schemas)

    if len(tables) == 0:
        # get all tables
        for schema in schemas:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = $1;
            """
            rows = await conn.fetch(query, schema)
            if schema == "public":
                tables += [row[0] for row in rows]
            else:
                tables += [schema + "." + row[0] for row in rows]

    if return_tables_only:
        await conn.close()
        return tables

    print("Getting schema for each table that you selected...")

    table_columns = {}

    # get the columns for each table
    for schema in schemas:
        for table_name in tables:
            if "." in table_name:
                _, table_name = table_name.split(".", 1)
            query = """
                SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT)
                FROM information_schema.columns
                WHERE table_name = $1 AND table_schema = $2;
            """
            rows = await conn.fetch(query, table_name, schema)
            rows = [row for row in rows]
            rows = [{"column_name": row[0], "data_type": row[1]} for row in rows]
            if len(rows) > 0:
                if scan:
                    rows = await async_identify_categorical_columns(
                        conn=conn, cur=None, table_name=table_name, rows=rows
                    )
                if schema == "public":
                    table_columns[table_name] = rows
                else:
                    table_columns[schema + "." + table_name] = rows
    await conn.close()

    print(
        "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
    )
    if upload:
        # send the schemas dict to the defog servers
        resp = await make_async_post_request(
            url=f"{self.base_url}/get_schema_csv",
            payload={
                "api_key": self.api_key,
                "schemas": table_columns,
            },
        )
        if "csv" in resp:
            csv = resp["csv"]
            if return_format == "csv":
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
            else:
                return csv
        else:
            print(f"We got an error!")
            if "message" in resp:
                print(f"Error message: {resp['message']}")
            print(
                f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
            )
    else:
        return table_columns


async def generate_redshift_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    # when upload is True, we send the schema to the defog servers and generate a CSV
    # when its false, we return the schema as a dict
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg not installed. Please install it with `pip install psycopg2-binary`."
        )

    if "schema" not in self.db_creds:
        schema = "public"
        conn = await asyncpg.connect(**self.db_creds)
    else:
        schema = self.db_creds["schema"]
        del self.db_creds["schema"]
        conn = await asyncpg.connect(**self.db_creds)

    schemas = {}

    if len(tables) == 0:
        table_names_query = (
            "SELECT table_name FROM information_schema.tables WHERE table_schema = $1;"
        )
        results = await conn.fetch(table_names_query, schema)
        tables = [row[0] for row in results]

    if return_tables_only:
        await conn.close()
        return tables

    print("Getting schema for each table that you selected...")
    # get the schema for each table
    for table_name in tables:
        table_schema_query = "SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = $1 AND table_schema= $2;"
        rows = await conn.fetch(table_schema_query, table_name, schema)
        rows = [row for row in rows]
        rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
        if len(rows) > 0:
            if scan:
                await conn.execute(f"SET search_path TO {schema}")
                rows = await async_identify_categorical_columns(
                    conn=conn, cur=None, table_name=table_name, rows=rows
                )

            schemas[table_name] = rows

    await conn.close()

    if upload:
        print(
            "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        # send the schemas dict to the defog servers
        resp = await make_async_post_request(
            url=f"{self.base_url}/get_schema_csv",
            payload={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
        )
        if "csv" in resp:
            csv = resp["csv"]
            if return_format == "csv":
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
            else:
                return csv
        else:
            print(f"We got an error!")
            if "message" in resp:
                print(f"Error message: {resp['message']}")
            print(
                f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
            )
    else:
        return schemas


async def generate_mysql_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    try:
        import aiomysql
    except:
        raise Exception("aiomysql not installed.")

    conn = await aiomysql.connect(**self.db_creds)
    cur = await conn.cursor()
    schemas = {}

    if len(tables) == 0:
        # get all tables
        db_name = self.db_creds.get("database", "")
        await cur.execute(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{db_name}';"
        )
        tables = [row[0] for row in await cur.fetchall()]

    if return_tables_only:
        await conn.close()
        return tables

    print("Getting schema for the relevant table in your database...")
    # get the schema for each table
    for table_name in tables:
        await cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;",
            (table_name,),
        )
        rows = await cur.fetchall()
        rows = [row for row in rows]
        rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
        if scan:
            rows = await async_identify_categorical_columns(
                conn=None,
                cur=cur,
                table_name=table_name,
                rows=rows,
                is_cursor_async=True,
            )
        if len(rows) > 0:
            schemas[table_name] = rows

    await conn.ensure_closed()

    if upload:
        resp = await make_async_post_request(
            url=f"{self.base_url}/get_schema_csv",
            payload={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
        )
        if "csv" in resp:
            csv = resp["csv"]
            if return_format == "csv":
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
            else:
                return csv
        else:
            print(f"We got an error!")
            if "message" in resp:
                print(f"Error message: {resp['message']}")
            print(
                f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
            )
    else:
        return schemas


async def generate_databricks_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    try:
        from databricks import sql
    except:
        raise Exception("databricks-sql-connector not installed.")

    conn = await asyncio.to_thread(sql.connect, **self.db_creds)
    schemas = {}
    async with await asyncio.to_thread(conn.cursor) as cur:
        print("Getting schema for each table that you selected...")
        # get the schema for each table

        if len(tables) == 0:
            # get all tables from databricks
            await asyncio.to_thread(
                cur.tables, schema_name=self.db_creds.get("schema", "default")
            )
            tables = [row.TABLE_NAME for row in await asyncio.to_thread(cur.fetchall())]

        if return_tables_only:
            await asyncio.to_thread(conn.close)
            return tables

        for table_name in tables:
            await asyncio.to_thread(
                cur.columns,
                schema_name=self.db_creds.get("schema", "default"),
                table_name=table_name,
            )
            rows = await asyncio.to_thread(cur.fetchall)
            rows = [row for row in rows]
            rows = [
                {"column_name": i.COLUMN_NAME, "data_type": i.TYPE_NAME} for i in rows
            ]
            if scan:
                rows = await async_identify_categorical_columns(
                    conn=None,
                    cur=cur,
                    table_name=table_name,
                    rows=rows,
                    is_cursor_async=False,
                )
            if len(rows) > 0:
                schemas[table_name] = rows

    await asyncio.to_thread(conn.close)

    if upload:
        resp = await make_async_post_request(
            url=f"{self.base_url}/get_schema_csv",
            payload={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
        )
        if "csv" in resp:
            csv = resp["csv"]
            if return_format == "csv":
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
            else:
                return csv
        else:
            print(f"We got an error!")
            if "message" in resp:
                print(f"Error message: {resp['message']}")
            print(
                f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
            )
    else:
        return schemas


async def generate_snowflake_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    try:
        import snowflake.connector
    except:
        raise Exception("snowflake-connector not installed.")

    conn = await asyncio.to_thread(
        snowflake.connector.connect,
        user=self.db_creds["user"],
        password=self.db_creds["password"],
        account=self.db_creds["account"],
    )

    await asyncio.to_thread(
        conn.cursor().execute, f"USE WAREHOUSE {self.db_creds['warehouse']}"
    )  # set the warehouse

    schemas = {}
    alt_types = {"DATE": "TIMESTAMP", "TEXT": "VARCHAR", "FIXED": "NUMERIC"}
    print("Getting schema for each table that you selected...")
    # get the schema for each table
    if len(tables) == 0:
        # get all tables from Snowflake database
        cur = await asyncio.to_thread(conn.cursor().execute, "SHOW TERSE TABLES;")
        res = await asyncio.to_thread(cur.fetchall)
        tables = [f"{row[3]}.{row[4]}.{row[1]}" for row in res]

    if return_tables_only:
        await asyncio.to_thread(conn.close)
        return tables

    for table_name in tables:
        rows = []
        cur = await asyncio.to_thread(conn.cursor)
        fetched_rows = await asyncio.to_thread(
            cur.execute, f"SHOW COLUMNS IN {table_name};"
        )
        for row in fetched_rows:
            rows.append(row)
        rows = [
            {
                "column_name": i[2],
                "data_type": json.loads(i[3])["type"],
                "column_description": i[8],
            }
            for i in rows
        ]
        for idx, row in enumerate(rows):
            if row["data_type"] in alt_types:
                row["data_type"] = alt_types[row["data_type"]]
            rows[idx] = row

        cur = await asyncio.to_thread(conn.cursor)
        if scan:
            rows = await async_identify_categorical_columns(
                conn=None,
                cur=cur,
                table_name=table_name,
                rows=rows,
                is_cursor_async=False,
            )
        await asyncio.to_thread(cur.close)
        if len(rows) > 0:
            schemas[table_name] = rows

    await asyncio.to_thread(conn.close)

    if upload:
        print(
            "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        resp = await make_async_post_request(
            url=f"{self.base_url}/get_schema_csv",
            payload={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
        )
        if "csv" in resp:
            csv = resp["csv"]
            if return_format == "csv":
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
            else:
                return csv
        else:
            print(f"We got an error!")
            if "message" in resp:
                print(f"Error message: {resp['message']}")
            print(
                f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
            )
    else:
        return schemas


async def generate_bigquery_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    try:
        from google.cloud import bigquery
    except:
        raise Exception("google-cloud-bigquery not installed.")

    client = await asyncio.to_thread(
        bigquery.Client.from_service_account_json, self.db_creds["json_key_path"]
    )
    project_id = [p.project_id for p in await asyncio.to_thread(client.list_projects)][
        0
    ]
    datasets = [
        dataset.dataset_id for dataset in await asyncio.to_thread(client.list_datasets)
    ]
    schemas = {}

    if len(tables) == 0:
        # get all tables
        tables = []
        for dataset in datasets:
            table_list = await asyncio.to_thread(client.list_tables, dataset)
            tables += [
                f"{project_id}.{dataset}.{table.table_id}" for table in table_list
            ]

    print("Getting the schema for each table that you selected...")
    # get the schema for each table
    for table_name in tables:
        table = await asyncio.to_thread(client.get_table, table_name)
        rows = table.schema
        rows = [{"column_name": i.name, "data_type": i.field_type} for i in rows]
        if len(rows) > 0:
            schemas[table_name] = rows

    await asyncio.to_thread(client.close)

    if upload:
        print(
            "Sending the schema to Defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        resp = await make_async_post_request(
            url=f"{self.base_url}/get_schema_csv",
            payload={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
        )
        if "csv" in resp:
            csv = resp["csv"]
            if return_format == "csv":
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
            else:
                return csv
        else:
            print(f"We got an error!")
            if "message" in resp:
                print(f"Error message: {resp['message']}")
            print(
                f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
            )
    else:
        return schemas


async def generate_sqlserver_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    return_tables_only: bool = False,
) -> str:
    try:
        import pyodbc
    except:
        raise Exception("pyodbc not installed.")

    if self.db_creds["database"] != "":
        connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};DATABASE={self.db_creds['database']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
    else:
        connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
    conn = await asyncio.to_thread(pyodbc.connect, connection_string)
    cur = await asyncio.to_thread(conn.cursor)
    schemas = {}
    schema = self.db_creds.get("schema", "dbo")

    if len(tables) == 0:
        table_names_query = (
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;"
        )
        await asyncio.to_thread(cur.execute, table_names_query, (schema,))
        if schema == "dbo":
            tables += [row[0] for row in await asyncio.to_thread(cur.fetchall)]
        else:
            tables += [
                schema + "." + row[0] for row in await asyncio.to_thread(cur.fetchall)
            ]

    if return_tables_only:
        await asyncio.to_thread(conn.close)
        return tables

    print("Getting schema for each table in your database...")
    # get the schema for each table
    for table_name in tables:
        await asyncio.to_thread(
            cur.execute,
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';",
        )
        rows = await asyncio.to_thread(cur.fetchall)
        rows = [row for row in rows]
        rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
        if len(rows) > 0:
            schemas[table_name] = rows

    await asyncio.to_thread(conn.close)
    if upload:
        print(
            "Sending the schema to Defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        resp = await make_async_post_request(
            url=f"{self.base_url}/get_schema_csv",
            payload={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
        )
        if "csv" in resp:
            csv = resp["csv"]
            if return_format == "csv":
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
            else:
                return csv
        else:
            print(f"We got an error!")
            if "message" in resp:
                print(f"Error message: {resp['message']}")
            print(
                f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
            )
    else:
        return schemas


async def generate_db_schema(
    self,
    tables: list,
    scan: bool = True,
    upload: bool = True,
    return_tables_only: bool = False,
    return_format: str = "csv",
) -> str:
    if self.db_type == "postgres":
        return await self.generate_postgres_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "mysql":
        return await self.generate_mysql_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "bigquery":
        return await self.generate_bigquery_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "redshift":
        return await self.generate_redshift_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "snowflake":
        return await self.generate_snowflake_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "databricks":
        return await self.generate_databricks_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "sqlserver":
        return await self.generate_sqlserver_schema(
            tables,
            return_format=return_format,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    else:
        raise ValueError(
            f"Creation of a DB schema for {self.db_type} is not yet supported via the library. If you are a premium user, please contact us at founder@defog.ai so we can manually add it."
        )
