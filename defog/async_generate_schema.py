from defog.util import (
    async_identify_categorical_columns,
    identify_categorical_columns,
)
from defog.local_storage import LocalStorage
import asyncio
import pandas as pd
import json
import warnings
from typing import List, Dict, Union


async def generate_postgres_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
    schemas: List[str] = ["public"],
) -> Union[Dict, List, str]:
    """
    Returns the schema of the tables in the database. Keys: column_name, data_type, column_description, custom_type_labels
    If tables is non-empty, we only generate the schema for the mentioned tables in the list.
    If schemas is non-empty, we only generate the schema for the mentioned schemas in the list.
    If return_tables_only is True, we return only the table names as a list.
    If upload is True, we save the schema to local storage and return the file path.
    If upload is False, we return the schema as a dict.
    If scan is True, we also scan the tables for categorical columns to enhance the column description.
    """
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )
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
                SELECT 
                    CAST(column_name AS TEXT), 
                    CAST(
                        CASE 
                            WHEN data_type = 'USER-DEFINED' THEN udt_name
                            ELSE data_type 
                        END AS TEXT
                    ) AS data_type,
                    col_description(
                        (quote_ident($2) || '.' || quote_ident($1))::regclass::oid, 
                        ordinal_position
                    ) AS column_description,
                    CASE
                        WHEN data_type = 'USER-DEFINED' THEN (
                            SELECT string_agg(enumlabel, ', ')
                            FROM pg_enum
                            WHERE enumtypid = (
                                SELECT oid
                                FROM pg_type
                                WHERE typname = udt_name
                            )
                        )
                        ELSE NULL
                    END AS custom_type_labels
                FROM information_schema.columns
                WHERE table_name = $1 AND table_schema = $2;
            """
            print(f"Schema for {schema}.{table_name}")
            rows = await conn.fetch(query, table_name, schema)
            rows = [row for row in rows]
            rows = [
                {
                    "column_name": row[0],
                    "data_type": row[1],
                    "column_description": row[2] or "",
                    "custom_type_labels": row[3].split(", ") if row[3] else [],
                }
                for row in rows
            ]
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

    if upload:
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in table_columns.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                        "custom_type_labels": (
                            ", ".join(col.get("custom_type_labels", []))
                            if col.get("custom_type_labels")
                            else ""
                        ),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "postgres"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None), getattr(self, "db_type", "postgres")
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
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
    # when upload is True, we save the schema to local storage and return the file path
    # when its false, we return the schema as a dict
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "redshift"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None), getattr(self, "db_type", "redshift")
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
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
    except ImportError:
        raise Exception("aiomysql not installed.")

    db_creds = self.db_creds.copy()
    db_creds["db"] = db_creds["database"]
    del db_creds["database"]
    conn = await aiomysql.connect(**db_creds)
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
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "mysql"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None), getattr(self, "db_type", "mysql")
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
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
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        from databricks import sql
    except ImportError:
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
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "databricks"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None),
                    getattr(self, "db_type", "databricks"),
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
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
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        import snowflake.connector
    except ImportError:
        raise Exception("snowflake-connector not installed.")

    conn = snowflake.connector.connect(
        user=self.db_creds["user"],
        password=self.db_creds["password"],
        account=self.db_creds["account"],
    )

    conn.cursor().execute(
        f"USE WAREHOUSE {self.db_creds['warehouse']}"
    )  # set the warehouse

    schemas = {}
    alt_types = {"DATE": "TIMESTAMP", "TEXT": "VARCHAR", "FIXED": "NUMERIC"}
    print("Getting schema for each table that you selected...")
    # get the schema for each table
    if len(tables) == 0:
        cur = conn.cursor()
        # get all tables from Snowflake database
        cur.execute_async("SHOW TERSE TABLES;")  # execute asynchrnously
        query_id = cur.sfqid  # get the query id to check the status
        while conn.is_still_running(conn.get_query_status(query_id)):
            await asyncio.sleep(1)
        res = cur.fetchall()
        tables = [f"{row[3]}.{row[4]}.{row[1]}" for row in res]

    if return_tables_only:
        conn.close()
        return tables

    for table_name in tables:
        rows = []
        cur = conn.cursor()
        cur.execute_async(f"SHOW COLUMNS IN {table_name};")
        query_id = cur.sfqid
        while conn.is_still_running(conn.get_query_status(query_id)):
            await asyncio.sleep(1)
        rows = cur.fetchall()
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
        cur = conn.cursor()
        if scan:
            rows = await async_identify_categorical_columns(
                conn=conn,
                cur=cur,
                table_name=table_name,
                rows=rows,
                is_cursor_async=False,
                db_type="snowflake",
            )
        cur.close()
        if len(rows) > 0:
            schemas[table_name] = rows

    conn.close()

    if upload:
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "snowflake"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None),
                    getattr(self, "db_type", "snowflake"),
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
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
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        from google.cloud import bigquery
    except ImportError:
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
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "bigquery"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None), getattr(self, "db_type", "bigquery")
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
    else:
        return schemas


async def generate_sqlserver_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    return_tables_only: bool = False,
) -> str:
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        import aioodbc
    except Exception:
        raise Exception(
            "aioodbc not installed. Please install it with `pip install aioodbc`."
        )
    if self.db_creds["database"] != "":
        connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};DATABASE={self.db_creds['database']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
    else:
        connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"

    conn = await aioodbc.connect(dsn=connection_string)
    cur = await conn.cursor()
    schemas = {}
    schema = self.db_creds.get("schema", "dbo")

    if len(tables) == 0:
        # get all tables
        await cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
            (schema,),
        )
        if schema == "dbo":
            tables = [row[0] for row in await cur.fetchall()]
        else:
            tables = [schema + "." + row[0] for row in await cur.fetchall()]

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
        if len(rows) > 0:
            schemas[table_name] = rows
    await conn.close()
    if upload:
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "sqlserver"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None),
                    getattr(self, "db_type", "sqlserver"),
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
    else:
        return schemas


async def generate_sqlite_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    """
    Generate schema for SQLite database (async version).

    Example:
        # File database
        defog = Defog(db_type="sqlite", db_creds={"database": "/path/to/database.db"})
        schema = await defog.async_generate_db_schema([], upload=False)

        # Memory database
        defog = Defog(db_type="sqlite", db_creds={"database": ":memory:"})
        schema = await defog.async_generate_db_schema([], upload=False)
    """
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )
    try:
        import aiosqlite
    except ImportError as e:
        raise ImportError(
            "aiosqlite module not available. Please install with 'pip install aiosqlite' for async SQLite support."
        ) from e

    database_path = self.db_creds.get("database", ":memory:")
    async with aiosqlite.connect(database_path) as conn:
        schemas = {}

        if len(tables) == 0:
            # get all tables
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ) as cur:
                tables = [row[0] async for row in cur]

        if return_tables_only:
            return tables

        print("Getting schema for each table that you selected...")
        # get the schema for each table
        for table_name in tables:
            async with conn.execute(f"PRAGMA table_info({table_name});") as cur:
                rows = [
                    {"column_name": row[1], "data_type": row[2]} async for row in cur
                ]
            if scan:
                rows = await async_identify_categorical_columns(
                    conn=conn, table_name=table_name, rows=rows, db_type="sqlite"
                )
            if len(rows) > 0:
                schemas[table_name] = rows

    if upload:
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "sqlite"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None), getattr(self, "db_type", "sqlite")
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
    else:
        return schemas


async def generate_duckdb_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    """
    Generate schema for DuckDB database (async version).

    Example:
        # File database
        defog = AsyncDefog(db_type="duckdb", db_creds={"database": "/path/to/database.duckdb"})
        schema = await defog.async_generate_db_schema([], upload=False)

        # Memory database
        defog = AsyncDefog(db_type="duckdb", db_creds={"database": ":memory:"})
        schema = await defog.async_generate_db_schema([], upload=False)
    """
    # Issue deprecation warning for upload parameter
    if upload:
        warnings.warn(
            "The 'upload' parameter is deprecated and will be removed in a future version. "
            "Schema data is now saved locally by default.",
            DeprecationWarning,
            stacklevel=2,
        )
    try:
        import duckdb
    except ImportError as e:
        raise ImportError(
            "duckdb not installed. Please install it with `pip install duckdb`."
        ) from e

    database_path = self.db_creds.get("database", ":memory:")

    # DuckDB doesn't have native async support, so we use asyncio.to_thread
    def _get_schema():
        nonlocal tables  # Make tables accessible from outer scope
        conn = duckdb.connect(database_path, read_only=True)
        schemas = {}

        try:
            if len(tables) == 0:
                # Get all tables (DuckDB supports schemas, so we need to check all schemas)
                tables_query = """
                SELECT table_schema || '.' || table_name as full_table_name
                FROM information_schema.tables 
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN ('information_schema', 'pg_catalog')
                """
                tables_result = conn.execute(tables_query).fetchall()
                all_tables = [row[0] for row in tables_result]

                # For tables in the main schema, also add without schema prefix
                main_tables = conn.execute(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_type = 'BASE TABLE' 
                    AND table_schema = 'main'
                """
                ).fetchall()
                all_tables.extend([row[0] for row in main_tables])

                if return_tables_only:
                    return all_tables
                tables = all_tables

            if return_tables_only:
                return tables

            # Get the schema for each table
            for table_name in tables:
                # Handle both schema.table and table formats
                if "." in table_name:
                    schema_name, table_only = table_name.split(".", 1)
                    columns_query = f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = '{schema_name}' 
                    AND table_name = '{table_only}'
                    ORDER BY ordinal_position
                    """
                else:
                    columns_query = f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    AND table_schema = 'main'
                    ORDER BY ordinal_position
                    """

                rows = conn.execute(columns_query).fetchall()
                rows = [{"column_name": row[0], "data_type": row[1]} for row in rows]

                if scan:
                    # Create a cursor-like object for DuckDB compatibility
                    class DuckDBCursor:
                        def __init__(self, connection):
                            self.connection = connection

                        def execute(self, query):
                            self.result = self.connection.execute(query)

                        def fetchall(self):
                            return self.result.fetchall()

                    cursor = DuckDBCursor(conn)
                    rows = identify_categorical_columns(cursor, table_name, rows)

                if len(rows) > 0:
                    schemas[table_name] = rows

            return schemas
        finally:
            conn.close()

    # Run synchronous code in thread
    result = await asyncio.to_thread(_get_schema)

    if return_tables_only:
        return result

    schemas = result

    if upload:
        print("Processing schema data...")

        # Flatten the nested structure for CSV
        flattened_data = []
        for table_name, columns in schemas.items():
            for col in columns:
                flattened_data.append(
                    {
                        "table_name": table_name,
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "column_description": col.get("column_description", ""),
                    }
                )

        df = pd.DataFrame(flattened_data)

        # Save using LocalStorage with asyncio.to_thread
        storage = LocalStorage()

        if return_format == "csv":
            # Save as CSV
            csv_data = df.to_csv(index=False)
            result = await asyncio.to_thread(
                storage.save_schema,
                csv_data,
                "defog_metadata.csv",
                api_key=getattr(self, "api_key", None),
                db_type=getattr(self, "db_type", "duckdb"),
            )
            return str(
                storage.storage_dir
                / "schemas"
                / storage._get_project_id(
                    getattr(self, "api_key", None), getattr(self, "db_type", "duckdb")
                )
                / "defog_metadata.csv"
            )
        else:
            # Return CSV string
            return df.to_csv(index=False)
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
    elif self.db_type == "sqlite":
        return await self.generate_sqlite_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "duckdb":
        return await self.generate_duckdb_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    else:
        raise ValueError(
            f"Creation of a DB schema for {self.db_type} is not yet supported via the library. If you are a premium user, please contact us at founder@defog.ai so we can manually add it."
        )
