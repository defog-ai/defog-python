import requests
from defog.util import identify_categorical_columns
from io import StringIO
import pandas as pd
import json
from typing import List, Dict, Union


def generate_postgres_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
    schemas: List[str] = [],
) -> Union[Dict, List, str]:
    """
    Returns the schema of the tables in the database. Keys: column_name, data_type, column_description, custom_type_labels
    If tables is non-empty, we only generate the schema for the mentioned tables in the list.
    If schemas is non-empty, we only generate the schema for the mentioned schemas in the list.
    If return_tables_only is True, we return only the table names as a list.
    If upload is True, we send the schema to the defog servers and generate a CSV.
    If upload is False, we return the schema as a dict.
    If scan is True, we also scan the tables for categorical columns to enhance the column description.
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
        )

    conn = psycopg2.connect(**self.db_creds)
    cur = conn.cursor()

    if len(tables) == 0:
        # get all tables
        if len(schemas) > 0:
            for schema in schemas:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                    (schema,),
                )
                if schema == "public":
                    tables += [row[0] for row in cur.fetchall()]
                else:
                    tables += [schema + "." + row[0] for row in cur.fetchall()]
        else:
            excluded_schemas = (
                "information_schema",
                "pg_catalog",
                "pg_toast",
                "pg_temp_1",
                "pg_toast_temp_1",
            )
            cur.execute(
                "SELECT table_name, table_schema FROM information_schema.tables WHERE table_schema NOT IN %s;",
                (excluded_schemas,),
            )
            for row in cur.fetchall():
                if row[1] == "public":
                    tables.append(row[0])
                else:
                    tables.append(f"{row[1]}.{row[0]}")

    if return_tables_only:
        conn.close()
        return tables

    print("Getting schema for each table that you selected...")

    table_columns = {}

    # get the columns for each table
    for table_name in tables:
        if "." in table_name:
            schema, table_name = table_name.split(".", 1)
        else:
            schema = "public"
        cur.execute(
            """
            SELECT 
                CAST(column_name AS TEXT), 
                CAST(
                    CASE 
                        WHEN data_type = 'USER-DEFINED' THEN udt_name
                        ELSE data_type 
                    END AS TEXT
                ) AS type,
                col_description(
                    FORMAT('%%s.%%s', table_schema, table_name)::regclass::oid, 
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
            WHERE table_name::text = %s 
            AND table_schema = %s;
            """,
            (
                table_name,
                schema,
            ),
        )
        rows = cur.fetchall()
        rows = [row for row in rows]
        rows = [
            {
                "column_name": i[0],
                "data_type": i[1],
                "column_description": i[2]
                or ""
                + (
                    " This is an enum column with the possible values: " + i[3]
                    if i[3]
                    else ""
                ),
                "custom_type_labels": i[3].split(", ") if i[3] else [],
            }
            for i in rows
        ]
        if len(rows) > 0:
            if scan:
                rows = identify_categorical_columns(cur, table_name, rows)
            if schema == "public":
                table_columns[table_name] = rows
            else:
                table_columns[schema + "." + table_name] = rows
    conn.close()

    print(
        "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
    )
    if upload:
        # send the schemas dict to the defog servers
        r = requests.post(
            f"{self.base_url}/get_schema_csv",
            json={
                "api_key": self.api_key,
                "schemas": table_columns,
            },
            verify=False,
        )
        resp = r.json()
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


def get_postgres_functions(
    self, schemas: List[str] = []
) -> Dict[str, List[Dict[str, str]]]:
    """
    Returns the custom functions and their definitions of the mentioned schemas in the database.
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
        )

    conn = psycopg2.connect(**self.db_creds)
    cur = conn.cursor()
    functions = {}

    if len(schemas) == 0:
        schemas = ["public"]

    for schema in schemas:
        cur.execute(
            """
            SELECT
                CAST(p.proname AS TEXT) AS function_name,
                pg_get_functiondef(p.oid) AS function_definition
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %s;
            """,
            (schema,),
        )
        rows = [
            {"function_name": row[0], "function_definition": row[1]}
            for row in cur.fetchall()
            if row[1] is not None
        ]
        if rows:
            functions[schema] = rows
    conn.close()
    return functions


def get_postgres_triggers(
    self, schemas: List[str] = []
) -> Dict[str, List[Dict[str, str]]]:
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
        )

    conn = psycopg2.connect(**self.db_creds)
    cur = conn.cursor()
    triggers = {}

    if len(schemas) == 0:
        schemas = ["public"]

    for schema in schemas:
        cur.execute(
            """
            SELECT
                CAST(t.tgname AS TEXT) AS trigger_name,
                pg_get_triggerdef(t.oid) AS trigger_definition
            FROM pg_trigger t
            JOIN pg_class c ON t.tgrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s
            """,
            (schema,),
        )
        rows = [
            {"trigger_name": row[0], "trigger_definition": row[1]}
            for row in cur.fetchall()
            if row[1] is not None
        ]
        if rows:
            triggers[schema] = rows

    conn.close()
    return triggers


def generate_redshift_schema(
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
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
        )

    if "schema" not in self.db_creds:
        schema = "public"
        conn = psycopg2.connect(**self.db_creds)
    else:
        schema = self.db_creds["schema"]
        del self.db_creds["schema"]
        conn = psycopg2.connect(**self.db_creds)
    cur = conn.cursor()

    schemas = {}

    if len(tables) == 0:
        # get all tables
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
            (schema,),
        )
        tables = [row[0] for row in cur.fetchall()]

    if return_tables_only:
        conn.close()
        return tables

    print("Getting schema for each table that you selected...")
    # get the schema for each table
    for table_name in tables:
        cur.execute(
            "SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s AND table_schema= %s;",
            (
                table_name,
                schema,
            ),
        )
        rows = cur.fetchall()
        rows = [row for row in rows]
        rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
        if len(rows) > 0:
            if scan:
                cur.execute(f"SET search_path TO {schema}")
                rows = identify_categorical_columns(cur, table_name, rows)
                cur.close()
            schemas[table_name] = rows

    if upload:
        print(
            "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        # send the schemas dict to the defog servers
        r = requests.post(
            f"{self.base_url}/get_schema_csv",
            json={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
            verify=False,
        )
        resp = r.json()
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


def generate_mysql_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
) -> str:
    try:
        import mysql.connector
    except:
        raise Exception("mysql-connector not installed.")

    conn = mysql.connector.connect(**self.db_creds)
    cur = conn.cursor()
    schemas = {}

    if len(tables) == 0:
        # get all tables
        db_name = self.db_creds.get("database", "")
        cur.execute(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{db_name}';"
        )
        tables = [row[0] for row in cur.fetchall()]

    if return_tables_only:
        conn.close()
        return tables

    print("Getting schema for the relevant table in your database...")
    # get the schema for each table
    for table_name in tables:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;",
            (table_name,),
        )
        rows = cur.fetchall()
        rows = [row for row in rows]
        rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
        if scan:
            rows = identify_categorical_columns(cur, table_name, rows)
        if len(rows) > 0:
            schemas[table_name] = rows

    conn.close()

    if upload:
        r = requests.post(
            f"{self.base_url}/get_schema_csv",
            json={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
            verify=False,
        )
        resp = r.json()
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


def generate_databricks_schema(
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

    conn = sql.connect(**self.db_creds)
    schemas = {}
    with conn.cursor() as cur:
        print("Getting schema for each table that you selected...")
        # get the schema for each table

        if len(tables) == 0:
            # get all tables from databricks
            cur.tables(schema_name=self.db_creds.get("schema", "default"))
            tables = [row.TABLE_NAME for row in cur.fetchall()]

        if return_tables_only:
            conn.close()
            return tables

        for table_name in tables:
            cur.columns(
                schema_name=self.db_creds.get("schema", "default"),
                table_name=table_name,
            )
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [
                {"column_name": i.COLUMN_NAME, "data_type": i.TYPE_NAME} for i in rows
            ]
            if scan:
                rows = identify_categorical_columns(cur, table_name, rows)
            if len(rows) > 0:
                schemas[table_name] = rows

    conn.close()

    if upload:
        r = requests.post(
            f"{self.base_url}/get_schema_csv",
            json={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
            verify=False,
        )
        resp = r.json()
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


def generate_snowflake_schema(
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
        # get all tables from Snowflake database
        cur = conn.cursor().execute("SHOW TERSE TABLES;")
        res = cur.fetchall()
        tables = [f"{row[3]}.{row[4]}.{row[1]}" for row in res]

    if return_tables_only:
        conn.close()
        return tables

    for table_name in tables:
        rows = []
        for row in conn.cursor().execute(f"SHOW COLUMNS IN {table_name};"):
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
        cur = conn.cursor()
        if scan:
            rows = identify_categorical_columns(cur, table_name, rows)
        cur.close()
        if len(rows) > 0:
            schemas[table_name] = rows

    conn.close()

    if upload:
        print(
            "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        r = requests.post(
            f"{self.base_url}/get_schema_csv",
            json={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
            verify=False,
        )
        resp = r.json()
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


def generate_bigquery_schema(
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

    client = bigquery.Client.from_service_account_json(self.db_creds["json_key_path"])
    project_id = [p.project_id for p in client.list_projects()][0]
    datasets = [dataset.dataset_id for dataset in client.list_datasets()]
    schemas = {}

    if len(tables) == 0:
        # get all tables
        tables = []
        for dataset in datasets:
            tables += [
                f"{project_id}.{dataset}.{table.table_id}"
                for table in client.list_tables(dataset=dataset)
            ]

    if return_tables_only:
        return tables

    print("Getting the schema for each table that you selected...")
    # get the schema for each table
    for table_name in tables:
        table = client.get_table(table_name)
        rows = table.schema
        rows = [{"column_name": i.name, "data_type": i.field_type} for i in rows]
        if len(rows) > 0:
            schemas[table_name] = rows

    client.close()

    if upload:
        print(
            "Sending the schema to Defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        r = requests.post(
            f"{self.base_url}/get_schema_csv",
            json={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
            verify=False,
        )
        resp = r.json()
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


def generate_sqlserver_schema(
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

    if "database" in self.db_creds and self.db_creds["database"] not in ["", None]:
        connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};DATABASE={self.db_creds['database']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
    else:
        connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
    conn = pyodbc.connect(connection_string)

    schemas = {}
    cur = conn.cursor()

    SYSTEM_SCHEMAS = ["sys", "information_schema", "guest", "INFORMATION_SCHEMA"]
    DEFAULT_TABLES = [
        "spt_fallback_db",
        "spt_fallback_dev",
        "spt_fallback_usg",
        "spt_monitor",
        "spt_values",
        "MSreplication_options",
    ]

    if len(tables) == 0:
        # get all tables
        cur.execute(
            "SELECT table_catalog, table_schema, table_name FROM information_schema.tables;"
        )

        tables = []
        for row in cur.fetchall():
            if (
                row.table_schema not in SYSTEM_SCHEMAS
                and row.table_name not in DEFAULT_TABLES
            ):
                if not self.db_creds["database"]:
                    # if database is not specified, we return the table catalog + table schema + table name
                    tables.append(
                        row.table_catalog
                        + "."
                        + row.table_schema
                        + "."
                        + row.table_name
                    )
                else:
                    # if database is specified, we return only the table schema + table name
                    # if the table schema is dbo, we return only the table name
                    if row.table_schema == "dbo":
                        tables.append(row.table_name)
                    else:
                        tables.append(row.table_schema + "." + row.table_name)

    if return_tables_only:
        conn.close()
        return tables

    print("Getting schema for each table in your database...")
    # get the schema for each table
    for orig_table_name in tables:
        # if there are two dots, we have the database name and the schema name
        if orig_table_name.count(".") == 2:
            db_name, schema, table_name = orig_table_name.split(".", 2)
        elif orig_table_name.count(".") == 1:
            schema, table_name = orig_table_name.split(".", 1)
            db_name = self.db_creds["database"]
        else:
            schema = "dbo"
            db_name = self.db_creds["database"]
            table_name = orig_table_name
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? AND table_schema = ? AND table_catalog = ?;",
            table_name,
            schema,
            db_name,
        )
        rows = cur.fetchall()
        rows = [row for row in rows]
        rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
        if len(rows) > 0:
            schemas[orig_table_name] = rows

    conn.close()
    if upload:
        print(
            "Sending the schema to Defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        r = requests.post(
            f"{self.base_url}/get_schema_csv",
            json={
                "api_key": self.api_key,
                "schemas": schemas,
                "foreign_keys": [],
                "indexes": [],
            },
            verify=False,
        )
        resp = r.json()
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


def generate_db_schema(
    self,
    tables: list,
    scan: bool = True,
    upload: bool = True,
    return_tables_only: bool = False,
    return_format: str = "csv",
) -> str:
    if self.db_type == "postgres":
        return self.generate_postgres_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "mysql":
        return self.generate_mysql_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "bigquery":
        return self.generate_bigquery_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "redshift":
        return self.generate_redshift_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "snowflake":
        return self.generate_snowflake_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "databricks":
        return self.generate_databricks_schema(
            tables,
            return_format=return_format,
            scan=scan,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    elif self.db_type == "sqlserver":
        return self.generate_sqlserver_schema(
            tables,
            return_format=return_format,
            upload=upload,
            return_tables_only=return_tables_only,
        )
    else:
        raise ValueError(
            f"Creation of a DB schema for {self.db_type} is not yet supported via the library. If you are a premium user, please contact us at founder@defog.ai so we can manually add it."
        )
