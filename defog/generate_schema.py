import requests
from defog.util import identify_categorical_columns
from io import StringIO
import pandas as pd
import json


def generate_postgres_schema(
    self,
    tables: list,
    upload: bool = True,
    return_format: str = "csv",
    scan: bool = True,
    return_tables_only: bool = False,
    schemas: list[str] = ["public"],
) -> str:
    # when upload is True, we send the schema to the defog servers and generate a CSV
    # when its false, we return the schema as a dict
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
        )

    conn = psycopg2.connect(**self.db_creds)
    cur = conn.cursor()
    schemas = tuple(schemas)

    if len(tables) == 0:
        # get all tables
        for schema in schemas:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                (schema,),
            )
            if schema == "public":
                tables += [row[0] for row in cur.fetchall()]
            else:
                tables += [schema + "." + row[0] for row in cur.fetchall()]

    if return_tables_only:
        return tables

    print("Getting schema for each table that you selected...")

    table_columns = {}

    # get the columns for each table
    for schema in schemas:
        for table_name in tables:
            if "." in table_name:
                _, table_name = table_name.split(".", 1)
            cur.execute(
                "SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s AND table_schema = %s;",
                (
                    table_name,
                    schema,
                ),
            )
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            if scan:
                rows = identify_categorical_columns(cur, table_name, rows)
            if schema == "public":
                table_columns[table_name] = rows
            else:
                table_columns[schema + table_name] = rows
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
    schemas = {}

    print("Getting the schema for each table that you selected...")
    # get the schema for each table
    for table_name in tables:
        table = client.get_table(table_name)
        rows = table.schema
        rows = [{"column_name": i.name, "data_type": i.field_type} for i in rows]
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
    else:
        raise ValueError(
            f"Creation of a DB schema for {self.db_type} is not yet supported via the library. If you are a premium user, please contact us at founder@defog.ai so we can manually add it."
        )
