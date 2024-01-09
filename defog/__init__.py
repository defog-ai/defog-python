import base64
import json
import os
import requests
import pandas as pd
from defog.query import execute_query
from importlib.metadata import version
from io import StringIO

try:
    __version__ = version("defog")
except:
    pass

SUPPORTED_DB_TYPES = [
    "postgres",
    "redshift",
    "mysql",
    "bigquery",
    "snowflake",
    "databricks",
]


class Defog:
    """
    The main class for Defog
    """

    def __init__(
        self,
        api_key: str = "",
        db_type: str = "",
        db_creds: dict = {},
        base64creds: str = "",
        save_json: bool = True,
        generate_query_url: str = "https://api.defog.ai/generate_query_chat",
    ):
        """
        Initializes the Defog class.
        We have the possible scenarios detailed below:
        1) no config file, no/incomplete params -> success if only db_creds missing, error otherwise
        2) no config file, wrong params -> error
        3) no config file, all right params -> save params to config file
        4) config file present, no params -> read params from config file
        5) config file present, some/all params -> ignore existing config file, save new params to config file
        """
        if base64creds != "":
            self.from_base64_creds(base64creds)
            return
        self.home_dir = os.path.expanduser("~")
        self.filepath = os.path.join(self.home_dir, ".defog", "connection.json")

        if not os.path.exists(self.filepath) and (api_key != "" and db_type != ""):
            self.check_db_creds(db_type, db_creds)  # throws error for case 2
            # case 3
            self.api_key = api_key
            self.db_type = db_type
            self.db_creds = db_creds
            self.generate_query_url = generate_query_url
            # write to filepath and print confirmation
            if save_json:
                self.save_connection_json()
        elif os.path.exists(self.filepath):  # case 4 and 5
            # read connection details from filepath
            print("Connection details found. Reading connection details from file...")
            if api_key == "":
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    if "api_key" in data and "db_type" in data and "db_creds" in data:
                        self.check_db_creds(data["db_type"], data["db_creds"])
                        self.api_key = data["api_key"]
                        self.db_type = data["db_type"]
                        self.db_creds = data["db_creds"]
                        self.generate_query_url = data.get(
                            "generate_query_url",
                            "https://api.defog.ai/generate_query_chat",
                        )
                        print(f"Connection details read from {self.filepath}.")
                    else:
                        raise KeyError(
                            f"Invalid file at {self.filepath}.\n"
                            "Json file should contain 'api_key', 'db_type', 'db_creds'.\n"
                            "Please delete the file and try again."
                        )
            else:  # case 5
                if api_key != "":
                    self.api_key = api_key
                if db_type != "":
                    self.db_type = db_type

                self.generate_query_url = generate_query_url
                self.db_creds = db_creds
                self.check_db_creds(self.db_type, self.db_creds)
                if save_json:
                    self.save_connection_json()
        else:  # case 1
            raise ValueError(
                "Connection details not found. Please set up with the CLI or pass in the api_key, db_type, and db_creds parameters."
            )

    def save_connection_json(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(
                {
                    "api_key": self.api_key,
                    "db_type": self.db_type,
                    "db_creds": self.db_creds,
                    "generate_query_url": self.generate_query_url,
                },
                f,
                indent=4,
            )
        print(f"Connection details saved to {self.filepath}.")

    @staticmethod
    def check_db_creds(db_type: str, db_creds: dict):
        # print(db_creds)
        if db_creds == {}:
            # special case for empty db_creds. Some customers just want these to be empty so they can just get the query and run it without giving the defog library any credentials
            return
        if db_type == "postgres" or db_type == "redshift":
            if "host" not in db_creds:
                raise KeyError("db_creds must contain a 'host' key.")
            if "port" not in db_creds:
                raise KeyError("db_creds must contain a 'port' key.")
            if "database" not in db_creds:
                raise KeyError("db_creds must contain a 'database' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "mysql":
            if "host" not in db_creds:
                raise KeyError("db_creds must contain a 'host' key.")
            if "database" not in db_creds:
                raise KeyError("db_creds must contain a 'database' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "snowflake":
            if "account" not in db_creds:
                raise KeyError("db_creds must contain a 'account' key.")
            if "warehouse" not in db_creds:
                raise KeyError("db_creds must contain a 'warehouse' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "databricks":
            if "server_hostname" not in db_creds:
                raise KeyError("db_creds must contain a 'server_hostname' key.")
            if "access_token" not in db_creds:
                raise KeyError("db_creds must contain a 'access_token' key.")
            if "http_path" not in db_creds:
                raise KeyError("db_creds must contain a 'http_path' key.")
        elif db_type == "mongo" or db_type == "sqlserver":
            if "connection_string" not in db_creds:
                raise KeyError("db_creds must contain a 'connection_string' key.")
        elif db_type == "bigquery":
            if "json_key_path" not in db_creds:
                raise KeyError("db_creds must contain a 'json_key_path' key.")
        elif db_type == "elastic":
            if "host" not in db_creds:
                raise KeyError("db_creds must contain a 'host' key.")
            if "api_key" not in db_creds:
                raise KeyError("db_creds must contain a 'api_key' key.")
        else:
            raise ValueError(
                f"Database `{db_type}` is not supported right now. db_type must be one of {', '.join(SUPPORTED_DB_TYPES)}"
            )

    def generate_postgres_schema(
        self, tables: list, upload: bool = True, return_format: str = "gsheets"
    ) -> str:
        # when upload is True, we send the schema to the defog servers and generate a Google Sheet
        # when its false, we return the schema as a dict
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
            )

        conn = psycopg2.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}

        if tables == [""]:
            # get all tables
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
            tables = [row[0] for row in cur.fetchall()]
        print("Retrieved the following tables:")
        for t in tables:
            print(f"\t{t}")

        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            cur.execute(
                "SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s;",
                (table_name,),
            )
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows

        # get foreign key relationships
        print("Getting foreign keys for each table in your database...")
        tables_regclass_str = ", ".join(
            [f"'{table_name}'::regclass" for table_name in tables]
        )
        query = f"""SELECT
                conrelid::regclass AS table_from,
                pg_get_constraintdef(oid) AS foreign_key_definition
                FROM pg_constraint
                WHERE contype = 'f'
                AND conrelid::regclass IN ({tables_regclass_str})
                AND confrelid::regclass IN ({tables_regclass_str});
                """
        cur.execute(query)
        foreign_keys = list(cur.fetchall())
        foreign_keys = [fk[0] + " " + fk[1] for fk in foreign_keys]

        # get indexes for each table
        print("Getting indexes for each table in your database...")
        tables_str = ", ".join([f"'{table_name}'" for table_name in tables])
        query = (
            f"""SELECT indexdef FROM pg_indexes WHERE tablename IN ({tables_str});"""
        )
        cur.execute(query)
        indexes = list(cur.fetchall())
        if len(indexes) > 0:
            indexes = [index[0] for index in indexes]
        else:
            indexes = []
            # print("No indexes found.")
        conn.close()

        print(
            "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
        )
        if upload:
            # send the schemas dict to the defog servers
            if return_format == "gsheets":
                r = requests.post(
                    "https://api.defog.ai/get_postgres_schema_gsheets",
                    json={
                        "api_key": self.api_key,
                        "schemas": schemas,
                        "foreign_keys": foreign_keys,
                        "indexes": indexes,
                    },
                )
                resp = r.json()
                if "sheet_url" in resp:
                    gsheet_url = resp["sheet_url"]
                    return gsheet_url
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
                    )
            else:
                r = requests.post(
                    "https://api.defog.ai/get_schema_csv",
                    json={
                        "api_key": self.api_key,
                        "schemas": schemas,
                        "foreign_keys": foreign_keys,
                        "indexes": indexes,
                    },
                )
                resp = r.json()
                if "csv" in resp:
                    csv = resp["csv"]
                    pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                    return "defog_metadata.csv"
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
                    )
        else:
            return schemas

    def generate_redshift_schema(
        self, tables: list, upload: bool = True, return_format: str = "gsheets"
    ) -> str:
        # when upload is True, we send the schema to the defog servers and generate a Google Sheet
        # when its false, we return the schema as a dict
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 not installed. Please install it with `pip install psycopg2-binary`."
            )

        conn = psycopg2.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}

        if len(tables) == 0:
            # get all tables
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
            tables = [row[0] for row in cur.fetchall()]
        print("Retrieved the following tables:")
        for t in tables:
            print(f"\t{t}")

        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            try:
                cur.execute(
                    "SELECT CAST(column_name AS TEXT), CAST(data_type AS TEXT) FROM information_schema.columns WHERE table_name::text = %s;",
                    (table_name,),
                )
                rows = cur.fetchall()
                rows = [row for row in rows]
            except:
                # dirty hack for redshift spectrum
                rows = []
            if len(rows) == 0:
                cur.execute(
                    f"SELECT CAST(columnname AS TEXT), CAST(external_type AS TEXT) FROM svv_external_columns WHERE table_name = '{table_name}';"
                )
                rows = cur.fetchall()
                rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows

        # get foreign key relationships
        print("Getting foreign keys for each table in your database...")
        tables_regclass_str = ", ".join(
            [f"'{table_name}'::regclass" for table_name in tables]
        )
        query = f"""SELECT
                conrelid::regclass AS table_from,
                pg_get_constraintdef(oid) AS foreign_key_definition
                FROM pg_constraint
                WHERE contype = 'f'
                AND conrelid::regclass IN ({tables_regclass_str})
                AND confrelid::regclass IN ({tables_regclass_str});
                """
        cur.execute(query)
        foreign_keys = list(cur.fetchall())
        foreign_keys = [fk[0] + " " + fk[1] for fk in foreign_keys]

        # get indexes for each table
        print("Getting indexes for each table in your database...")
        tables_str = ", ".join([f"'{table_name}'" for table_name in tables])
        query = (
            f"""SELECT indexdef FROM pg_indexes WHERE tablename IN ({tables_str});"""
        )
        cur.execute(query)
        indexes = list(cur.fetchall())
        if len(indexes) > 0:
            indexes = [index[0] for index in indexes]
        else:
            indexes = []
            # print("No indexes found.")
        conn.close()

        if upload:
            print(
                "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
            )
            # send the schemas dict to the defog servers
            if return_format == "gsheets":
                r = requests.post(
                    "https://api.defog.ai/get_postgres_schema_gsheets",
                    json={
                        "api_key": self.api_key,
                        "schemas": schemas,
                        "foreign_keys": foreign_keys,
                        "indexes": indexes,
                    },
                )
                resp = r.json()
                if "sheet_url" in resp:
                    gsheet_url = resp["sheet_url"]
                    return gsheet_url
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue if this a generic library issue, or email support@defog.ai if you need dedicated customer-specific support."
                    )
            else:
                r = requests.post(
                    "https://api.defog.ai/get_schema_csv",
                    json={
                        "api_key": self.api_key,
                        "schemas": schemas,
                        "foreign_keys": foreign_keys,
                        "indexes": indexes,
                    },
                )
                resp = r.json()
                if "csv" in resp:
                    csv = resp["csv"]
                    pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                    return "defog_metadata.csv"
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
        self, tables: list, upload: bool = True, return_format: str = "gsheets"
    ) -> str:
        try:
            import mysql.connector
        except:
            raise Exception("mysql-connector not installed.")

        conn = mysql.connector.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}

        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;",
                (table_name,),
            )
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows

        conn.close()

        if upload:
            if return_format == "gsheets":
                print(
                    "Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes..."
                )
                # send the schemas dict to the defog servers
                r = requests.post(
                    "https://api.defog.ai/get_postgres_schema_gsheets",
                    json={"api_key": self.api_key, "schemas": schemas},
                )
                resp = r.json()
                if "sheet_url" in resp:
                    gsheet_url = resp["sheet_url"]
                    return gsheet_url
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue if this a generic library issue, or email support@defog.ai if you need dedicated customer-specific support."
                    )
            else:
                r = requests.post(
                    "https://api.defog.ai/get_schema_csv",
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
                    pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                    return "defog_metadata.csv"
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
                    )
        else:
            return schema

    def generate_databricks_schema(
        self, tables: list, upload: bool = True, return_format: str = "csv"
    ) -> str:
        try:
            from databricks import sql
        except:
            raise Exception("databricks-sql-connector not installed.")

        conn = sql.connect(**self.db_creds)
        schemas = {}
        with conn.cursor() as cur:
            print("Getting schema for each table in your database...")
            # get the schema for each table
            for table_name in tables:
                cur.columns(
                    schema_name=self.db_creds.get("schema", "default"),
                    table_name=table_name,
                )
                rows = cur.fetchall()
                rows = [row for row in rows]
                rows = [{"column_name": i[3], "data_type": i[5]} for i in rows]
                schemas[table_name] = rows

            conn.close()

        if upload:
            r = requests.post(
                "https://api.defog.ai/get_schema_csv",
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
                pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                return "defog_metadata.csv"
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
        self, tables: list, upload: bool = True, return_format: str = "gsheets"
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
        print("Getting schema for each table in your database...")
        # get the schema for each table
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
            schemas[table_name] = rows

        conn.close()

        if upload:
            print(
                "Sending the schema to the defog servers and generating column descriptions. This might take up to 2 minutes..."
            )
            if return_format == "gsheets":
                # send the schemas dict to the defog servers
                r = requests.post(
                    "https://api.defog.ai/get_postgres_schema_gsheets",
                    json={"api_key": self.api_key, "schemas": schemas},
                )
                resp = r.json()
                if "sheet_url" in resp:
                    gsheet_url = resp["sheet_url"]
                    return gsheet_url
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue if this a generic library issue, or email support@defog.ai if you need dedicated customer-specific support."
                    )
            else:
                r = requests.post(
                    "https://api.defog.ai/get_schema_csv",
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
                    pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                    return "defog_metadata.csv"
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
        self, tables: list, upload: bool = True, return_format: str = "gsheets"
    ) -> str:
        try:
            from google.cloud import bigquery
        except:
            raise Exception("google-cloud-bigquery not installed.")

        client = bigquery.Client.from_service_account_json(
            self.db_creds["json_key_path"]
        )
        schemas = {}

        print("Getting the schema for each table in your database...")
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
            if return_format == "gsheets":
                # send the schemas dict to the defog servers
                r = requests.post(
                    "https://api.defog.ai/get_bigquery_schema_gsheets",
                    json={"api_key": self.api_key, "schemas": schemas},
                )
                resp = r.json()
                if "sheet_url" in resp:
                    gsheet_url = resp["sheet_url"]
                    return gsheet_url
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue if this a generic library issue, or email support@defog.ai if you need dedicated customer-specific support."
                    )
            else:
                r = requests.post(
                    "https://api.defog.ai/get_schema_csv",
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
                    pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                    return "defog_metadata.csv"
                else:
                    print(f"We got an error!")
                    if "message" in resp:
                        print(f"Error message: {resp['message']}")
                    print(
                        f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
                    )
        else:
            return schemas

    def generate_db_schema(self, tables: list) -> str:
        if self.db_type == "postgres":
            return self.generate_postgres_schema(tables, return_format="csv")
        elif self.db_type == "mysql":
            return self.generate_mysql_schema(tables, return_format="csv")
        elif self.db_type == "bigquery":
            return self.generate_bigquery_schema(tables, return_format="csv")
        elif self.db_type == "redshift":
            return self.generate_redshift_schema(tables, return_format="csv")
        elif self.db_type == "snowflake":
            return self.generate_snowflake_schema(tables, return_format="csv")
        elif self.db_type == "databricks":
            return self.generate_databricks_schema(tables, return_format="csv")
        else:
            raise ValueError(
                f"Creation of a DB schema for {self.db_type} is not yet supported via the library. If you are a premium user, please contact us at founder@defog.ai so we can manually add it."
            )

    def update_mysql_schema(self, gsheet_url: str):
        """
        Updates the postgres schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post(
            "https://api.defog.ai/update_postgres_schema",
            json={"api_key": self.api_key, "gsheet_url": gsheet_url},
        )
        resp = r.json()
        return resp

    def update_postgres_schema(self, gsheet_url: str):
        """
        Updates the postgres schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post(
            "https://api.defog.ai/update_postgres_schema",
            json={"api_key": self.api_key, "gsheet_url": gsheet_url},
        )
        resp = r.json()
        return resp

    def update_redshift_schema(self, gsheet_url: str):
        """
        Updates the redshift schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post(
            "https://api.defog.ai/update_postgres_schema",
            json={"api_key": self.api_key, "gsheet_url": gsheet_url},
        )
        resp = r.json()
        return resp

    def update_mongo_schema(self, gsheet_url: str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post(
            "https://api.defog.ai/update_mongo_schema",
            json={"api_key": self.api_key, "gsheet_url": gsheet_url},
        )
        resp = r.json()
        return resp

    def update_bigquery_schema(self, gsheet_url: str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post(
            "https://api.defog.ai/update_bigquery_schema",
            json={"api_key": self.api_key, "gsheet_url": gsheet_url},
        )
        resp = r.json()
        return resp

    def update_snowflake_schema(self, gsheet_url: str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post(
            "https://api.defog.ai/update_snowflake_schema",
            json={"api_key": self.api_key, "gsheet_url": gsheet_url},
        )
        resp = r.json()
        return resp

    def update_sqlserver_schema(self, gsheet_url: str):
        """
        Updates the mongo schema on the defog servers.
        :param gsheet_url: The url of the google sheet containing the schema.
        """
        r = requests.post(
            "https://api.defog.ai/update_postgres_schema",
            json={"api_key": self.api_key, "gsheet_url": gsheet_url},
        )
        resp = r.json()
        return resp

    def update_db_schema(self, gsheet_url: str):
        print(
            "Updating the schema on the Defog servers. This might take a couple minutes..."
        )
        if self.db_type == "postgres":
            return self.update_postgres_schema(gsheet_url)
        elif self.db_type == "mysql":
            return self.update_mysql_schema(gsheet_url)
        elif self.db_type == "mongo":
            return self.update_mongo_schema(gsheet_url)
        elif self.db_type == "bigquery":
            return self.update_bigquery_schema(gsheet_url)
        elif self.db_type == "redshift":
            return self.update_redshift_schema(gsheet_url)
        elif self.db_type == "snowflake":
            return self.update_snowflake_schema(gsheet_url)
        elif self.db_type == "sqlserver":
            return self.update_sqlserver_schema(gsheet_url)
        else:
            raise Exception(
                "Updating the schema for this database type via the library is not yet supported. If you are a premium user, please contact us and we will manually add it."
            )

    def get_query(
        self,
        question: str,
        hard_filters: str = "",
        previous_context: list = [],
        schema: dict = {},
        glossary: str = "",
        language: str = None,
        debug: bool = False,
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
                "language": language,
                "hard_filters": hard_filters,
            }
            if schema != {}:
                data["schema"] = schema
                data["is_direct"] = True
            r = requests.post(
                self.generate_query_url,
                json=data,
                timeout=300,
            )
            resp = r.json()
            query_generated = resp.get("sql", resp.get("query_generated"))
            ran_successfully = resp.get("ran_successfully")
            error_message = resp.get("error_message")
            query_db = self.db_type
            return {
                "query_generated": query_generated,
                "ran_successfully": ran_successfully,
                "error_message": error_message,
                "query_db": query_db,
                "previous_context": resp.get("previous_context"),
                "reason_for_query": resp.get("reason_for_query"),
            }
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
            )
        if query["ran_successfully"]:
            try:
                print("Query generated, now running it on your database...")
                colnames, result, executed_query = execute_query(
                    query["query_generated"],
                    self.api_key,
                    self.db_type,
                    self.db_creds,
                    question,
                    hard_filters,
                    retries,
                )
                return {
                    "columns": colnames,
                    "data": result,
                    "query_generated": executed_query,
                    "ran_successfully": True,
                    "reason_for_query": query.get("reason_for_query"),
                    "previous_context": query.get("previous_context"),
                }
            except Exception as e:
                return {
                    "ran_successfully": False,
                    "error_message": str(e),
                    "query_generated": query["query_generated"],
                }
        else:
            return {"ran_successfully": False, "error_message": query["error_message"]}

    def get_quota(self) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.get(
            "https://api.defog.ai/quota",
            headers=headers,
        )
        return response.json()

    def to_base64_creds(self) -> str:
        creds = {
            "api_key": self.api_key,
            "db_type": self.db_type,
            "db_creds": self.db_creds,
        }
        return base64.b64encode(json.dumps(creds).encode("utf-8")).decode("utf-8")

    def from_base64_creds(self, base64_creds: str):
        creds = json.loads(base64.b64decode(base64_creds).decode("utf-8"))
        self.api_key = creds["api_key"]
        self.db_type = creds["db_type"]
        self.db_creds = creds["db_creds"]

    def update_predefined_queries(self, predefined_queries: list):
        """
        Updates the predefined queries on the defog servers.
        :param predefined_queries: The predefined queries to be used.
        """
        # [{'question': 'What is the total number of employees?', 'query': 'SELECT COUNT(*) FROM employees'}}]
        for item in predefined_queries:
            if "question" not in item or "query" not in item:
                raise Exception(
                    "Each predefined query should have a question and a SQL query. It should be in the format {{'question': 'YOUR QUESTION', 'query': 'SELECT ...'}}"
                )

        r = requests.post(
            "https://api.defog.ai/update_predefined_queries",
            json={"api_key": self.api_key, "predefined_queries": predefined_queries},
        )
        resp = r.json()
        return resp

    def get_predefined_queries(self):
        """
        Gets the predefined queries on the defog servers.
        """
        r = requests.post(
            "https://api.defog.ai/get_predefined_queries",
            json={
                "api_key": self.api_key,
            },
        )
        resp = r.json()
        if resp["status"] == "success":
            return resp["predefined_queries"]
        else:
            return []

    def execute_predefined_query(self, query):
        """
        Executes a predefined query
        """
        resp = execute_query(
            query["query"],
            self.api_key,
            self.db_type,
            self.db_creds,
        )
        return resp

    def update_db_schema_csv(self, path_to_csv):
        """
        Update the DB schema via a CSV, rather than by via a Google Sheet
        """
        schema_df = pd.read_csv(path_to_csv).fillna("")
        schema = {}
        for table_name in schema_df["table_name"].unique():
            schema[table_name] = schema_df[schema_df["table_name"] == table_name][
                ["column_name", "data_type", "column_description"]
            ].to_dict(orient="records")

        r = requests.post(
            "https://api.defog.ai/update_metadata",
            json={"api_key": self.api_key, "table_metadata": schema},
        )
        resp = r.json()
        return resp

    def update_glossary(self, glossary: str = "", customized_glossary: dict = None):
        """
        Updates the glossary on the defog servers.
        :param glossary: The glossary to be used.
        """
        r = requests.post(
            "https://api.defog.ai/update_glossary",
            json={
                "api_key": self.api_key,
                "glossary": glossary,
                "customized_glossary": customized_glossary,
            },
        )
        resp = r.json()
        return resp

    def get_glossary(self, mode="general"):
        """
        Gets the glossary on the defog servers.
        """
        r = requests.post(
            "https://api.defog.ai/get_metadata",
            json={"api_key": self.api_key},
        )
        resp = r.json()
        if mode == "general":
            return resp["glossary"]
        elif mode == "customized":
            return resp["customized_glossary"]

    def get_metadata(self, format="markdown", export_path=None):
        """
        Gets the metadata on the defog servers.
        """
        r = requests.post(
            "https://api.defog.ai/get_metadata",
            json={"api_key": self.api_key},
        )
        resp = r.json()
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

    def get_feedback(self, n_rows: int = 50, start_from: int = 0):
        """
        Gets the feedback on the defog servers.
        """
        r = requests.post(
            "https://api.defog.ai/get_feedback",
            json={"api_key": self.api_key},
        )
        resp = r.json()
        df = pd.DataFrame(resp["data"], columns=resp["columns"])
        df["created_at"] = df["created_at"].apply(lambda x: x[:10])
        for col in ["query_generated", "feedback_text"]:
            df[col] = df[col].fillna("")
            df[col] = df[col].apply(lambda x: x.replace("\n", "\\n"))
        return df.iloc[start_from:].head(n_rows).to_markdown(index=False)
