import base64
import json
import os

import requests

SUPPORTED_DB_TYPES = [
    "postgres",
    "redshift",
    "mysql",
    "bigquery",
    "mongo",
    "snowflake",
    "sqlserver",
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
    ):
        """
        Initializes the Defog class.
        :param api_key: The API key for the defog account.
        """
        if base64creds != "":
            self.from_base64_creds(base64creds)
            return
        home_dir = os.path.expanduser("~")
        filepath = os.path.join(home_dir, ".defog", "connection.json")
        if not os.path.exists(filepath) or (
            api_key != "" and db_type != "" and db_creds != {}
        ):
            # read connection details from args
            self.check_db_creds(db_type, db_creds)
            self.api_key = api_key
            self.db_type = db_type
            self.db_creds = db_creds
            data = {"api_key": api_key, "db_type": db_type, "db_creds": db_creds}
            # write to filepath and print confirmation
            if save_json:
                print(
                    f"Connection details not found in {filepath}.\nSaving connection details to file..."
                )
                if not os.path.exists(os.path.join(home_dir, ".defog")):
                    os.mkdir(os.path.join(home_dir, ".defog"))
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Connection details saved to {filepath}.")
        elif os.path.exists(filepath):
            # read connection details from filepath
            print("Connection details found. Reading connection details from file...")
            with open(filepath, "r") as f:
                data = json.load(f)
                if "api_key" in data and "db_type" in data and "db_creds" in data:
                    self.check_db_creds(data["db_type"], data["db_creds"])
                    self.api_key = data["api_key"]
                    self.db_type = data["db_type"]
                    self.db_creds = data["db_creds"]
                    print(f"Connection details read from {filepath}.")
                else:
                    raise KeyError(
                        f"Invalid file at {filepath}.\n"
                        "Json file should contain 'api_key', 'db_type', 'db_creds'.\n"
                        "Please delete the file and try again."
                    )
        else:
            raise ValueError(
                "Connection details not found. Please set up with the CLI or pass in the api_key, db_type, and db_creds parameters."
            )

    @staticmethod
    def check_db_creds(db_type: str, db_creds: dict):
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
        elif db_type == "mongo" or db_type == "sqlserver":
            if "connection_string" not in db_creds:
                raise KeyError("db_creds must contain a 'connection_string' key.")
        elif db_type == "bigquery":
            if "json_key_path" not in db_creds:
                raise KeyError("db_creds must contain a 'json_key_path' key.")
        else:
            raise ValueError(
                f"Database `{db_type}` is not supported right now. db_type must be one of {', '.join(SUPPORTED_DB_TYPES)}"
            )
    
    def check_db_suitability(self, gsheets_url=None, tables=None):
        # either send a request to the defog api or do it locally
        if gsheets_url is not None:
            # send a request to the defog api
            raise NotImplemented("Checking suitability from a Google Sheets URL is not supported right now.")
        elif tables:
            # run heuristic locally
            # see how many table + column combinations the user has passed
            # then, see if the column names are semantically meaningful
            # if there are more than 15 tables and more than 200 columns, we should say that they should contact us to check suitability
            # else, we should say that we can support this use case

            # get the schema for each table
            if self.db_type == "postgres":
                schemas = self.generate_postgres_schema(tables, upload=False)
            elif self.db_type == "redshift":
                schemas = self.generate_redshift_schema(tables, upload=False)
            elif self.db_type == "mysql":
                schemas = self.generate_mysql_schema(tables, upload=False)
            elif self.db_type == "snowflake":
                schemas = self.generate_snowflake_schema(tables, upload=False)
            elif self.db_type == "bigquery":
                schemas = self.generate_bigquery_schema(tables, upload=False)
            else:
                raise NotImplemented(
                    f"Database `{self.db_type}` is not supported right now for schema checks. Please contact us at founders@defog.ai to request support."
                )
            
            tot_columns = 0
            for table in schemas:
                tot_columns += len(schemas[table])

            if len(schemas) > 15:
                message = "You want to query more than than 15 tables in total. We should be able to support your use-case, but yuo should contact us at founders@defog.ai to confirm."
            elif tot_columns > 200:
                message = "You want to query more than 200 columns in total. We should be able to support your use-case, but you should contact us at at founders@defog.ai to confirm."
            else:
                # check if there are a lot of columns with the json type
                json_columns = 0
                for table in schemas:
                    for item in schemas[table]:
                        if "json" in item["data_type"].lower():
                            json_columns += 1
                if json_columns > 2:
                    message = "There are 2 or more columns with the json type. If you do not need to make joins between JSON columns and others, we can definitely support your use case. If you do need to make such joins, please contact us at founders@defog.ai."
                else:
                    message = "We should be able to support your use-case! Feel free to upgrade to a paid plan to get started, or contact as at founders@defog.ai if you have any questions."
            
            print(message)
            return True

    def generate_postgres_schema(self, tables: list, upload: bool =True) -> str:
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
            print("No indexes found.")
        conn.close()

        print(
            "Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes..."
        )
        if upload:
            # send the schemas dict to the defog servers
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
            try:
                gsheet_url = resp["sheet_url"]
                return gsheet_url
            except Exception as e:
                print(resp)
                raise resp["message"]
        else:
            return schemas

    def generate_redshift_schema(self, tables: list, upload: bool = True) -> str:
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
            print("No indexes found.")
        conn.close()

        if upload:
            print(
                "Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes..."
            )
            # send the schemas dict to the defog servers
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
            try:
                gsheet_url = resp["sheet_url"]
                return gsheet_url
            except Exception as e:
                print(resp)
                raise resp["message"]
        else:
            return schemas

    def generate_mysql_schema(self, tables: list, upload: bool = True) -> str:
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
            print(
                "Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes..."
            )
            # send the schemas dict to the defog servers
            r = requests.post(
                "https://api.defog.ai/get_postgres_schema_gsheets",
                json={"api_key": self.api_key, "schemas": schemas},
            )
            resp = r.json()
            try:
                gsheet_url = resp["sheet_url"]
                return gsheet_url
            except Exception as e:
                print(resp)
                raise resp["message"]
        else:
            return schemas

    def generate_sqlserver_schema(self, tables: list, upload: bool = True) -> str:
        try:
            import pyodbc
        except:
            raise Exception("pyodbc not installed.")

        conn = pyodbc.connect(self.db_creds["connection_string"])
        cur = conn.cursor()
        schemas = {}

        print("Getting schema for each table in your database...")
        # get the schema for each table
        for table_name in tables:
            cur.execute(
                f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"
            )
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1]} for i in rows]
            schemas[table_name] = rows

        conn.close()
        if upload:
            print(
                "Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes..."
            )
            # send the schemas dict to the defog servers
            r = requests.post(
                "https://api.defog.ai/get_postgres_schema_gsheets",
                json={"api_key": self.api_key, "schemas": schemas},
            )
            resp = r.json()
            try:
                gsheet_url = resp["sheet_url"]
                return gsheet_url
            except Exception as e:
                print(resp)
                raise resp["message"]
        else:
            return schemas

    def generate_snowflake_schema(self, tables: list, upload: bool = True) -> str:
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
                "Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes..."
            )
            # send the schemas dict to the defog servers
            r = requests.post(
                "https://api.defog.ai/get_postgres_schema_gsheets",
                json={"api_key": self.api_key, "schemas": schemas},
            )
            resp = r.json()
            try:
                gsheet_url = resp["sheet_url"]
                return gsheet_url
            except Exception as e:
                print(resp)
                raise resp["message"]
        else:
            return schemas

    def generate_mongo_schema(self, collections: list) -> str:
        try:
            from pymongo import MongoClient
        except:
            raise Exception("pymongo not installed.")

        client = MongoClient(self.db_creds["connection_string"])
        db = client.get_database()

        schemas = {}

        print("Getting schema for each collection in your database...")
        # get the schema for each table
        for collection_name in collections:
            collection = db[collection_name]
            rows = collection.find_one()
            rows = [
                {"field_name": i, "data_type": type(rows[i]).__name__} for i in rows
            ]
            schemas[collection_name] = rows

        client.close()

        print(
            "Sending the schema to the defog servers and generating a Google Sheet. This might take up to 2 minutes..."
        )
        # send the schemas dict to the defog servers
        r = requests.post(
            "https://api.defog.ai/get_mongo_schema_gsheets",
            json={"api_key": self.api_key, "schemas": schemas},
        )
        resp = r.json()
        try:
            gsheet_url = resp["sheet_url"]
            return gsheet_url
        except Exception as e:
            print(resp)
            raise resp["message"]

    def generate_bigquery_schema(self, tables: list, upload: bool = True) -> str:
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
                "Sending the schema to Defog servers and generating a Google Sheet. This might take up to 2 minutes..."
            )
            # send the schemas dict to the defog servers
            r = requests.post(
                "https://api.defog.ai/get_bigquery_schema_gsheets",
                json={"api_key": self.api_key, "schemas": schemas},
            )
            resp = r.json()
            try:
                gsheet_url = resp["sheet_url"]
                return gsheet_url
            except Exception as e:
                print(resp)
                raise resp["message"]
        else:
            return schemas

    def generate_db_schema(self, tables: list) -> str:
        if self.db_type == "postgres":
            return self.generate_postgres_schema(tables)
        elif self.db_type == "mysql":
            return self.generate_mysql_schema(tables)
        elif self.db_type == "mongo":
            return self.generate_mongo_schema(tables)
        elif self.db_type == "bigquery":
            return self.generate_bigquery_schema(tables)
        elif self.db_type == "redshift":
            return self.generate_redshift_schema(tables)
        elif self.db_type == "snowflake":
            return self.generate_snowflake_schema(tables)
        elif self.db_type == "sqlserver":
            return self.generate_sqlserver_schema(tables)
        else:
            raise ValueError(
                "Invalid database type. Valid types are: postgres, mysql, mongo, bigquery, and redshift"
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
                "Invalid database type. Valid types are: postgres, mysql, mongo, bigquery, and redshift"
            )

    def update_glossary(self, glossary: str):
        """
        Updates the glossary on the defog servers.
        :param glossary: The glossary to be used.
        """
        r = requests.post(
            "https://api.defog.ai/update_glossary",
            json={"api_key": self.api_key, "glossary": glossary},
        )
        resp = r.json()
        return resp

    def get_query(
        self,
        question: str,
        hard_filters: str = "",
        previous_context: list = [],
        schema: dict = {},
        mode: str = "default",
    ):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :return: The response from the defog server.
        """
        if schema == {}:
            schema = None

        try:
            if mode == "default":
                r = requests.post(
                    "https://api.defog.ai/generate_query",
                    json={
                        "question": question,
                        "api_key": self.api_key,
                        "hard_filters": hard_filters,
                        "db_type": self.db_type,
                        "schema": schema,
                    },
                    timeout=30,
                )
                resp = r.json()
                query_generated = resp.get("query_generated")
                ran_successfully = resp["ran_successfully"]
                error_message = resp.get("error_message")
                query_db = resp.get("query_db", "postgres")
            else:
                r = requests.post(
                    "https://api.defog.ai/generate_query_chat",
                    json={
                        "question": question,
                        "api_key": self.api_key,
                        "previous_context": previous_context,
                        "db_type": self.db_type,
                        "schema": schema,
                    },
                    timeout=30,
                )
                resp = r.json()
                query_generated = resp.get("sql")
                ran_successfully = resp.get("ran_successfully")
                error_message = resp.get("error_message")
                query_db = self.db_type
            return {
                "query_generated": query_generated,
                "ran_successfully": ran_successfully,
                "error_message": error_message,
                "query_db": query_db,
                "previous_context": resp.get("previous_context"),
                "suggestion_for_further_questions": resp.get(
                    "suggestion_for_further_questions"
                ),
                "reason_for_query": resp.get("reason_for_query"),
            }
        except:
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
        mode: str = "default",
    ):
        """
        Sends the query to the defog servers, and return the response.
        :param question: The question to be asked.
        :return: The response from the defog server.
        """
        print(f"Generating the query for your question: {question}...")
        query = self.get_query(question, hard_filters, previous_context, mode=mode)
        if query["ran_successfully"]:
            print("Query generated, now running it on your database...")
            if query["query_db"] == "postgres" or query["query_db"] == "redshift":
                try:
                    import psycopg2
                except:
                    raise Exception("psycopg2 not installed.")

                conn = psycopg2.connect(**self.db_creds)
                cur = conn.cursor()
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True,
                        "reason_for_query": query.get("reason_for_query"),
                        "suggestion_for_further_questions": query.get(
                            "suggestion_for_further_questions"
                        ),
                        "previous_context": query.get("previous_context"),
                    }
                except Exception as e:
                    print(f"Query generated was: {query['query_generated']}")
                    print(
                        f"There was an error {str(e)} when running the previous query. Retrying with adaptive learning..."
                    )
                    # retry the query with the exception
                    r = requests.post(
                        "https://api.defog.ai/retry_query_after_error",
                        json={
                            "api_key": self.api_key,
                            "previous_query": query["query_generated"],
                            "error": str(e),
                            "db_type": self.db_type,
                            "hard_filters": hard_filters,
                            "question": question,
                        },
                    )
                    query = r.json()
                    conn = psycopg2.connect(**self.db_creds)
                    cur = conn.cursor()
                    try:
                        cur.execute(query["new_query"])
                        colnames = [desc[0] for desc in cur.description]
                        result = cur.fetchall()
                        cur.close()
                        conn.close()
                        print("Query ran succesfully!")
                        return {
                            "columns": colnames,
                            "data": result,
                            "query_generated": query["new_query"],
                            "ran_successfully": True,
                        }
                    except Exception as e:
                        return {"error_message": str(e), "ran_successfully": False}
            elif query["query_db"] == "mysql":
                try:
                    import mysql.connector
                except:
                    raise Exception("mysql.connector not installed.")
                conn = mysql.connector.connect(**self.db_creds)
                cur = conn.cursor()
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True,
                    }
                except Exception as e:
                    print(f"Query generated was: {query['query_generated']}")
                    print(
                        f"There was an error {str(e)} when running the previous query. Retrying with adaptive learning..."
                    )
                    # retry the query with the exception
                    r = requests.post(
                        "https://api.defog.ai/retry_query_after_error",
                        json={
                            "api_key": self.api_key,
                            "previous_query": query["query_generated"],
                            "error": str(e),
                            "db_type": self.db_type,
                            "hard_filters": hard_filters,
                            "question": question,
                        },
                    )
                    query = r.json()
                    conn = mysql.connector.connect(**self.db_creds)
                    cur = conn.cursor()
                    try:
                        cur.execute(query["new_query"])
                        colnames = [desc[0] for desc in cur.description]
                        result = cur.fetchall()
                        cur.close()
                        conn.close()
                        print("Query ran succesfully!")
                        return {
                            "columns": colnames,
                            "data": result,
                            "query_generated": query["new_query"],
                            "ran_successfully": True,
                        }
                    except Exception as e:
                        return {"error_message": str(e), "ran_successfully": False}
            elif query["query_db"] == "mongo":
                try:
                    from pymongo import MongoClient
                except:
                    raise Exception("pymongo not installed.")
                client = MongoClient(self.db_creds["connection_string"])
                db = client.get_database()
                try:
                    results = eval(f"{query['query_generated']}")
                    results = [i for i in results]
                    if len(results) > 0:
                        columns = results[0].keys()
                    else:
                        columns = []
                    return {
                        "columns": columns,  # assumes that all objects have the same keys
                        "data": results,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True,
                    }
                except Exception as e:
                    return {"error_message": str(e), "ran_successfully": False}
            elif query["query_db"] == "bigquery":
                try:
                    from google.cloud import bigquery
                except:
                    raise Exception("google.cloud.bigquery not installed.")

                json_key = self.db_creds
                client = bigquery.Client.from_service_account_json(json_key)
                try:
                    query_job = client.query(query["query_generated"])
                    results = query_job.result()
                    columns = [i.name for i in results.schema]
                    rows = []
                    for row in results:
                        rows.append([row[i] for i in range(len(row))])

                    return {
                        "columns": columns,  # assumes that all objects have the same keys
                        "data": rows,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True,
                    }
                except Exception as e:
                    print(f"Query generated was: {query['query_generated']}")
                    print(
                        f"There was an error {str(e)} when running the previous query. Retrying with adaptive learning..."
                    )
                    # retry the query with the exception
                    r = requests.post(
                        "https://api.defog.ai/retry_query_after_error",
                        json={
                            "api_key": self.api_key,
                            "previous_query": query["query_generated"],
                            "error": str(e),
                            "db_type": self.db_type,
                            "hard_filters": hard_filters,
                            "question": question,
                        },
                    )
                    client = bigquery.Client.from_service_account_json(json_key)
                    query_job = client.query(r.json()["new_query"])
                    results = query_job.result()
                    columns = [i.name for i in results.schema]
                    rows = []
                    for row in results:
                        rows.append([row[i] for i in range(len(row))])

                    return {
                        "columns": columns,  # assumes that all objects have the same keys
                        "data": rows,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True,
                    }
            elif query["query_db"] == "snowflake":
                try:
                    import snowflake.connector
                except:
                    raise Exception("snowflake.connector not installed.")
                conn = snowflake.connector.connect(
                    user=self.db_creds["user"],
                    password=self.db_creds["password"],
                    account=self.db_creds["account"],
                )
                cur = conn.cursor()
                cur.execute(
                    f"USE WAREHOUSE {self.db_creds['warehouse']}"
                )  # set the warehouse
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True,
                        "reason_for_query": query.get("reason_for_query"),
                        "suggestion_for_further_questions": query.get(
                            "suggestion_for_further_questions"
                        ),
                        "previous_context": query.get("previous_context"),
                    }
                except Exception as e:
                    return {"error_message": str(e), "ran_successfully": False}
            elif query["query_db"] == "sqlserver":
                try:
                    import pyodbc
                except:
                    raise Exception("pyodbc not installed.")
                conn = pyodbc.connect(self.db_creds)
                cur = conn.cursor()
                try:
                    cur.execute(query["query_generated"])
                    colnames = [desc[0] for desc in cur.description]
                    result = cur.fetchall()
                    cur.close()
                    conn.close()
                    print("Query ran succesfully!")
                    return {
                        "columns": colnames,
                        "data": result,
                        "query_generated": query["query_generated"],
                        "ran_successfully": True,
                    }
                except Exception as e:
                    return {"error_message": str(e), "ran_successfully": False}
            else:
                raise Exception("Database type not yet supported.")
        else:
            return {"ran_successfully": False, "error_message": query["error_message"]}

    def to_base64_creds(self) -> str:
        return base64.b64encode(json.dumps(self.__dict__).encode("utf-8")).decode(
            "utf-8"
        )

    def from_base64_creds(self, base64_creds: str):
        self.__dict__ = json.loads(base64.b64decode(base64_creds).decode("utf-8"))
