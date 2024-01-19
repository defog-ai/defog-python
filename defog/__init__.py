import base64
import json
import os
import requests
import pandas as pd
from defog.query import execute_query
from importlib.metadata import version
from defog import generate_schema

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
        base_url: str = "https://api.defog.ai",
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
            self.base_url = base_url
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
                        self.base_url = data.get("base_url", "https://api.defog.ai")
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

                self.base_url = base_url
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
                    "base_url": self.base_url,
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
            f"{self.base_url}/quota",
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

    def update_db_schema(self, path_to_csv):
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
            f"{self.base_url}/update_metadata",
            json={
                "api_key": self.api_key,
                "table_metadata": schema,
                "db_type": self.db_type,
            },
        )
        resp = r.json()
        return resp

    def update_glossary(self, glossary: str = "", customized_glossary: dict = None):
        """
        Updates the glossary on the defog servers.
        :param glossary: The glossary to be used.
        """
        r = requests.post(
            f"{self.base_url}/update_glossary",
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
            f"{self.base_url}/get_metadata",
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
            f"{self.base_url}/get_metadata",
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
            f"{self.base_url}/get_feedback",
            json={"api_key": self.api_key},
        )
        resp = r.json()
        df = pd.DataFrame(resp["data"], columns=resp["columns"])
        df["created_at"] = df["created_at"].apply(lambda x: x[:10])
        for col in ["query_generated", "feedback_text"]:
            df[col] = df[col].fillna("")
            df[col] = df[col].apply(lambda x: x.replace("\n", "\\n"))
        return df.iloc[start_from:].head(n_rows).to_markdown(index=False)


# Add all methods from generate_schema to Defog
for name in dir(generate_schema):
    attr = getattr(generate_schema, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)
