import base64
import json
import os
from importlib.metadata import version
from defog import generate_schema, query_methods, admin_methods, health_methods

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
        base_url: str = "https://api.defog.ai",
        generate_query_url: str = "https://api.defog.ai/generate_query_chat",
        verbose: bool = False,
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
            if verbose:
                print(
                    "Connection details found. Reading connection details from file..."
                )
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
                            f"{self.base_url}/generate_query_chat",
                        )
                        if verbose:
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
        elif db_type == "sqlserver":
            if "server" not in db_creds:
                raise KeyError("db_creds must contain a 'server' key.")
            if "database" not in db_creds:
                raise KeyError("db_creds must contain a 'database' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "mongo":
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


# Add all methods from generate_schema to Defog
for name in dir(generate_schema):
    attr = getattr(generate_schema, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)

# Add all methods from query_methods to Defog
for name in dir(query_methods):
    attr = getattr(query_methods, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)

# Add all methods from admin_methods to Defog
for name in dir(admin_methods):
    attr = getattr(admin_methods, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)

# Add all methods from health_methods to Defog
for name in dir(health_methods):
    attr = getattr(health_methods, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)
