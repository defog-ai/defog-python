import json
from typing import Dict, List, Optional
import requests
import pandas as pd


def update_db_schema(self, path_to_csv, dev=False):
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

    r = requests.post(
        f"{self.base_url}/update_metadata",
        json={
            "api_key": self.api_key,
            "table_metadata": schema,
            "db_type": self.db_type,
            "dev": dev,
        },
    )
    resp = r.json()
    return resp


def update_glossary(
    self, glossary: str = "", customized_glossary: dict = None, dev: bool = False
):
    """
    Updates the glossary on the defog servers.
    :param glossary: The glossary to be used.
    """
    data = {
        "api_key": self.api_key,
        "glossary": glossary,
        "dev": dev,
    }
    if customized_glossary:
        data["customized_glossary"] = customized_glossary
    r = requests.post(f"{self.base_url}/update_glossary", json=data)
    resp = r.json()
    return resp


def delete_glossary(self, user_type=None, dev=False):
    """
    Deletes the glossary on the defog servers.
    """
    data = {
        "api_key": self.api_key,
        "dev": dev,
    }
    if user_type:
        data["key"] = user_type
    r = requests.post(f"{self.base_url}/delete_glossary", json=data)
    if r.status_code == 200:
        print("Glossary deleted successfully.")
    else:
        error_message = r.json().get("message", "")
        print(f"Glossary deletion failed.\nError message: {error_message}")


def get_glossary(self, mode="general", dev=False):
    """
    Gets the glossary on the defog servers.
    """
    r = requests.post(
        f"{self.base_url}/get_metadata",
        json={"api_key": self.api_key, "dev": dev},
    )
    resp = r.json()
    if mode == "general":
        return resp["glossary"]
    elif mode == "customized":
        return resp["customized_glossary"]


def get_metadata(self, format="markdown", export_path=None, dev=False):
    """
    Gets the metadata on the defog servers.
    """
    r = requests.post(
        f"{self.base_url}/get_metadata",
        json={"api_key": self.api_key, "dev": dev},
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


def get_quota(self) -> Optional[Dict]:
    api_key = self.api_key
    response = requests.post(
        f"{self.base_url}/check_api_usage",
        json={"api_key": api_key},
    )
    # get status code and return None if not 200
    if response.status_code != 200:
        return None
    return response.json()


def update_golden_queries(
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

    r = requests.post(
        f"{self.base_url}/update_golden_queries",
        json={
            "api_key": self.api_key,
            "golden_queries": golden_queries,
            "scrub": scrub,
            "dev": dev,
        },
    )
    resp = r.json()
    print(
        "Golden queries have been received by the system, and will be processed shortly..."
    )
    print(
        "Once that is done, you should be able to see improved results for your questions."
    )
    return resp


def delete_golden_queries(
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
        r = requests.post(
            f"{self.base_url}/delete_golden_queries",
            json={"api_key": self.api_key, "all": True, "dev": dev},
        )
        resp = r.json()
        print("All golden queries have now been deleted.")
    else:
        if golden_queries is None:
            golden_queries = (
                pd.read_csv(golden_queries_path).fillna("").to_dict(orient="records")
            )

        r = requests.post(
            f"{self.base_url}/update_golden_queries",
            json={
                "api_key": self.api_key,
                "golden_queries": golden_queries,
            },
        )
        resp = r.json()
    return resp


def get_golden_queries(
    self, format: str = "csv", export_path: str = None, dev: bool = False
):
    """
    Gets the golden queries on the defog servers.
    """
    r = requests.post(
        f"{self.base_url}/get_golden_queries",
        json={
            "api_key": self.api_key,
            "dev": dev,
        },
    )
    resp = r.json()
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
        with open(export_path, "w") as f:
            json.dump(resp, f, indent=4)
        print(f"{len(golden_queries)} golden queries exported to {export_path}")
        return golden_queries
    else:
        raise ValueError("format must be either 'csv' or 'json'.")
