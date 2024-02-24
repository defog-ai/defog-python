import requests
import pandas as pd


def update_db_schema(self, path_to_csv):
    """
    Update the DB schema via a CSV
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


def get_quota(self) -> str:
    headers = {
        "Authorization": f"Bearer {self.api_key}",
    }
    response = requests.get(
        f"{self.base_url}/quota",
        headers=headers,
    )
    return response.json()


def update_golden_queries(
    self, golden_queries: dict = None, golden_queries_path: str = None
):
    """
    Updates the golden queries on the defog servers.
    :param golden_queries: The golden queries to be used.
    :param golden_queries_path: The path to the golden queries CSV.
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


def get_golden_queries(self, format="csv", export_path=None):
    """
    Gets the golden queries on the defog servers.
    """
    r = requests.post(
        f"{self.base_url}/get_golden_queries",
        json={"api_key": self.api_key},
    )
    resp = r.json()
    if format == "csv":
        if export_path is None:
            export_path = "golden_queries.csv"
        pd.DataFrame(resp["golden_queries"]).to_csv(export_path, index=False)
        print(f"Golden queries exported to {export_path}")
        return True
    elif format == "json":
        return resp["golden_queries"]
    else:
        raise ValueError("format must be either 'csv' or 'json'.")
