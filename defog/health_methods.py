import requests


def check_golden_queries_coverage(self, dev: bool = False):
    """
    Check the number of tables and columns inside the metadata schema that are covered by the golden queries.
    """
    try:
        r = requests.post(
            f"{self.base_url}/get_golden_queries_coverage",
            json={"api_key": self.api_key, "dev": dev},
        )
        resp = r.json()
        return resp
    except Exception as e:
        return {"error": str(e)}


def check_md_valid(self, dev: bool = False):
    """
    Check if the metadata schema is valid.
    """
    try:
        r = requests.post(
            f"{self.base_url}/check_md_valid",
            json={"api_key": self.api_key, "db_type": self.db_type, "dev": dev},
        )
        resp = r.json()
        return resp
    except Exception as e:
        return {"error": str(e)}


def check_gold_queries_valid(self, dev: bool = False):
    """
    Check if the golden queries are valid, and can actually be executed on a given database without errors
    """
    r = requests.post(
        f"{self.base_url}/check_gold_queries_valid",
        json={"api_key": self.api_key, "db_type": self.db_type, "dev": dev},
    )
    resp = r.json()
    return resp


def check_glossary_valid(self, dev: bool = False):
    """
    Check if glossary is valid by verifying if all schema, table and column names referenced are present in the metadata.
    """
    r = requests.post(
        f"{self.base_url}/check_glossary_valid",
        json={"api_key": self.api_key, "dev": dev},
    )
    resp = r.json()
    return resp


def check_glossary_consistency(self, dev: bool = False):
    """
    Check if all logic in the glossary is consistent and coherent.
    """
    r = requests.post(
        f"{self.base_url}/check_glossary_consistency",
        json={"api_key": self.api_key, "dev": dev},
    )
    resp = r.json()
    return resp
