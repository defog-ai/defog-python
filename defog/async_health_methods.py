from defog.util import make_async_post_request


async def check_golden_queries_coverage(self, dev: bool = False):
    """
    Check the number of tables and columns inside the metadata schema that are covered by the golden queries.
    """
    url = f"{self.base_url}/get_golden_queries_coverage"
    payload = {"api_key": self.api_key, "dev": dev}
    return await make_async_post_request(url, payload)


async def check_md_valid(self, dev: bool = False):
    """
    Check if the metadata schema is valid.
    """
    url = f"{self.base_url}/check_md_valid"
    payload = {"api_key": self.api_key, "db_type": self.db_type, "dev": dev}
    return await make_async_post_request(url, payload)


async def check_gold_queries_valid(self, dev: bool = False):
    """
    Check if the golden queries are valid and can be executed on a given database without errors.
    """
    url = f"{self.base_url}/check_gold_queries_valid"
    payload = {"api_key": self.api_key, "db_type": self.db_type, "dev": dev}
    return await make_async_post_request(url, payload)


async def check_glossary_valid(self, dev: bool = False):
    """
    Check if the glossary is valid by verifying if all schema, table, and column names referenced are present in the metadata.
    """
    url = f"{self.base_url}/check_glossary_valid"
    payload = {"api_key": self.api_key, "dev": dev}
    return await make_async_post_request(url, payload)


async def check_glossary_consistency(self, dev: bool = False):
    """
    Check if all logic in the glossary is consistent and coherent.
    """
    url = f"{self.base_url}/check_glossary_consistency"
    payload = {"api_key": self.api_key, "dev": dev}
    return await make_async_post_request(url, payload)
