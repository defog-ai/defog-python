import os
from setuptools import find_packages, setup

extras = {
    "postgres": ["psycopg2-binary"],
    "mysql": ["mysql-connector-python"],
    "snowflake": ["snowflake-connector-python"],
    "bigquery": ["google-cloud-bigquery"],
    "redshift": ["psycopg2-binary"],
    "databricks": ["databricks-sql-connector"],
    "sqlserver": ["pyodbc"],
}


setup(
    name="defog",
    packages=find_packages(),
    version="0.72.4",
    description="Defog is a Python library that helps you generate data queries from natural language questions.",
    author="Full Stack Data Pte. Ltd.",
    license="MIT",
    install_requires=[
        "httpx>=0.28.1",
        "psycopg2-binary>=2.9.5",
        "prompt-toolkit>=3.0.38",
        "tqdm",
        "pydantic",
        "portalocker>=3.2.0",
    ],
    author_email="founders@defog.ai",
    url="https://github.com/defog-ai/defog-python",
    long_description="Defog is a Python library that helps you generate data queries from natural language questions.",
    long_description_content_type="text/markdown",
    extras_require=extras,
)
