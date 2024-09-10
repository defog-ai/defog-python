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


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


next_static_files = package_files("defog/static")

setup(
    name="defog",
    packages=find_packages(),
    package_data={"defog": ["gcp/*", "aws/*"] + next_static_files},
    version="0.65.11",
    description="Defog is a Python library that helps you generate data queries from natural language questions.",
    author="Full Stack Data Pte. Ltd.",
    license="MIT",
    install_requires=[
        "requests>=2.28.2",
        "psycopg2-binary>=2.9.5",
        "prompt-toolkit>=3.0.38",
        "fastapi",
        "uvicorn",
        "tqdm",
        "pwinput",
    ],
    entry_points={
        "console_scripts": [
            "defog=defog.cli:main",
        ],
    },
    author_email="founders@defog.ai",
    url="https://github.com/defog-ai/defog-python",
    long_description="Defog is a Python library that helps you generate data queries from natural language questions.",
    long_description_content_type="text/markdown",
    extras_require=extras,
)
