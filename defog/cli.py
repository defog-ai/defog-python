import datetime
import decimal
import getpass
import json
import os
import re
import shutil
import subprocess
import sys

import defog
from defog import Defog
from defog.util import parse_update

USAGE_STRING = """
Usage: defog <command>

Available commands:
    init\t\t\tSetup defog credentials and your database connection
    gen <table1> <table2>\tSpecify tables to generate schema for
    update <url>\t\tupdate schema (google sheets url) to defog
    query\t\t\tRun a query
    deploy <gcp|aws>\t\tDeploy a defog server as a cloud function
    check\t\t\tCheck if your database is suitable for defog (for prospective customers)
    quota\t\t\tCheck your API quota limits
    docs\t\t\tPrint documentation
"""

home_dir = os.path.expanduser("~")


def main():
    if len(sys.argv) < 2:
        print(USAGE_STRING)
        sys.exit(1)
    if sys.argv[1] == "init":
        init()
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "update":
        update()
    elif sys.argv[1] == "query":
        query()
    elif sys.argv[1] == "deploy":
        deploy()
    elif sys.argv[1] == "check":
        check_suitability()
    elif sys.argv[1] == "quota":
        quota()
    elif sys.argv[1] == "docs":
        # TODO
        raise NotImplementedError("docs not implemented yet")
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print(USAGE_STRING)
        sys.exit(1)


def init():
    """
    Initialize defog by creating a config file at ~/.defog/connection.json
    """
    # print welcome message
    print("Welcome to \033[94mdefog.ai\033[0m!\n")
    # check if .defog/connection.json exists
    # if it does, ask if user wants to overwrite
    # if it doesn't, create it
    filepath = os.path.join(home_dir, ".defog", "connection.json")
    if os.path.exists(filepath):
        print(
            "It looks like you've already initialized defog. Do you want to overwrite your existing configuration? (y/n)"
        )
        overwrite = input()
        if overwrite.lower() != "y":
            print("We'll keep your existing config. No changes were made.")
            sys.exit(0)
        else:
            print("We'll overwrite a new config file at ~/.defog/connection.json")
    else:
        print("We'll create a new config file at ~/.defog/connection.json")
        if not os.path.exists(os.path.join(home_dir, ".defog")):
            os.mkdir(os.path.join(home_dir, ".defog"))

    # prompt user for defog api key if not in environment variable
    if os.environ.get("DEFOG_API_KEY"):
        print(
            "We found your DEFOG_API_KEY in your environment variables. We'll use that."
        )
        api_key = os.environ.get("DEFOG_API_KEY")
    else:
        api_key = getpass.getpass(
            prompt="Please enter your DEFOG_API_KEY. You can get it from https://defog.ai/account and creating an account:"
        )
    # prompt user for db_type
    print(
        "What database are you using? Available options are: "
        + ", ".join(defog.SUPPORTED_DB_TYPES)
    )
    db_type = input().lower()
    while db_type not in defog.SUPPORTED_DB_TYPES:
        print(
            "Sorry, we don't support that database yet. Available options are: "
            + ", ".join(defog.SUPPORTED_DB_TYPES)
        )
        db_type = input().lower()
    # depending on db_type, prompt user for appropriate db_creds
    if db_type == "postgres" or db_type == "redshift":
        print("Please enter your database host:")
        host = input()
        print("Please enter your database port:")
        port = input()
        print("Please enter your database name:")
        database = input()
        print("Please enter your database user:")
        user = input()
        password = getpass.getpass(prompt="Please enter your database password:")
        db_creds = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

    elif db_type == "mysql":
        print("Please enter your database host:")
        host = input()
        print("Please enter your database name:")
        database = input()
        print("Please enter your database user:")
        user = input()
        print("Please enter your database password:")
        password = input()
        db_creds = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
        }
    elif db_type == "snowflake":
        print("Please enter your database account:")
        account = input()
        print("Please enter your database warehouse:")
        warehouse = input()
        print("Please enter your database user:")
        user = input()
        print("Please enter your database password:")
        password = input()
        db_creds = {
            "account": account,
            "warehouse": warehouse,
            "user": user,
            "password": password,
        }
    elif db_type == "mongo" or db_type == "sqlserver":
        print("Please enter your database connection string:")
        connection_string = input()
        db_creds = {
            "connection_string": connection_string,
        }
    elif db_type == "bigquery":
        print("Please enter your bigquery json key's path:")
        json_key_path = input()
        db_creds = {
            "json_key_path": json_key_path,
        }

    # write to filepath and print confirmation
    with open(filepath, "w") as f:
        data = {"api_key": api_key, "db_type": db_type, "db_creds": db_creds}
        json.dump(data, f, indent=4)
    print(f"Your configuration has been saved to {filepath}.")

    # prompt user for tables that they would like to register
    print("We're going to register your tables' schema with defog.")
    print(
        "Please enter the names of the tables you would like to register, separated by a space:"
    )
    table_names = input()
    table_name_list = re.split(r"\s+", table_names.strip())
    # if input is empty, exit
    if table_name_list == [""]:
        print("No tables were registered. Exiting.")
        sys.exit(0)
    else:
        df = defog.Defog(api_key=api_key, db_type=db_type, db_creds=db_creds)
        gsheets_url = df.generate_db_schema(table_name_list)
        print("Your schema has been generated and is available at:\n")
        print(f"\033[1m{gsheets_url}\033[0m.\n")
        print(
            "If you do modify the schema in the link provided, please run `defog update <url>` to update the updated schema."
        )


def gen():
    """
    Generate a schema for a list of tables and print the link to the schema.
    """
    df = defog.Defog()  # load config from .defog/connection.json
    if len(sys.argv) < 3:
        print(
            "defog gen requires a list of tables to generate. Please enter the names of the tables whose schema you would like to generate, separated by a space:"
        )
        table_names = input()
        table_name_list = re.split(r"\s+", table_names.strip())
    else:
        table_name_list = sys.argv[2:]
    gsheets_url = df.generate_db_schema(table_name_list)
    print("Your schema has been generated and is available at:\n")
    print(f"\033[1m{gsheets_url}\033[0m.\n")
    print(
        "If you do modify the schema in the link provided, please run `defog update <url>` to update the updated schema."
    )


def update():
    """
    Update the schema in defog with a new schema using the url provided.
    """
    # check for 3rd arg (url), if not there, prompt user for url
    if len(sys.argv) < 3:
        print(
            "defog update requires a google sheets url. Please enter the url of the google sheets document you would like to update:"
        )
        gsheets_url = input()
    else:
        gsheets_url = sys.argv[2]
    # load config from .defog/connection.json
    df = defog.Defog()
    # upload schema to defog
    resp = df.update_db_schema(gsheets_url)
    if resp["status"] == "success":
        print("Your schema has been updated. You're ready to start querying!")
    else:
        print("There was an error updating your schema:")
        print(resp)
        print("Please try again, or contact us at founders@defog.ai")


def query():
    """
    Run a query and print the results alongside the generated SQL query.
    """
    df = defog.Defog()  # load config from .defog/connection.json
    if len(sys.argv) < 3:
        print("defog query requires a query. Please enter your query:")
        query = input()
    else:
        query = sys.argv[2]
    while query != "e":
        resp = df.run_query(query, retries=3)
        if not resp["ran_successfully"]:
            print(f"Your query did not run successfully. Please try again.")
        else:
            print("Your question generated the following query:\n")
            print(f"\033[1m{resp['query_generated']}\033[0m\n")
            print("Results:\n")
            # print results in tabular format using 'columns' and 'data' keys
            try:
                print_table(resp["columns"], resp["data"])
            except:
                print(resp)
        query = input("Enter another query, or type 'e' to exit: ")


def deploy():
    """
    Deploy a cloud function that can be used to run queries.
    """
    # check args for gcp or aws
    if len(sys.argv) < 3:
        print("defog deploy requires a cloud provider. Please enter 'gcp' or 'aws':")
        cloud_provider = input().lower()
    else:
        cloud_provider = sys.argv[2].lower()

    # load config from .defog/connection.json
    df = defog.Defog()

    if cloud_provider == "gcp":
        # base64 encode defog credentials for ease of passing around in cli
        creds64_str = df.to_base64_creds()
        source_path = os.path.join(defog.__path__[0], "gcp")
        cmd = [
            "gcloud",
            "functions",
            "deploy",
            "defog-gcp",
            "--runtime",
            "python310",
            "--region",
            "us-central1",
            "--source",
            source_path,
            "--entry-point",
            "defog_query_http",
            "--max-instances",
            "1",
            "--set-env-vars",
            f"DEFOG_CREDS_64={creds64_str}",
            "--trigger-http",
            "--gen2",
            "--allow-unauthenticated",
        ]
        try:
            print("executing gcloud command...")
            subprocess.check_call(cmd)
            print("gcloud command executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error deploying Cloud Function:\n{e}")
    elif cloud_provider == "aws":
        # base64 encode defog credentials for ease of passing around in cli
        creds64_str = df.to_base64_creds()
        # get base config from defog package, add env vars
        base_config_path = os.path.join(defog.__path__[0], "aws", "base_config.json")
        with open(base_config_path, "r") as f:
            chalice_config = json.load(f)
        chalice_config["environment_variables"] = {"DEFOG_CREDS_64": creds64_str}
        chalice_config = parse_update(
            sys.argv[3:], ["app_name", "version"], chalice_config
        )
        aws_path = os.path.join(home_dir, ".defog", "aws")
        chalice_path = os.path.join(aws_path, ".chalice")
        if not os.path.exists(chalice_path):
            print(f"creating {chalice_path}")
            os.makedirs(chalice_path)
        chalice_config_path = os.path.join(chalice_path, "config.json")
        # save to .defog/aws/.chalice/config.json
        with open(chalice_config_path, "w") as f:
            json.dump(chalice_config, f)
        # copy over app.py and requirements.txt to .defog/aws
        app_path = os.path.join(defog.__path__[0], "aws", "app.py")
        req_path = os.path.join(defog.__path__[0], "aws", "requirements.txt")
        shutil.copy(app_path, aws_path)
        shutil.copy(req_path, aws_path)

        # deploy with chalice
        try:
            print("deploying with Chalice...")
            os.chdir(aws_path)
            subprocess.check_call(["chalice", "deploy"])
            os.chdir("../..")
            print("deployed aws lambda successfully with Chalice.")
            print(f"You can find the chalice deployment artifacts in {aws_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error deploying with Chalice:\n{e}")
    else:
        raise ValueError("Cloud provider must be 'gcp' or 'aws'.")


def check_suitability():
    """
    Check if tables in customer's schema are supported by our product.
    """
    df = defog.Defog()  # Note that api key can be invalid here for presales purposes
    # parse list of tables from args
    if len(sys.argv) < 3:
        print(
            "defog check-suitability requires a list of tables. Please enter your table names, separated by a space:"
        )
        table_names = input()
        table_name_list = re.split(r"\s+", table_names.strip())
    else:
        table_name_list = sys.argv[2:]
    df.check_db_suitability(tables=table_name_list)  # prints out messages to stdout


def quota():
    """
    Check your current usage and quota.
    """
    df = defog.Defog()
    resp = df.get_quota()
    if resp["ran_successfully"]:
        if resp["premium"]:
            print(f"You are currently on the premium plan with unrestricted usage.")
            print(f"Your current usage is {resp['queries_made']} queries.")
        else:
            print(
                f"You are currently on the free plan with {100-resp['queries_made']} queries remaining for the month."
            )
            print(f"Your current usage is {resp['queries_made']} queries.")
    else:
        print(f"Failed to get quota")


# helper function to format different field types into strings
def to_str(field) -> str:
    if isinstance(field, str):
        return field
    elif isinstance(field, int):
        return str(field)
    elif isinstance(field, float):
        return str(field)
    elif isinstance(field, datetime.datetime):
        return field.strftime("%Y-%m-%d")
    elif isinstance(field, datetime.date):
        return field.strftime("%Y-%m-%d")
    elif isinstance(field, datetime.timedelta):
        return str(field)
    elif isinstance(field, datetime.time):
        return field.strftime("%H:%M:%S")
    elif isinstance(field, list):
        return str(field)
    elif isinstance(field, dict):
        return str(field)
    elif isinstance(field, bool):
        return str(field)
    elif isinstance(field, decimal.Decimal):
        return str(field)
    elif field is None:
        return "NULL"
    else:
        raise ValueError(f"Unknown type: {type(field)}")


# helper function to print results in tabular format
def print_table(columns, data):
    # Calculate the maximum width of each column, including headers
    data_header = data + [tuple(columns)]
    column_widths = [
        max(len(to_str(row[i])) for row in data_header) for i in range(len(columns))
    ]

    # Print the table headers
    for i, column in enumerate(columns):
        print(column.ljust(column_widths[i]), end=" | ")
    print()

    # Print the table divider
    for i, column_width in enumerate(column_widths):
        print("-" * column_width, end="-+-" if i < len(column_widths) - 1 else "-|\n")

    # Print the table data
    for row in data:
        for i, value in enumerate(row):
            print(to_str(value).ljust(column_widths[i]), end=" | ")
        print()


if __name__ == "__main__":
    main()
