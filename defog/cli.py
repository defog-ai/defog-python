import os
import sys

import defog
from defog import Defog

USAGE_STRING = """
Usage: defog <command>

Available commands:
    init\t\tSetup defog credentials and your database connection
    gen\t\tGenerate schema from your database
    query\t\tRun a query
    quota\t\tCheck your API quota limits
    docs\t\tPrint documentation
"""


def main():
    if len(sys.argv) < 2:
        print(USAGE_STRING)
        sys.exit(1)
    if sys.argv[1] == "init":
        init()
    elif sys.argv[1] == "gen":
        # TODO
        raise NotImplementedError("gen not implemented yet")
    elif sys.argv[1] == "query":
        # TODO
        raise NotImplementedError("query not implemented yet")
    elif sys.argv[1] == "quota":
        # TODO
        raise NotImplementedError("quota not implemented yet")
    elif sys.argv[1] == "docs":
        # TODO
        raise NotImplementedError("docs not implemented yet")
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print(USAGE_STRING)
        sys.exit(1)


def init():
    # print welcome message
    print(
        "Welcome to \033[94mdefog.ai\033[0m!\n"
    )
    # check if .defog/connection.json exists
    # if it does, ask if user wants to overwrite
    # if it doesn't, create it
    home_dir = os.path.expanduser("~")
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
    
    # prompt user for defog api key if not in environment variable
    if os.environ.get("DEFOG_API_KEY"):
        print("We found your DEFOG_API_KEY in your environment variables. We'll use that.")
    else:
        print("Please enter your DEFOG_API_KEY. You can get it from https://defog.ai/account and creating an account:")
        api_key = input()
    # prompt user for db_type
    print("What database are you using? Available options are: " + ", ".join(defog.SUPPORTED_DB_TYPES))
    db_type = input()
    while db_type not in defog.SUPPORTED_DB_TYPES:
        print("Sorry, we don't support that database yet. Available options are: " + ", ".join(defog.SUPPORTED_DB_TYPES))
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
        print("Please enter your database password:")
        password = input()
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
    # save information above to .defog/connection.json
    df = defog.Defog(api_key=api_key, db_type=db_type, db_creds=db_creds)
    pass


if __name__ == "__main__":
    main()
