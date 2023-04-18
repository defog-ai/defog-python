import sys

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
        "Welcome to \033[94mdefog.ai\033[0m!"
        "Let's get started by setting up your credentials and database connection."
    )

    # check args and print help if needed

    # check if .defog/connection.json exists

    # prompt user for defog api key if not in environment variable

    # prompt user for db_type

    # depending on db_type, prompt user for appropriate db_creds

    # save information above to .defog/connection.json

    pass


if __name__ == "__main__":
    main()
