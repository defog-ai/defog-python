import os
from typing import List

from prompt_toolkit import prompt
import requests


def parse_update(
    args_list: List[str], attributes_list: List[str], config_dict: dict
) -> dict:
    """
    Parse the command line arguments from args_list for each attribute in
    attributes_list, and update the config dictionary in place if present.

    Args:
        args_list (List[str]): The command line arguments.
        config_dict (dict): The given config dictionary.

    Returns:
        dict: The updated config dictionary.
    """
    if len(args_list) % 2 != 0:
        print("Error: args_list must be given in pairs.")
        print(f"{args_list} is not a valid args_list.")
    while len(args_list) >= 2 and len(args_list) % 2 == 0:
        args_name = args_list[0][2:]  # remove the '--'
        if args_name in attributes_list:
            args_value = args_list[1]
            config_dict[args_name] = args_value
        else:
            print(f"Error: {args_name} is not a valid argument.")
            print(f"Valid arguments are: {attributes_list}")
        args_list = args_list[2:]
    return config_dict


def write_logs(msg: str) -> None:
    """
    Write out log messages to ~/.defog/logs to avoid bloating cli output,
    while still preserving more verbose error messages when debugging.

    Args:
        msg (str): The message to write.
    """
    log_file_path = os.path.expanduser("~/.defog/logs")

    try:
        if not os.path.exists(log_file_path):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, "a") as file:
            file.write(msg + "\n")
    except Exception as e:
        pass


def identify_categorical_columns(
    cur,  # a cursor object for any database
    table_name: str,
    rows: list,
):
    """
    Identify categorical columns in the table and return the top 10 distinct values for each column.

    Args:
        cur (cursor): A cursor object for any database.
        table_name (str): The name of the table.
        rows (list): A list of dictionaries containing the column names and data types.

    Returns:
        rows (list): The updated list of dictionaries containing the column names, data types and top 10 distinct values.
    """
    # loop through each column, look at whether it is a string column, and then determine if it might be a categorical variable
    # if it is a categorical variable, then we want to get the distinct values and their counts
    # we will then send this to the defog servers so that we can generate a column description
    # for each categorical variable
    print(
        f"Identifying categorical columns in {table_name}. This might take a while if you have many rows in your table."
    )
    for idx, row in enumerate(rows):
        if row["data_type"].lower() in [
            "character varying",
            "text",
            "character",
            "varchar",
            "char",
            "string",
        ]:
            # get the total number of rows and number of distinct values in the table for this column
            cur.execute(
                f"SELECT COUNT({row['column_name']}) as tot_count, COUNT(DISTINCT {row['column_name']}) AS unique_count FROM {table_name};"
            )
            total_rows, num_distinct_values = cur.fetchone()

            if num_distinct_values <= 10:
                # get the top 10 distinct values
                cur.execute(
                    f"SELECT {row['column_name']}, COUNT({row['column_name']}) AS col_count FROM {table_name} GROUP BY {row['column_name']} ORDER BY col_count DESC LIMIT 10;"
                )
                top_values = cur.fetchall()
                top_values = [i[0] for i in top_values if i[0] is not None]
                rows[idx]["top_values"] = top_values
            print(f"Identified {row['column_name']} as a likely categorical column.")
    return rows


def get_feedback(api_key: str, db_type: str, user_question: str, sql_generated: str):
    """
    Get feedback from the user on whether the query was good or bad, and why.
    """
    feedback = prompt(
        "Did Defog answer your question well? Just hit enter to skip (y/n):\n"
    )
    while feedback not in ["y", "n", ""]:
        feedback = prompt("Please enter y or n:\n")
    if feedback == "":
        # user skipped feedback
        return
    # get explanation for negative feedback
    if feedback == "n":
        feedback_text = prompt(
            "Could you tell us why this was a bad query? This will help us improve the model for you. Just hit enter if you want to leave this blank.\n"
        )
    else:
        feedback_text = ""
    try:
        data = {
            "api_key": api_key,
            "feedback": "good" if feedback == "y" else "bad",
            "db_type": db_type,
            "question": user_question,
            "query": sql_generated,
        }
        if feedback_text != "":
            data["feedback_text"] = feedback_text
        requests.post(
            "https://api.defog.ai/feedback",
            json=data,
            timeout=1,
        )
        if feedback == "y":
            print("Thank you for the feedback!")
        else:
            print(
                "Thank you for the feedback! We retrain our models every week, and you should see much better performance on these kinds of queries in another week.\n"
            )
            print(
                f"If you continue to get these errors, please consider updating the metadata in your schema by editing the google sheet generated and running `defog update <url>`, or by updating your glossary.\n"
            )
    except Exception as e:
        write_logs(f"Error in get_feedback:\n{e}")
        pass
