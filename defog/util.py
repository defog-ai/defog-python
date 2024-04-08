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


def is_str_type(data_type: str) -> bool:
    """
    Check if the given data_type is a string type.

    Args:
        data_type (str): The data_type to check.

    Returns:
        bool: True if the data_type is a string type, False otherwise.
    """
    return data_type.lower().strip() in {
        "character varying",
        "text",
        "character",
        "varchar",
        "char",
        "string",
    }


def identify_categorical_columns(
    cur,  # a cursor object for any database
    table_name: str,
    rows: list,
    distinct_threshold: int = 10,
    character_length_threshold: int = 50,
):
    """
    Identify categorical columns in the table and return the top distinct values for each column.

    Args:
        cur (cursor): A cursor object for any database. This cursor should support the following methods:
            - execute(sql, params)
            - fetchone()
            - fetchall()
        table_name (str): The name of the table.
        rows (list): A list of dictionaries containing the column names and data types.a
        distinct_threshold (int): The threshold for the number of distinct values in a column to be considered categorical.
        character_length_threshold (int): The threshold for the maximum length of a string column to be considered categorical.
        This is a heuristic for pruning columns that might contain arbitrarily long strings like json / configs.

    Returns:
        rows (list): The updated list of dictionaries containing the column names, data types and top distinct values.
        The list is modified in-place.
    """
    # loop through each column, look at whether it is a string column, and then determine if it might be a categorical variable
    # if it is a categorical variable, then we want to get the distinct values and their counts
    # we will then send this to the defog servers so that we can generate a column description
    # for each categorical variable
    print(
        f"Identifying categorical columns in {table_name}. This might take a while if you have many rows in your table."
    )
    for idx, row in enumerate(rows):
        if is_str_type(row["data_type"]):
            # get the total number of rows and number of distinct values in the table for this column
            column_name = row["column_name"]

            cur.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT {column_name} FROM {table_name} LIMIT 10000) AS temp;"
            )
            try:
                num_distinct_values = cur.fetchone()[0]
            except Exception as e:
                num_distinct_values = 0
            if num_distinct_values <= distinct_threshold and num_distinct_values > 0:
                # get the top distinct_threshold distinct values
                cur.execute(
                    f"""SELECT {column_name}, COUNT({column_name}) AS col_count FROM {table_name} GROUP BY {column_name} ORDER BY col_count DESC LIMIT %s;""",
                    (distinct_threshold,),
                )
                top_values = cur.fetchall()
                top_values = [i[0] for i in top_values if i[0] is not None]
                rows[idx]["top_values"] = ",".join(sorted(top_values))
                print(
                    f"Identified {column_name} as a likely categorical column. The unique values are: {top_values}"
                )
    return rows


def get_feedback(
    api_key: str, db_type: str, user_question: str, sql_generated: str, base_url: str
):
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
            f"{base_url}/feedback",
            json=data,
            timeout=1,
        )
        if feedback == "y":
            print("Thank you for the feedback!")
        elif feedback == "n":
            data = {
                "api_key": api_key,
                "question": user_question,
                "sql_generated": sql_generated,
                "error": feedback_text,
            }
            print(
                "Thank you for the feedback, let us see how can we improve this for you...\n\nGenerating an automated assessment for improving the metadata and glossary. This can take up to 60 seconds. Please be patient...\n"
            )
            response = requests.post(
                f"{base_url}/reflect_on_error",
                json=data,
            )

            if response.status_code == 200:
                response_dict = response.json()
                feedback = response_dict.get("feedback")
                if feedback:
                    print(f"Here is our automated assessment:\n{feedback}\n")
                # 1) validate and update glossary
                instruction_set = response_dict.get("instruction_set")
                if instruction_set:
                    print(
                        f"We came up with the following additions for improving your glossary:\n{instruction_set}"
                    )
                    add_to_glossary = prompt(
                        "If you would like to add these suggestions to your glossary, please enter 'y'. If you would like to amend it, just type in the new glossary and hit enter. Otherwise, enter 'n'.\n"
                    )
                    if add_to_glossary == "y":
                        md_resp = requests.post(
                            f"{base_url}/get_metadata",
                            json={"api_key": api_key},
                        )
                        md_resp_dict = md_resp.json()
                        glossary_current = md_resp_dict.get("glossary", "")
                        glossary_updated = glossary_current + "\n" + instruction_set
                        requests.post(
                            f"{base_url}/update_glossary",
                            json={
                                "api_key": api_key,
                                "glossary": glossary_updated,
                            },
                        )
                        print("Glossary updated successfully.\n")
                    elif add_to_glossary != "n":
                        md_resp = requests.post(
                            f"{base_url}/get_metadata",
                            json={"api_key": api_key},
                        )
                        md_resp_dict = md_resp.json()
                        glossary_current = md_resp_dict.get("glossary", "")
                        glossary_updated = glossary_current + "\n" + add_to_glossary
                        requests.post(
                            f"{base_url}/update_glossary",
                            json={
                                "api_key": api_key,
                                "glossary": glossary_updated,
                            },
                        )
                        print("Glossary updated successfully.\n")
                    else:
                        print("Glossary not updated.\n")

                # 2) validate and update column descriptions in metadata
                new_column_descriptions = response_dict.get("column_descriptions")
                if new_column_descriptions:
                    print(
                        f"We came up with the following suggestions for improving your column descriptions:\n{new_column_descriptions}"
                    )
                    # get original metadata
                    r = requests.post(
                        f"{base_url}/get_metadata",
                        json={"api_key": api_key},
                    )
                    resp = r.json()
                    md = resp.get("table_metadata", {})
                    # we will be editing md in place
                    column_changed = False
                    for new_column_description in new_column_descriptions:
                        table_name = new_column_description.get("table_name")
                        column_name = new_column_description.get("column_name")
                        description = new_column_description.get("description")
                        if table_name in md:
                            for column in md[table_name]:
                                if column.get("column_name") == column_name:
                                    print(
                                        f"\nCurrent description for {column_name}: {column.get('column_description')}"
                                    )
                                    print(
                                        f"Suggested description for {column_name}: {description}"
                                    )
                                    replace = prompt(
                                        "Would you like to replace this description with our suggestion? Please enter 'y' to replace, or your own description to amend. Otherwise, enter 'n' to skip.\n"
                                    )
                                    if replace == "y":
                                        column["column_description"] = description
                                        print("Updated description.")
                                        column_changed = True
                                        break
                                    elif replace != "n":
                                        column["column_description"] = replace
                                        print("Updated description.")
                                        column_changed = True
                                        break
                                    else:
                                        print("Description not updated.")
                                        break
                    if column_changed:
                        requests.post(
                            f"{base_url}/update_metadata",
                            json={
                                "api_key": api_key,
                                "table_metadata": new_column_description,
                                "db_type": db_type,
                            },
                        )
                        print("Metadata updated successfully.\n")
                    else:
                        print("No metadata changes to update.\n")
                # 3) validate and update reference_queries
                new_reference_queries = response_dict.get("reference_queries")
                if (
                    isinstance(new_reference_queries, list)
                    and len(new_reference_queries) > 0
                ):
                    reference_queries_to_add = []
                    print(
                        f"We came up with the following suggestions for adding as your reference queries:"
                    )
                    for new_reference_query in new_reference_queries:
                        question = new_reference_query.get("question")
                        sql = new_reference_query.get("sql")
                        print(f"Question: {question}\nSQL: {sql}")
                        update_reference_queries = prompt(
                            "Would you like to add this as one of your reference queries? Please hit 'y' to add, or anything else to skip to the next suggestion.\n"
                        )
                        if update_reference_queries == "y":
                            reference_queries_to_add.append(new_reference_query)
                    if len(reference_queries_to_add) > 0:
                        r = requests.post(
                            f"{base_url}/update_golden_queries",
                            json={
                                "api_key": api_key,
                                "golden_queries": reference_queries_to_add,
                                "scrub": True,
                            },
                        )
                        if r.status_code == 200:
                            print(
                                f"{len(reference_queries_to_add)} reference queries added successfully."
                            )
                        else:
                            print("Reference queries not updated.")
                    else:
                        print("No reference queries to update.")
                    print()
            else:
                print("There was an error in getting suggestions. Our apologies!")

    except Exception as e:
        write_logs(f"Error in get_feedback:\n{e}")
        pass
