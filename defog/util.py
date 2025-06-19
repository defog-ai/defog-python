import os
from typing import List
import warnings

import asyncio


def parse_update(
    args_list: List[str], attributes_list: List[str], config_dict: dict
) -> dict:
    """
    Parse the arguments from args_list for each attribute in
    attributes_list, and update the config dictionary in place if present.

    Args:
        args_list (List[str]): The arguments list.
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
    Write out log messages to ~/.defog/logs to preserve
    more verbose error messages when debugging.

    Args:
        msg (str): The message to write.
    """
    log_file_path = os.path.expanduser("~/.defog/logs")

    try:
        if not os.path.exists(log_file_path):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, "a") as file:
            file.write(msg + "\n")
    except Exception:
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
            except Exception:
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


async def async_identify_categorical_columns(
    conn=None,  # a connection object for any database
    cur=None,  # a cursor object for any database
    table_name: str = "",
    rows: list = [],
    is_cursor_async: bool = False,
    db_type="",
    distinct_threshold: int = 10,
    character_length_threshold: int = 50,
):
    """
    Identify categorical columns in the table and return the top distinct values for each column.

    Args:
        conn (optional): Async connection for databases (e.g., asyncpg).
        cur (optional): Sync/async cursor object for database queries.
        table_name (str): The name of the table to analyze.
        rows (list): List of column info dictionaries (with keys like "column_name", "data_type").
        is_cursor_async (bool): Set True if using an async cursor.
        distinct_threshold (int): Max distinct values to classify a column as categorical.
        character_length_threshold (int): Max length of a string column to be considered categorical.
        This is a heuristic for pruning columns that might contain arbitrarily long strings like json / configs.

    Note:
        - The function requires one of conn or cur to be provided.

    Returns:
        rows (list): The updated list of dictionaries containing the column names, data types and top distinct values.
        The list is modified in-place.
    """
    print(
        f"Identifying categorical columns in {table_name}. This might take a while if you have many rows in your table."
    )

    async def run_query(query, params=None):
        if db_type == "snowflake":
            if params is not None:
                cur.execute_async(query, params)
            else:
                cur.execute_async(query)
            query_id = cur.sfqid
            while conn.is_still_running(conn.get_query_status(query_id)):
                await asyncio.sleep(1)
            return cur.fetchall()

        if conn:
            # If using an async connection like asyncpg
            return await conn.fetch(query, *params if params else ())
        elif cur:
            if is_cursor_async:
                # If using an async cursor (like aiomysql or others)
                await cur.execute(query, params)
                return await cur.fetchall()
            else:
                if params:
                    await asyncio.to_thread(cur.execute, query, params)
                else:
                    await asyncio.to_thread(cur.execute, query)
                return await asyncio.to_thread(cur.fetchall)

    for idx, row in enumerate(rows):
        if is_str_type(row["data_type"]):
            # get the total number of rows and number of distinct values in the table for this column
            column_name = row["column_name"]

            query = f"SELECT COUNT(*) FROM (SELECT DISTINCT {column_name} FROM {table_name} LIMIT 10000) AS temp;"

            result = await run_query(query)
            try:
                num_distinct_values = result[0][0]
            except Exception:
                num_distinct_values = 0
            if num_distinct_values <= distinct_threshold and num_distinct_values > 0:
                # get the top distinct_threshold distinct values
                query = f"""SELECT {column_name}, COUNT({column_name}) AS col_count FROM {table_name} GROUP BY {column_name} ORDER BY col_count DESC LIMIT %s;"""
                top_values = await run_query(query, (distinct_threshold,))
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
    DEPRECATED: This function relied on API endpoints that are no longer available.
    Feedback collection has been removed as part of the local-only refactor.
    """
    warnings.warn(
        "get_feedback is deprecated and no longer functional. "
        "Feedback collection has been removed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return


async def make_async_post_request(
    url: str, payload: dict, timeout=300, return_response_object=False
):
    """
    DEPRECATED: This function is no longer needed as API calls have been removed.
    """
    warnings.warn(
        "make_async_post_request is deprecated and will be removed in a future version. "
        "API calls have been replaced with local operations.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {"error": "API calls are no longer supported"}
