import os
from typing import List


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

# writes out error message to ~/.defog/logs to avoid bloating cli output
# while still preserving more verbose error messages when debugging
def write_err(err_msg: str) -> None:
    """
    Write the error message to the error log.

    Args:
        err_msg (str): The error message to write.
    """
    log_file_path = os.path.expanduser("~/.defog/logs")
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, "a") as file:
        file.write(err_msg + "\n")