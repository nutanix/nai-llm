"""
This module contains utilities to run system operations.

Attributes:
    nvidia_smi_cmd (dict): Contains the nvidia-smi command in different operating systems.
"""
import os
import sys
from pathlib import Path

nvidia_smi_cmd = {
    "Windows": "nvidia-smi.exe",
    "Darwin": "nvidia-smi",
    "Linux": "nvidia-smi",
}


def check_if_path_exists(filepath: str, err: str = "", is_dir: bool = False) -> None:
    """
    This function checks if a given path exists.

    Args:
        filepath (str): Path to check.
        param (str, optional): Error message to print if path doesn't exists. Defaults to "".
        is_dir (bool, optional): Set to True if path is a directory, else False. Defaults to "".
    """
    if (not is_dir and not os.path.isfile(filepath)) or (
        is_dir and not os.path.isdir(filepath)
    ):
        print(f"Filepath does not exist {err} - {filepath}")
        sys.exit(1)


def create_folder_if_not_exists(path: str) -> None:
    """
    This function creates a dirctory if it doesn't already exist.

    Args:
        path (str): Path of the dirctory to create
    """
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path}")


def check_if_folder_empty(path: str) -> bool:
    """
    This function checks if a directory is empty.

    Args:
        path (str): Path of the dirctory to check.

    Returns:
        bool: True if directory is empty, False otherwise.
    """
    dir_items = os.listdir(path)
    return len(dir_items) == 0


def remove_suffix_if_starts_with(string: str, suffix: str) -> str:
    """
    This function removes a suffix of a string is it starts with a given suffix

    Args:
        string (str): String to check.
        suffix (str): Suffix to remove.

    Returns:
        str: String with the suffix removed
    """
    if string.startswith(suffix):
        return string[len(suffix) :]
    return string


def get_all_files_in_directory(directory):
    """
    This function provides a list of file names in a directory
    and its sub-directories
    Args:
        path (str): The path to the directory.
    Returns:
        ["file.txt", "sub-directory/file.txt"]
    """
    output = []
    directory_path = Path(directory)
    output = [
        str(file.relative_to(directory_path))
        for file in directory_path.rglob("*")
        if file.is_file()
    ]
    return output
