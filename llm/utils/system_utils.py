"""
system_utils
This module contains utilities to run system operations.

Attributes:
    nvidia_smi_cmd (dict): Contains the nvidia-smi command in different operating systems.
"""
import os
import platform
import subprocess
import sys

nvidia_smi_cmd = {
    "Windows": "nvidia-smi.exe",
    "Darwin": "nvidia-smi",
    "Linux": "nvidia-smi",
}


def is_gpu_instance():
    """
    is_gpu_instance
    This function checks if CUDA drivers are installed and GPU is present.

    Raises:
        exp: Exception caused if CUDA drivers are not installed (nvidia-smi not found).

    Returns:
        bool: True if CUDA drivers exist and GPU is present.
    """
    try:
        subprocess.check_output(nvidia_smi_cmd[platform.system()])
        print("\n## Nvidia GPU detected!")
        return True
    except subprocess.CalledProcessError as exp:
        print("\n## No Nvidia GPU in system!")
        raise exp


def check_if_path_exists(filepath, err="", is_dir=False):
    """
    check_if_path_exists
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


def create_folder_if_not_exists(path):
    """
    create_folder_if_not_exists
    This function creates a dirctory if it doesn't already exist.

    Args:
        path (str): Path of the dirctory to create
    """
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path}")


def check_if_folder_empty(path):
    """
    check_if_folder_empty
    This function checks if a directory is empty.

    Args:
        path (str): Path of the dirctory to check.

    Returns:
        bool: True if directory is empty, False otherwise.
    """
    dir_items = os.listdir(path)
    return len(dir_items) == 0


def remove_suffix_if_starts_with(string, suffix):
    """
    remove_suffix_if_starts_with
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
