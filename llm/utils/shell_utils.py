"""
This module contains utilities to run shell operations namely:
remove files, remove folder, move file
"""
import os
import shutil
import glob
from pathlib import Path


def rm_file(path, regex=False):
    """
    This function deletes file or files in a path recursively.

    Args:
        path (str): Path of file / directory containing files.
        regex (bool, optional): Flag to remove files recursively. Defaults to False.
    """
    if regex:
        file_list = glob.glob(path, recursive=True)
    else:
        file_list = [path]
    for file in file_list:
        path = Path(file)
        if os.path.exists(path):
            print(f"Removing file : {path}")
            os.remove(path)


def rm_dir(path):
    """
    This function deletes a directory.

    Args:
        path (str): Path to directory.
    """
    path = Path(path)
    if os.path.exists(path):
        print(f"Deleting directory : {path}")
        shutil.rmtree(path)


def mv_file(src, dst):
    """
    This function moves a file from src to dst.

    Args:
        src (str): Source file path.
        dst (str): Destination file path.
    """
    shutil.move(src, dst)
