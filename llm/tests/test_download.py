"""
This module runs pytest tests for download.py file.

Attributes:
    MODEL_NAME: Name of the model used for testing (gpt2).
    MODEL_PATH: Default path to model files.
    MODEL_STORE: Default model store location.
    MODEL_CONFIG_PATH: Path to model_config.json file.
    MODEL_TEMP_CONFIG_PATH: Path to backup model_config.json file.
"""
import os
import argparse
import shutil
import json
from pathlib import Path
import pytest
import download
from utils.shell_utils import copy_file

MODEL_NAME = "gpt2"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_path")
MODEL_STORE = os.path.join(os.path.dirname(__file__), "model_store")
MODEL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model_config.json"
)
MODEL_TEMP_CONFIG_PATH = os.path.join(MODEL_STORE, "temp_model_config.json")


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


def download_setup():
    """
    This function deletes and creates model store and
    model path directories.
    """
    rm_dir(MODEL_PATH)
    rm_dir(MODEL_STORE)
    os.makedirs(MODEL_PATH)
    os.makedirs(MODEL_STORE)


def cleanup_folders():
    """
    This function deletes model store and model path directories.
    """
    rm_dir(MODEL_PATH)
    rm_dir(MODEL_STORE)


def set_generate_args(
    model_name=MODEL_NAME,
    repo_version="",
    model_path=MODEL_PATH,
    mar_output=MODEL_STORE,
    handler_path="",
):
    """
    This function sets the arguments to run download.py.

    Args:
        repo_version (str, optional): Repository version of the model. Defaults to "".
        model_path (str, optional): Path to model files. Defaults to MODEL_PATH.
        mar_output (str, optional): Model store location. Defaults to MODEL_STORE.
        handler_path (str, optional): Path to Torchserve handler. Defaults to "".

    Returns:
        argparse.Namespace: Parameters to run download.py.
    """
    args = argparse.Namespace()
    args.model_name = model_name
    args.model_path = model_path
    args.mar_output = mar_output
    args.no_download = False
    args.repo_version = repo_version
    args.handler_path = handler_path
    args.debug = False
    args.hf_token = ""
    return args


def test_case_1():
    """
    This function tests the default GPT2 model.
    Expected result: Success.
    """
    download_setup()
    args = set_generate_args()
    try:
        result = download.run_script(args)
    except SystemExit:
        assert False
    else:
        assert result is True


def test_case_2():
    """
    This function tests wrong model store path.
    Expected result: Failure.
    """
    download_setup()
    args = set_generate_args(mar_output="wrong_model_store")
    try:
        download.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_case_3():
    """
    This function tests wrong model files path.
    Expected result: Failure.
    """
    download_setup()
    args = set_generate_args(model_path="wrong_model_path")
    try:
        download.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_case_4():
    """
    This function tests non empty model files path without skip download.
    Expected result: Failure.
    """
    download_setup()
    with open(os.path.join(MODEL_PATH, "non_empty.txt"), "w", encoding="UTF-8") as file:
        file.write("non empty text")
    args = set_generate_args()
    try:
        download.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_case_5():
    """
    This function tests invalid repo version.
    Expected result: Failure.
    """
    download_setup()
    args = set_generate_args(repo_version="invalid_repo_version")
    try:
        download.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_case_6():
    """
    This function tests valid repo version.
    Expected result: Success.
    """
    download_setup()
    args = set_generate_args(repo_version="e7da7f221d5bf496a48136c0cd264e630fe9fcc8")
    try:
        result = download.run_script(args)
    except SystemExit:
        assert False
    else:
        assert result is True


def test_case_7():
    """
    This function tests invalid handler path.
    Expected result: Failure.
    """
    download_setup()
    args = set_generate_args(handler_path="invalid_handler.py")
    try:
        download.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_case_8():
    """
    This function tests skip download without model files.
    Expected result: Failure.
    """
    download_setup()
    args = set_generate_args()
    args.no_download = True
    try:
        download.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_case_9():
    """
    This function tests if MAR file already exists.
    Expected result: Exits.
    """
    download_setup()
    args = set_generate_args()
    download.run_script(args)
    try:
        download.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_case_10():
    """
    This function tests skip download case.
    Expected result: Success.
    """
    download_setup()
    args = set_generate_args()
    download.run_script(args)

    # clear model store directory
    rm_dir(MODEL_STORE)
    os.makedirs(MODEL_STORE)
    args = set_generate_args()
    args.no_download = True
    try:
        result = download.run_script(args)
    except SystemExit:
        assert False
    else:
        assert result is True


def custom_model_setup():
    """
    This function is used to setup custom model case.
    It runs download.py to download model files and
    deletes the contents of 'model_config.json' after
    making a backup.
    """
    download_setup()
    args = set_generate_args()
    download.run_script(args)

    # creating a backup of original model_config.json
    copy_file(MODEL_CONFIG_PATH, MODEL_TEMP_CONFIG_PATH)
    with open(MODEL_CONFIG_PATH, "w", encoding="UTF-8") as file:
        json.dump({}, file)


def custom_model_restore():
    """
    This function restores the 'model_config.json' file
    and runs cleanup_folders function.
    """
    os.remove(MODEL_CONFIG_PATH)
    shutil.copyfile(MODEL_TEMP_CONFIG_PATH, MODEL_CONFIG_PATH)
    cleanup_folders()


def test_case_11():
    """
    This function tests the custom model case.
    This is done by clearing the 'model_config.json' and
    generating the 'GPT2' MAR file.
    Expected result: Success.
    """
    custom_model_setup()
    args = set_generate_args()
    args.no_download = True
    try:
        result = download.run_script(args)
    except SystemExit:
        assert False
    else:
        assert result is True
    custom_model_restore()


# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])
