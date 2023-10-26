"""
This module runs pytest tests for run.sh file.

Attributes:
    INPUT_PATH: Path to input data folder.
"""
import os
import subprocess
import pytest
import download
from tests.test_download import (
    MODEL_STORE,
    MODEL_NAME,
    set_generate_args,
    custom_model_restore,
    custom_model_setup,
    test_case_1,
)

INPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "qa"
)


def test_generate_mar():
    """
    This function calls the default testcase from test_download.py
    This is done to generate the MAR file used in the rest of the
    tests.
    """
    test_case_1()


def get_run_cmd(
    model_name=MODEL_NAME, model_store=MODEL_STORE, input_path="", repo_version=""
):
    """
    This function is used to generate the bash command to be run using given
    parameters

    Args:
        model_name (str, optional): Name of the model. Defaults to MODEL_NAME.
        model_store (str, optional): Model store location. Defaults to MODEL_STORE.
        input_path (str, optional): Path to input data folder. Defaults to "".
        repo_version (str, optional): Repository version of the model. Defaults to "".

    Returns:
        list(str): Bash command converted to list of strings spilt by spaces.
    """
    cmd = "bash run.sh"
    if model_name:
        cmd = f"{cmd} -n {model_name}"
    if model_store:
        cmd = f"{cmd} -a {model_store}"
    if input_path:
        cmd = f"{cmd} -d {input_path}"
    if repo_version:
        cmd = f"{cmd} -v {repo_version}"
    return cmd.split()


def test_run_case_1():
    """
    This function tests the default GPT2 model with input path.
    Expected result: Success.
    """
    process = subprocess.run(get_run_cmd(input_path=INPUT_PATH), check=False)
    assert process.returncode == 0


def test_run_case_2():
    """
    This function tests the default GPT2 model without input path.
    Expected result: Success.
    """
    process = subprocess.run(get_run_cmd(), check=False)
    assert process.returncode == 0


def test_run_case_3():
    """
    This function tests missing model name.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_name=""), check=False)
    assert process.returncode == 1


def test_run_case_4():
    """
    This function tests wrong model name.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_name="wrong_model_name"), check=False)
    assert process.returncode == 1


def test_run_case_5():
    """
    This function tests missing model store.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_store=""), check=False)
    assert process.returncode == 1


def test_run_case_6():
    """
    This function tests wrong model store.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_store="wrong_model_store"), check=False)
    assert process.returncode == 1


def test_run_case_7():
    """
    This function tests wrong input path.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(input_path="wrong_input_path"), check=False)
    assert process.returncode == 1


def test_run_case_8():
    """
    This function tests valid repo version.
    Expected result: Success.
    """
    process = subprocess.run(
        get_run_cmd(repo_version="11c5a3d5811f50298f278a704980280950aedb10"),
        check=False,
    )
    assert process.returncode == 0


def test_run_case_9():
    """
    This function tests invalid repo version.
    Expected result: Failure.
    """
    process = subprocess.run(
        get_run_cmd(repo_version="invalid_repo_version"), check=False
    )
    assert process.returncode == 1


def test_run_case_10():
    """
    This function tests custom model with input folder.
    Expected result: Success.
    """
    custom_model_setup()
    args = set_generate_args()
    args.no_download = True
    try:
        download.run_script(args)
    except SystemExit:
        assert False

    process = subprocess.run(get_run_cmd(input_path=INPUT_PATH), check=False)
    assert process.returncode == 0

    custom_model_restore()
    process = subprocess.run(["python3", "cleanup.py"], check=False)


# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])
