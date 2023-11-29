"""
This module runs pytest tests for run.sh file.

Attributes:
    INPUT_PATH: Path to input data folder.
"""
import os
import subprocess
from typing import List
import json
import requests
import pytest
import download
from tests.test_download import (
    MODEL_STORE,
    MODEL_NAME,
    set_generate_args,
    custom_model_restore,
    custom_model_setup,
    test_default_generate_success,
)

INPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "qa"
)


def test_generate_mar_success() -> None:
    """
    This function calls the default testcase from test_download.py
    This is done to generate the MAR file used in the rest of the
    tests.
    """
    test_default_generate_success()


def get_run_cmd(
    model_name: str = MODEL_NAME,
    model_store: str = MODEL_STORE,
    input_path: str = "",
    repo_version: str = "",
) -> List[str]:
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


def test_default_success() -> None:
    """
    This function tests the default GPT2 model with input path.
    Expected result: Success.
    """
    process = subprocess.run(get_run_cmd(input_path=INPUT_PATH), check=False)
    assert process.returncode == 0


def test_default_no_input_path_success() -> None:
    """
    This function tests the default GPT2 model without input path.
    Expected result: Success.
    """
    process = subprocess.run(get_run_cmd(), check=False)
    assert process.returncode == 0


def test_no_model_name_throw_error() -> None:
    """
    This function tests missing model name.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_name=""), check=False)
    assert process.returncode == 1


def test_wrong_model_name_throw_error() -> None:
    """
    This function tests wrong model name.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_name="wrong_model_name"), check=False)
    assert process.returncode == 1


def test_no_model_store_throw_error() -> None:
    """
    This function tests missing model store.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_store=""), check=False)
    assert process.returncode == 1


def test_wrong_model_store_throw_error() -> None:
    """
    This function tests wrong model store.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(model_store="wrong_model_store"), check=False)
    assert process.returncode == 1


def test_wrong_input_path_throw_error() -> None:
    """
    This function tests wrong input path.
    Expected result: Failure.
    """
    process = subprocess.run(get_run_cmd(input_path="wrong_input_path"), check=False)
    assert process.returncode == 1


def test_vaild_repo_version_success() -> None:
    """
    This function tests valid repo version.
    Expected result: Success.
    """
    process = subprocess.run(
        get_run_cmd(repo_version="11c5a3d5811f50298f278a704980280950aedb10"),
        check=False,
    )
    assert process.returncode == 0


def test_invalid_repo_version_throw_error() -> None:
    """
    This function tests invalid repo version.
    Expected result: Failure.
    """
    process = subprocess.run(
        get_run_cmd(repo_version="invalid_repo_version"), check=False
    )
    assert process.returncode == 1


def test_inference_txt_file_success() -> None:
    """
    This function tests inference with example .txt file.
    Expected result: Success.
    """
    subprocess.run(get_run_cmd(), check=False)
    url = "http://localhost:8080/predictions/gpt2"
    headers = {"Content-Type": "application/text; charset=utf-8"}
    file_name = os.path.join(INPUT_PATH, "sample_text1.txt")
    with open(file_name, "r", encoding="utf-8") as file:
        data = file.read()
    try:
        response = requests.post(url, data=data, timeout=120, headers=headers)
    except requests.exceptions.RequestException:
        assert False
    if response.status_code != 200:
        assert False
    # Throw error if output is not text (it is in JSON format)
    try:
        json.loads(response.text)
    except json.JSONDecodeError:
        assert True
        return
    assert False


def test_inference_json_file_success() -> None:
    """
    This function tests inference with example .json file.
    Expected result: Success.
    """
    subprocess.run(get_run_cmd(), check=False)
    url = "http://localhost:8080/predictions/gpt2"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    file_name = os.path.join(INPUT_PATH, "sample_text4.json")
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.loads(file.read())
    try:
        response = requests.post(url, json=data, timeout=120, headers=headers)
    except requests.exceptions.RequestException:
        assert False
    if response.status_code != 200:
        assert False
    try:
        output = json.loads(response.text)
        assert output["id"] and output["outputs"]
    except (json.JSONDecodeError, ValueError, KeyError):
        assert False


def test_custom_model_skip_download_success() -> None:
    """
    This function tests custom model skipping download with input folder.
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


def test_custom_model_download_success() -> None:
    """
    This function tests download custom model input folder.
    Expected result: Success.
    """
    custom_model_setup(download=False)
    args = set_generate_args()
    args.repo_id = "gpt2"
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
