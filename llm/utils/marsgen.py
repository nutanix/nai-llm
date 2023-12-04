"""
This module contains utilities for MAR file generation

Attributes:
    MAR_NAME_LEN (int): Number of characters to include from repo_version in MAR name
"""
import os
import sys
import time
import threading
import subprocess
from typing import Dict
import tqdm
from utils.system_utils import (
    check_if_path_exists,
    get_all_files_in_directory,
    get_files_sizes,
)
from utils.generate_data_model import GenerateDataModel

# MAR_NAME_LEN - Number of characters to include from repo_version in MAR name
MAR_NAME_LEN = 7


def monitor_marfile_size(
    file_path: str, approx_marfile_size: float, stop_monitoring: threading.Event
) -> None:
    """
    Monitor the generation of a Model Archive File and display progress.

    Args:
        file_path (str): The path to the Model Archive File.
        approx_marfile_size (int): The approximate size of the Model Archive File in bytes.
        stop_monitoring (threading.Event): Threading Event to stop progress bar.
    """
    print("Model Archive File is Generating...\n")
    previous_file_size = 0
    progress_bar = tqdm.tqdm(
        total=approx_marfile_size,
        unit="B",
        unit_scale=True,
        desc="Creating Model Archive",
    )
    while not stop_monitoring.is_set():
        try:
            current_file_size = os.path.getsize(file_path)
        except FileNotFoundError:
            current_file_size = 0
        size_change = current_file_size - previous_file_size
        previous_file_size = current_file_size
        progress_bar.update(size_change)
        time.sleep(2)
    progress_bar.update(approx_marfile_size - current_file_size)
    progress_bar.close()
    print(
        f"\nModel Archive file size: {os.path.getsize(file_path) / (1024 ** 3):.2f} GB\n"
    )


def get_mar_name(
    model_name: str, repo_version: str, is_custom_model: str = False
) -> str:
    """
    This function returns MAR file name using model name and repo version.

    Args:
        model_name (str): Name of the model.
        repo_version (str): Commit ID of model's repo from HuggingFace repository.
        is_custom_model (bool): Set to True for custom models.

    Returns:
        str: MAR file name.
    """
    mar_name = (
        f"{model_name}"
        if is_custom_model
        else f"{model_name}_{repo_version[0:MAR_NAME_LEN]}"
    )
    return mar_name


def generate_mars(
    gen_model: GenerateDataModel,
    mar_config: str,
    model_store_dir: str,
    debug: str = False,
) -> None:
    """
    This function runs Torch Model Archiver command to generate MAR file. It calls the
    model_archiver_command_builder function to generate the command which it then runs.
    It also starts a thread for the progress bar of Model Archive file generation.

    Args:
        gen_model (GenerateDataModel): Dataclass that contains data required to generate MAR file.
        mar_config (str): Path to model_config.json.
        model_store_dir (str): Absolute path of export of MAR file.
        debug (bool, optional): Flag to print debug statements. Defaults to False.
    """
    if debug:
        print(
            f"## Starting generate_mars, mar_config:{mar_config}"
            f", model_store_dir:{model_store_dir}\n"
        )
    cwd = os.getcwd()
    os.chdir(os.path.dirname(mar_config))

    handler = gen_model.mar_utils.handler_path
    check_if_path_exists(handler, "Handler file", is_dir=False)

    # Reading all files in model_path to make extra_files string
    extra_files_list = get_all_files_in_directory(gen_model.mar_utils.model_path)
    extra_files_list = [
        os.path.join(gen_model.mar_utils.model_path, file) for file in extra_files_list
    ]
    extra_files = ",".join(extra_files_list)

    export_path = model_store_dir
    check_if_path_exists(export_path, "Model Store", is_dir=True)

    model_archiver_args = {
        "model_name": gen_model.model_name,
        "version": gen_model.repo_info.repo_version,
        "handler": handler,
        "extra_files": extra_files,
        "export_path": export_path,
    }

    cmd = model_archiver_command_builder(
        model_archiver_args=model_archiver_args, debug=debug
    )

    if debug:
        print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")

    try:
        # Event to stop the thread from monitoring output file size.
        stop_monitoring = threading.Event()

        # Approximate size of output Model Archive file.
        approx_marfile_size = get_files_sizes(extra_files_list) / 1.15

        # Creating a thread to monitor MAR file size while generation and show progress bar
        mar_progress_thread = threading.Thread(
            target=monitor_marfile_size,
            args=(
                os.path.join(model_store_dir, f"{gen_model.model_name}.mar"),
                approx_marfile_size,
                stop_monitoring,
            ),
        )
        mar_progress_thread.start()
        subprocess.check_call(cmd, shell=True)
        stop_monitoring.set()
        mar_progress_thread.join()
        print(f"## {gen_model.model_name}.mar is generated.\n")
    except subprocess.CalledProcessError as exc:
        print("## Creation failed !\n")
        if debug:
            print(f"## {gen_model.model_name} creation failed !, error: {exc}\n")
        sys.exit(1)

    os.chdir(cwd)


def model_archiver_command_builder(
    model_archiver_args: Dict[str, str], debug: bool = False
) -> str:
    """
    This function makes the Torch Model Archiver command using model_archiver_args parameter.

    Args:
        model_archiver_args (dict): Contains dictionary of arguments required to generate
        torch model archiever command
        debug (bool, optional): Flag to print debug statements. Defaults to False.

    Returns:
        str: Torch Model Achiver command
    """
    cmd = "torch-model-archiver"
    if model_archiver_args["model_name"]:
        cmd += f" --model-name {model_archiver_args['model_name']}"
    if model_archiver_args["version"]:
        cmd += f" --version {model_archiver_args['version']}"
    if model_archiver_args["handler"]:
        cmd += f" --handler {model_archiver_args['handler']}"
    if model_archiver_args["extra_files"]:
        cmd += f" --extra-files \"{model_archiver_args['extra_files']}\""
    if model_archiver_args["export_path"]:
        cmd += f" --export-path {model_archiver_args['export_path']}"
    cmd += " --force"
    print("\n## Generating mar file, will take few mins.\n")
    if debug:
        print(cmd)
    return cmd
