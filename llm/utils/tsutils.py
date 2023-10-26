"""
This module contains the utilities required to manage Torchserve start,
model registration, inferencing and Torchserve stop.

Attributes:
    torchserve_command (dict): Contains the torchserve command in different operating systems.
    torch_model_archiver_command (dict): Contains the torch-model-archiver command in
                                         different operating systems.
"""
import os
import platform
import time
import json
import requests
from utils.inference_data_model import InferenceDataModel, TorchserveStartData
from utils.system_utils import check_if_path_exists
from utils.shell_utils import copy_file


torchserve_command = {
    "Windows": "torchserve.exe",
    "Darwin": "torchserve",
    "Linux": "torchserve",
}

torch_model_archiver_command = {
    "Windows": "torch-model-archiver.exe",
    "Darwin": "torch-model-archiver",
    "Linux": "torch-model-archiver",
}


def generate_ts_start_cmd(ncs, ts_data: TorchserveStartData, debug):
    """
    This function generates the Torchserve start command.

    Args:
        ncs (bool): To enable '--no-config-snapshots' in Torchserve.
        ts_data (TorchserveStartData): Stores information required to start Torchserve.
        debug (bool): Flag to print debug statements.

    Returns:
        str: The Torchserve start command.
    """
    cmd = f"{torchserve_command[platform.system()]}"
    cmd += f" --start --model-store={ts_data.ts_model_store}"
    if ncs:
        cmd += " --ncs"
    if ts_data.ts_config_file:
        cmd += f" --ts-config={ts_data.ts_config_file}"
    if ts_data.ts_log_config:
        cmd += f" --log-config {ts_data.ts_log_config}"
    if ts_data.ts_log_file:
        print(f"## Console logs redirected to file: {ts_data.ts_log_file} \n")
        dirpath = os.path.dirname(ts_data.ts_log_file)
        cmd += f" >> {os.path.join(dirpath,ts_data.ts_log_file)}"
    if debug:
        print(f"## In directory: {os.getcwd()} | Executing command: {cmd} \n")
    return cmd


def start_torchserve(ts_data: TorchserveStartData, ncs=True, wait_for=10, debug=False):
    """
    This function calls generate_ts_start_cmd function to get the Torchserve start command
    and runs the same to start Torchserve.

    Args:
        ncs (bool, optional): To enable '--no-config-snapshots' in Torchserve. Defaults to True.
        ts_data (TorchserveStartData): Stores data required to start Torchserve. Defaults to None.
        wait_for (int, optional): Wait time(in secs) after running command. Defaults to 10.
        debug (bool, optional): Flag to print debug statements. Defaults to False.

    Returns:
        bool: True for successful Torchserve start and False otherwise
    """
    print("\n## Starting TorchServe \n")
    cmd = generate_ts_start_cmd(ncs, ts_data, debug)
    print(cmd)
    status = os.system(cmd)
    if status == 0:
        print("\n## Successfully started TorchServe \n")
        time.sleep(wait_for)
        print("## Registering model: this might take a while\n")
        return True
    print("## TorchServe failed to start ! Make sure it's not running already\n")
    return False


def stop_torchserve(wait_for=10):
    """
    This function is used to stop Torchserve.

    Args:
        wait_for (int, optional): Wait time(in secs) after running command. Defaults to 10.

    Returns:
        bool: True for successful Torchserve stop and False otherwise.
    """
    try:
        requests.get("http://localhost:8080/ping", timeout=wait_for)
    except requests.exceptions.RequestException:
        return False

    print("## Stopping TorchServe \n")
    cmd = f"{torchserve_command[platform.system()]} --stop"
    status = os.system(cmd)
    if status == 0:
        print("## Successfully stopped TorchServe \n")
        time.sleep(wait_for)
        return True
    print("## TorchServe failed to stop ! \n")
    return False


def set_config_properties(data_model: InferenceDataModel):
    """
    This function creates a configuration file for the model and sets certain parameters.
    Args:
        data_model (InferenceDataModel): An instance of the InferenceDataModel
                                      class with relevant information.
    Returns:
        None
    """
    template_config_path = data_model.ts_data.ts_config_file
    check_if_path_exists(template_config_path, "config.properties file", is_dir=False)

    dir_path = os.path.dirname(__file__)
    dst_config_path = os.path.join(dir_path, data_model.gen_folder, "config.properties")
    copy_file(template_config_path, dst_config_path)
    check_if_path_exists(
        dst_config_path, "config.properties file in gen folder", is_dir=False
    )

    (
        initial_workers,
        batch_size,
        max_batch_delay,
        response_timeout,
    ) = get_params_for_registration(data_model.model_name)

    mar_name = os.path.basename(data_model.mar_filepath)

    config_info = [
        f'\nmodel_snapshot={{"name":"startup.cfg","modelCount":1,'
        f'"models":{{"{data_model.model_name}":{{'
        f'"{data_model.repo_version}":{{"defaultVersion":true,"marName":"{mar_name}",'
        f'"minWorkers":{initial_workers or 1},'
        f'"maxWorkers":{initial_workers or 1},'
        f'"batchSize":{batch_size or 1},'
        f'"maxBatchDelay":{max_batch_delay or 1},'
        f'"responseTimeout":{response_timeout or 120}}}}}}}}}',
    ]
    with open(dst_config_path, "a", encoding="utf-8") as config_file:
        config_file.writelines(config_info)

    data_model.ts_data.ts_config_file = dst_config_path


def set_model_params(model_name):
    """
    This function reads generation parameters from model_config.json and sets them as
    environment variables for the handler to read. The generation parameters are :
    temperature, repetition_penalty, top_p, max_new_tokens.

    Args:
        model_name (str): Name of the model.
    """
    dirpath = os.path.dirname(__file__)
    generation_params = {
        "NAI_TEMPERATURE": "temperature",
        "NAI_REP_PENALTY": "repetition_penalty",
        "NAI_TOP_P": "top_p",
        "NAI_MAX_TOKENS": "max_new_tokens",
    }
    # Set the new environment variables with the provided values in model_config and
    # delete the environment variable value if not specified in model_config
    # or if modes_parms not present in model_config
    with open(os.path.join(dirpath, "../model_config.json"), encoding="UTF-8") as file:
        model_config = json.loads(file.read())
        if model_name in model_config:
            if "model_params" in model_config[model_name]:
                param_config = model_config[model_name]["model_params"]
                for param_name, param_value in generation_params.items():
                    if param_value in param_config:
                        os.environ[param_name] = str(param_config[param_value])
                    elif param_name in os.environ:
                        del os.environ[param_name]
            else:
                for param_name, param_value in generation_params.items():
                    if param_name in os.environ:
                        del os.environ[param_name]


def get_params_for_registration(model_name):
    """
    This function reads registration parameters from model_config.json returns them.
    The generation parameters are :
    initial_workers, batch_size, max_batch_delay, response_timeout.

    Args:
        model_name (str): Name of the model.

    Returns:
        str: initial_workers, batch_size, max_batch_delay, response_timeout
    """
    dirpath = os.path.dirname(__file__)
    initial_workers = batch_size = max_batch_delay = response_timeout = None

    with open(
        os.path.join(dirpath, "../model_config.json"), encoding="UTF-8"
    ) as config:
        model_config = json.loads(config.read())
        if model_name in model_config:
            param_config = model_config[model_name]["registration_params"]
            if "initial_workers" in param_config:
                initial_workers = param_config["initial_workers"]

            if "batch_size" in param_config:
                batch_size = param_config["batch_size"]

            if "max_batch_delay" in param_config:
                max_batch_delay = param_config["max_batch_delay"]

            if "response_timeout" in param_config:
                response_timeout = param_config["response_timeout"]

    return initial_workers, batch_size, max_batch_delay, response_timeout


def run_inference(
    model_inference_data, protocol="http", host="localhost", port="8080", timeout=120
):
    """
    This function sends request to run inference on Torchserve.

    Args:
        model_inference_data (str, list(str)): The model name and paths of input files.
        protocol (str, optional): Request protocol. Defaults to "http".
        host (str, optional): Request host. Defaults to "localhost".
        port (str, optional): Request Port. Defaults to "8080".
        timeout (int, optional): Request timeout (sec). Defaults to 120.

    Returns:
        requests.Response: Reponse of the inference request
    """
    model_name, file_name = model_inference_data

    url = f"{protocol}://{host}:{port}/predictions/{model_name}"
    files = {}
    with open(file_name, "rb") as file:
        files["data"] = (file_name, file)
        response = requests.post(url, files=files, timeout=timeout)
    return response


def run_health_check(
    model_name, protocol="http", host="localhost", port="8081", timeout=120
):
    """
    This function runs a health check for the workers of the deployed model

    Args:
        model_name (str): The name of the model.
        protocol (str, optional): Request protocol. Defaults to "http".
        host (str, optional): Request host. Defaults to "localhost".
        port (str, optional): Request Port. Defaults to "8081".
        timeout (int, optional): Request timeout (sec). Defaults to 120.

    Returns:
        bool: True for succesful health check and false otherwise.
    """
    url = f"{protocol}://{host}:{port}/models/{model_name}"
    response = requests.get(url, timeout=timeout)
    workers = response.json()[0]["workers"]
    for worker in workers:
        if worker["status"] != "READY":
            return False
    return True
