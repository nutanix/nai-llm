"""
tsutils
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


def generate_ts_start_cmd(ncs, ts_data, gpus, debug):
    """
    generate_ts_start_cmd
    This function generates the Torchserve start command.

    Args:
        ncs (bool): To enable '--no-config-snapshots' in Torchserve.
        ts_data (TorchserveStartData): Stores information required to start Torchserve.
        gpus (int): To set number of GPUs.
        debug (bool): Flag to print debug statements.

    Returns:
        str: The Torchserve start command.
    """
    cmd = f"TS_NUMBER_OF_GPU={gpus} {torchserve_command[platform.system()]}"
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


def start_torchserve(ts_data, ncs=True, wait_for=10, gpus=0, debug=False):
    """
    start_torchserve
    This function calls generate_ts_start_cmd function to get the Torchserve start command
    and runs the same to start Torchserve.

    Args:
        ncs (bool, optional): To enable '--no-config-snapshots' in Torchserve. Defaults to True.
        ts_data (TorchserveStartData): Stores data required to start Torchserve. Defaults to None.
        wait_for (int, optional): Wait time(in secs) after running command. Defaults to 10.
        gpus (int, optional): Number of GPUs. Defaults to 0.
        debug (bool, optional): Flag to print debug statements. Defaults to False.

    Returns:
        bool: True for successful Torchserve start and False otherwise
    """
    print("## Starting TorchServe \n")
    cmd = generate_ts_start_cmd(ncs, ts_data, gpus, debug)
    print(cmd)
    status = os.system(cmd)
    if status == 0:
        print("## Successfully started TorchServe \n")
        time.sleep(wait_for)
        return True
    print("## TorchServe failed to start ! Make sure it's not running already\n")
    return False


def stop_torchserve(wait_for=10):
    """
    stop_torchserve
    This function is used to stop Torchserve.

    Args:
        wait_for (int, optional): Wait time(in secs) after running command. Defaults to 10.

    Returns:
        bool: True for successful Torchserve stop and False otherwise.
    """
    try:
        requests.get("http://localhost:8080/ping")
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


def set_model_params(model_name):
    """
    set_model_params
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
    get_params_for_registration
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


def register_model(
    model_register_data, gpus, protocol="http", host="localhost", port="8081"
):
    """
    register_model
    This functions is used to register a model on Torchserve.

    Args:
        model_register_data (str, str): The model name and MAR file name
        gpus (int): Number of GPUs
        protocol (str, optional): Request protocol. Defaults to "http".
        host (str, optional): Request host. Defaults to "localhost".
        port (str, optional): Request Port. Defaults to "8081".

    Returns:
        requests.Response: Reponse of the registration request
    """
    model_name, marfile = model_register_data
    print(f"\n## Registering {model_name} model, this might take a while \n")
    (
        initial_workers,
        batch_size,
        max_batch_delay,
        response_timeout,
    ) = get_params_for_registration(model_name)
    if (
        initial_workers is None
    ):  # setting the default value of workers to number of gpus
        initial_workers = gpus

    params = (
        ("url", marfile),
        ("initial_workers", initial_workers or 1),
        ("batch_size", batch_size or 1),
        ("max_batch_delay", max_batch_delay or 200),
        ("response_timeout", response_timeout or 2000),
        ("synchronous", "true"),
    )

    url = f"{protocol}://{host}:{port}/models"
    response = requests.post(url, params=params, verify=False)
    return response


def run_inference(
    model_inference_data, protocol="http", host="localhost", port="8080", timeout=120
):
    """
    run_inference
    This function sends request to run inference on Torchserve.

    Args:
        model_inference_data (str, list(str)): The model name and paths of input files.
        protocol (str, optional): Request protocol. Defaults to "http".
        host (str, optional): Request host. Defaults to "localhost".
        port (str, optional): Request Port. Defaults to "8081".
        timeout (int, optional): Request timeout (sec). Defaults to 120.

    Returns:
        requests.Response: Reponse of the inference request
    """
    model_name, file_name = model_inference_data

    print(f"## Running inference on {model_name} model \n")
    url = f"{protocol}://{host}:{port}/predictions/{model_name}"
    files = {}
    with open(file_name, "rb") as file:
        files["data"] = (file_name, file)
        response = requests.post(url, files=files, timeout=timeout)
    print(response)
    return response


def unregister_model(model_name, protocol="http", host="localhost", port="8081"):
    """
    unregister_model
    This function sends request to unregister model on Torchserve.

    Args:
        model_name (str): Name of the model
        protocol (str, optional): Request protocol. Defaults to "http".
        host (str, optional): Request host. Defaults to "localhost".
        port (str, optional): Request Port. Defaults to "8081".

    Returns:
        requests.Response: Reponse of the unregister request
    """
    print(f"## Unregistering {model_name} model \n")
    url = f"{protocol}://{host}:{port}/models/{model_name}"
    response = requests.delete(url, verify=False)
    return response
