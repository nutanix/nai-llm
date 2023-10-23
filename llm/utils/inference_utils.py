"""
This module contains utilities to start and manage Torchserve server.
"""
import os
import sys
import time
import traceback
import torch
import requests
import utils.tsutils as ts
import utils.system_utils as su
from utils.inference_data_model import (
    InferenceDataModel,
    TorchserveStartData,
    prepare_settings,
)

PATH_TO_SAMPLE = "../../data/qa/sample_text1.txt"


def error_msg_print():
    """
    This function prints an error message and stops Torchserve.
    """
    print("\n**************************************")
    print("*\n*\n*  Error found - Unsuccessful ")
    print("*\n*\n**************************************")
    ts.stop_torchserve()


def set_compute_setting(gpus):
    """
    This function read the compute setting (either GPU or CPU).

    Args:
        gpus (int): Number of GPUs.
    """
    if gpus > 0 and su.is_gpu_instance():
        if not torch.cuda.is_available():
            sys.exit("## CUDA not found \n")
        print(f"\n## Running on {gpus} GPU(s) \n")

    else:
        print("\n## Running on CPU \n")
        gpus = 0


def start_ts_server(ts_data: TorchserveStartData, gpus, debug):
    """
    This function starts Torchserve by calling start_torchserve from tsutils
    and throws error if it doesn't start.

    Args:
        ts_data (TorchserveStartData): Stores information required to start Torchserve.
        gpus (int): Number of GPUs.
        debug (bool): Flag to print debug statements.
    """
    started = ts.start_torchserve(ts_data=ts_data, gpus=gpus, debug=debug)
    if not started:
        error_msg_print()
        sys.exit(1)


def ts_health_check(model_name, model_timeout=1200):
    """
    This function checks if the model is registered or not.
    Args:
      model_name (str): The name of the model that is being registered.
      deploy_name (str): The name of the server where the model is registered.
      model_timeout (int): Maximum amount of time to wait for a response from server.
    Raises:
        requests.exceptions.RequestException: In case of request errors.
    """
    model_input = os.path.join(os.path.dirname(__file__), PATH_TO_SAMPLE)
    su.check_if_path_exists(model_input, "health check input", is_dir=False)
    retry_count = 0
    sleep_time = 10
    success = False
    while not success and retry_count * sleep_time < model_timeout:
        try:
            success = execute_inference_on_inputs([model_input], model_name, retry=True)
        except requests.exceptions.RequestException:
            pass
        if not success:
            time.sleep(sleep_time)
            retry_count += 1
    if success:
        print("## Health check passed. Model registered.\n")
    else:
        print(
            f"## Failed health check after multiple retries for model - {model_name} \n"
        )
        sys.exit(1)


def execute_inference_on_inputs(model_inputs, model_name, retry=False):
    """
    This function runs inference on given input data files and model name by
    calling run_inference from tsutils.

    Args:
        model_inputs (list(str)): Paths to input data files.
        model_name (str): Name of the model.
    """
    for data in model_inputs:
        model_inference_data = (model_name, data)
        response = ts.run_inference(model_inference_data)
        if response and response.status_code == 200:
            if not retry:
                print(
                    f"## Successfully ran inference on {model_name} model."
                    f"\n\n Output - {response.text}\n\n"
                )
            is_success = True
        else:
            if not retry:
                print(f"## Failed to run inference on {model_name} model \n")
                error_msg_print()
                sys.exit(1)
            is_success = False
    return is_success


def validate_inference_model(models_to_validate, debug):
    """
    This function consolidates model name and input to use for inference
    and calls execute_inference_on_inputs

    Args:
        models_to_validate (list(dict)): List of dict containing model name and
                                        list of paths of input files.
        debug (bool): Flag to print debug statements.
    """
    for model in models_to_validate:
        model_name = model["name"]
        model_inputs = model["inputs"]

        print(f"## Running inference on {model_name} model \n")
        execute_inference_on_inputs(model_inputs, model_name)

        if debug:
            os.system(f"curl http://localhost:8081/models/{model_name}")
        print(f"## {model_name} Handler is stable. \n")


def get_inference(data_model: InferenceDataModel, debug):
    """
    This function starts Torchserve, runs health check of server, registers model,
    and runs inference on input folder path. It catches KeyError and HTTPError exceptions

    Args:
        data_model (InferenceDataModel): Dataclass containing information for running Torchserve.
        debug (bool): Flag to print debug statements.
    Raises:
        KeyError: In case of reading JSON files.
        requests.exceptions.RequestException: In case of request errors.
    """
    data_model = prepare_settings(data_model)
    set_compute_setting(data_model.gpus)
    ts.set_config_properties(data_model)

    start_ts_server(ts_data=data_model.ts_data, gpus=data_model.gpus, debug=debug)
    ts_health_check(data_model.model_name)

    try:
        if data_model.input_path:
            # get relative paths of files
            inputs = su.get_all_files_in_directory(data_model.input_path)
            # prefix with model path
            inputs = [os.path.join(data_model.input_path, file) for file in inputs]
            inference_model = {
                "name": data_model.model_name,
                "inputs": inputs,
            }
            models_to_validate = [inference_model]
            if inputs:
                validate_inference_model(models_to_validate, debug)
    except (KeyError, requests.exceptions.RequestException):
        error_msg_print()
        if debug:
            traceback.print_exc()
        sys.exit(1)
