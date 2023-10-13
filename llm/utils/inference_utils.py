"""
inference_utils
This module contains utilities to start and manage Torchserve server.
"""
import os
import sys
import traceback
import torch
import requests
import utils.tsutils as ts
import utils.system_utils as su
import utils.inference_data_model as idm


def error_msg_print():
    """
    error_msg_print
    This function prints an error message and stops Torchserve.
    """
    print("\n**************************************")
    print("*\n*\n*  Error found - Unsuccessful ")
    print("*\n*\n**************************************")
    ts.stop_torchserve()


def set_compute_setting(gpus):
    """
    set_compute_setting
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


def ts_health_check():
    """
    ts_health_check
    This function makes health check request to server.
    """
    os.system("curl localhost:8080/ping")


def start_ts_server(ts_data, gpus, debug):
    """
    start_ts_server
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


def execute_inference_on_inputs(model_inputs, model_name):
    """
    execute_inference_on_inputs
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
            print(
                f"## Successfully ran inference on {model_name} model."
                f"\n\n Output - {response.text}\n\n"
            )
        else:
            print(f"## Failed to run inference on {model_name} model \n")
            error_msg_print()
            sys.exit(1)


def register_model(model_name, input_mar, gpus):
    """
    register_model
    This function registers model on Torchserve by calling register_model from tsutils.

    Args:
        model_name (str): Name of the model.
        input_mar (str): Path to input data directory.
        gpus (int): Number of GPUs.
    """
    model_register_data = (model_name, input_mar)
    response = ts.register_model(model_register_data, gpus)
    if response and response.status_code == 200:
        print(f"## Successfully registered {model_name} model with torchserve \n")
    else:
        print("## Failed to register model with torchserve \n")
        error_msg_print()
        sys.exit(1)


def unregister_model(model_name):
    """
    unregister_model
    This function unregisters model on Torchserve by calling unregister_model from tsutils.

    Args:
        model_name (str): Name of the model.
    """
    response = ts.unregister_model(model_name)
    if response and response.status_code == 200:
        print(f"## Successfully unregistered {model_name} \n")
    else:
        print(f"## Failed to unregister {model_name} \n")
        error_msg_print()
        sys.exit(1)


def validate_inference_model(models_to_validate, debug):
    """
    validate_inference_model
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

        execute_inference_on_inputs(model_inputs, model_name)

        if debug:
            os.system(f"curl http://localhost:8081/models/{model_name}")
        print(f"## {model_name} Handler is stable. \n")


def get_inference_internal(data_model, debug):
    """
    get_inference_internal
    This function starts Torchserve, runs health check of server, registers model,
    and runs inference on input folder path.

    Args:
        data_model (InferenceDataModel): Dataclass containing information for running Torchserve.
        debug (bool): Flag to print debug statements.
    """
    set_compute_setting(data_model.gpus)
    start_ts_server(ts_data=data_model.ts_data, gpus=data_model.gpus, debug=debug)
    ts_health_check()
    register_model(data_model.model_name, data_model.mar_filepath, data_model.gpus)

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


def get_inference_with_mar(data_model, debug=False):
    """
    get_inference_with_mar
    This function sets ts_data in data_model (InferenceDataModel), and calls get_inference_internal,
    and catches any execptions caused by sending requests.

    Args:
        data_model (InferenceDataModel): Dataclass containing information for running Torchserve.
        debug (bool, optional): Flag to print debug statements. Defaults to False.
    """
    try:
        data_model = idm.prepare_settings(data_model)
        get_inference_internal(data_model, debug=debug)
    except (KeyError, requests.exceptions.RequestException):
        error_msg_print()
        if debug:
            traceback.print_exc()
        sys.exit(1)
