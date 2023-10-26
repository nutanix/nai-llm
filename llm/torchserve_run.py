"""
This module starts Torchserve, registers the given LLM model and runs inference using
the given inputs after validating all the input parameters required to do the same.

Attributes:
    MODEL_CONFIG_PATH (str): Path to model_config.json file.
"""
import os
import argparse
import json
from utils.inference_utils import get_inference
from utils.shell_utils import rm_dir
import utils.tsutils as ts
from utils.system_utils import check_if_path_exists
from utils.system_utils import create_folder_if_not_exists, remove_suffix_if_starts_with
import utils.inference_data_model as idm
from utils.marsgen import get_mar_name

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")


def read_config_for_inference(params):
    """
    Function that reads repo version and validates GPU type

    Args:
        params (Namespace): An argparse.Namespace object containing command-line arguments.
                            These are the necessary parameters and configurations for the script.

    Raises:
        sys.exit(1): If model name is not valid, the function will terminate
                     the program with an exit code of 1.
    Returns:
        Namespace: params updated with repo version
    """
    with open(MODEL_CONFIG_PATH, encoding="UTF-8") as config:
        models = json.loads(config.read())
        params.is_custom_model = False
        if params.model_name not in models:
            print(
                f"## Using custom MAR file : {params.model_name}.mar\n\n"
                f"WARNING: This model has not been validated on any GPUs\n\n"
            )
            params.is_custom_model = True

        if not params.is_custom_model and params.gpu_type:
            gpu_type_list = models[params.model_name]["gpu_type"]
            gpu_type = remove_suffix_if_starts_with(params.gpu_type, "NVIDIA")
            if gpu_type not in gpu_type_list:
                print(
                    "WARNING: This GPU Type is not validated, the validated GPU Types are:"
                )
                for gpu in gpu_type_list:
                    print(gpu)

        if (
            not params.is_custom_model
            and models[params.model_name]["repo_version"]
            and not params.repo_version
        ):
            params.repo_version = models[params.model_name]["repo_version"]
    return params


def set_mar_filepath(model_store, model_name, repo_version, is_custom_model):
    """
    Funtion that creates the MAR file path given the model store, model name and repo version.
    The name of the MAR file is returned from get_mar_name from marsgen.

    Args:
        model_store (str): Path to model store.
        model_name (str): Name of the model.
        repo_version (str): Commit ID of model's repo from HuggingFace repository.
        is_custom_model (bool): Set to True for custom models.

    Returns:
        str: Path to MAR file.
    """

    mar_name = f"{get_mar_name(model_name, repo_version, is_custom_model)}.mar"
    return os.path.join(model_store, mar_name)


def run_inference_with_mar(params):
    """
    Function that checks sets the required parameters, starts Torchserve, registers
    the model and runs inference on given input data.

    Args:
        params (Namespace): An argparse.Namespace object containing command-line arguments.
                            These are the necessary parameters and configurations for the script.
    """
    data_model = idm.set_data_model(params)
    get_inference(data_model, params.debug_mode)


def run_inference(params):
    """
    This function validates model store directory, MAR file path, input data directory,
    generates the temporary gen folder to store logs and sets model generation parameters as
    environment variables. Then it calls run_inference_with_mar.

    Args:
        params (Namespace): An argparse.Namespace object containing command-line arguments.
                            These are the necessary parameters and configurations for the script.
    """
    check_if_path_exists(params.model_store, "Model Store", is_dir=True)

    params.mar = set_mar_filepath(
        params.model_store,
        params.model_name,
        params.repo_version,
        params.is_custom_model,
    )
    check_if_path_exists(params.mar, "MAR file", is_dir=False)

    if params.data:
        check_if_path_exists(params.data, "Input data folder", is_dir=True)

    create_folder_if_not_exists(
        os.path.join(os.path.dirname(__file__), "utils", params.gen_folder_name)
    )

    ts.set_model_params(params.model_name)
    run_inference_with_mar(params)


def torchserve_run(params):
    """
    This function calls cleanup function, check if model config exists and then calls run_inference.

    Args:
        params (Namespace): An argparse.Namespace object containing command-line arguments.
                            These are the necessary parameters and configurations for the script.
    """
    try:
        # Stop the server if anything is running
        cleanup(params.gen_folder_name, True, False)

        check_if_path_exists(MODEL_CONFIG_PATH, "Model Config", is_dir=False)
        params = read_config_for_inference(params)

        run_inference(params)

        print("\n**************************************")
        print("*\n*\n*  Ready For Inferencing  ")
        print("*\n*\n**************************************")

    finally:
        cleanup(params.gen_folder_name, params.stop_server, params.ts_cleanup)


def cleanup(gen_folder, ts_stop=True, ts_cleanup=True):
    """
    This function stops Torchserve, deletes the temporary gen folder and the logs in it.

    Args:
        gen_folder (str): Path to gen directory.
        ts_stop (bool, optional): Flag set to stop Torchserve. Defaults to True.
        ts_cleanup (bool, optional): Flag set to delete gen folder. Defaults to True.
    """
    if ts_stop:
        ts.stop_torchserve()
        dirpath = os.path.dirname(__file__)
        # clean up the logs folder to reset logs before the next run
        rm_dir(os.path.join(dirpath, "utils", gen_folder, "logs"))

        if ts_cleanup:
            # clean up the entire generate folder
            rm_dir(os.path.join(dirpath, "utils", gen_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference run script")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        metavar="d",
        help="absolute path to the inputs folder that contains data to be predicted.",
    )
    parser.add_argument(
        "--model_name", type=str, default="", metavar="n", help="name of the model file"
    )
    parser.add_argument(
        "--repo_version",
        type=str,
        default="",
        metavar="n",
        help="HuggingFace repository version",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="",
        metavar="gn",
        help="type of gpus to use for execution",
    )
    parser.add_argument(
        "--gen_folder_name",
        type=str,
        default="gen",
        metavar="f",
        help="Name for generate folder used to create temp files",
    )
    parser.add_argument(
        "--stop_server",
        type=int,
        default=0,
        metavar="stop",
        help="Stop torchserve after run completion",
    )
    parser.add_argument(
        "--ts_cleanup",
        type=int,
        default=0,
        metavar="cleanup",
        help="clean up torchserve temp files after run completion",
    )
    parser.add_argument(
        "--debug_mode", type=int, default=0, metavar="debug", help="run debug mode"
    )
    parser.add_argument(
        "--model_store",
        type=str,
        default="",
        metavar="model_store",
        help="absolute path to the model store directory",
    )
    args = parser.parse_args()
    torchserve_run(args)
