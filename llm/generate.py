"""
This module downloads model files and generates MAR file.

Attributes:
    FILE_EXTENSIONS_TO_IGNORE (list(str)): List of strings containing extensions to ignore
    during download and validation of model files.
    MAR_CONFIG_PATH (str): Path of model_config.json.
"""

import os
import argparse
import json
import sys
import uuid
from typing import List
import huggingface_hub as hfh
from utils.marsgen import get_mar_name, generate_mars
from utils.system_utils import (
    check_if_path_exists,
    check_if_folder_empty,
    create_folder_if_not_exists,
)
from utils.shell_utils import mv_file, rm_dir
from utils.generate_data_model import GenerateDataModel

PREFERRED_MODEL_FORMATS = [".safetensors", ".bin"]  # In order of Preference
OTHER_MODEL_FORMATS = [
    "*.pt",
    "*.h5",
    "*.gguf",
    "*.msgpack",
    "*.tflite",
    "*.ot",
    "*.onnx",
]

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")


def get_ignore_pattern_list(gen_model: GenerateDataModel) -> List[str]:
    """
    This method creates a list of file extensions to ignore from a priority list based on files
    present in the Hugging Face Repo. It filters out extensions not found in the repository and
    returns them as ignore patterns prefixed with '*' which is expected by Hugging Face client.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel class

    Returns:
        list(str): A list of patterns with '*' prepended to each extension,
                    suitable for filtering files.
    """
    repo_file_extensions = gen_model.get_repo_file_extensions()
    for desired_extension in PREFERRED_MODEL_FORMATS:
        if desired_extension in repo_file_extensions:
            ignore_list = [
                "*" + ignore_extension
                for ignore_extension in PREFERRED_MODEL_FORMATS
                if ignore_extension != desired_extension
            ]
            ignore_list.extend(OTHER_MODEL_FORMATS)
            return ignore_list
    return []


def create_tmp_model_store(mar_output: str, mar_name: str) -> str:
    """
    This function creates a temporary directory in model store in which
    the MAR file will be stored temporarily.

    Args:
        mar_output (str): Path to export of MAR file.
        mar_name (str): Name of the MAR file.

    Returns:
        str: Path of temporary directory.
    """
    rand_uuid = str(uuid.uuid4())[-5:]  # select last 5 digits of uuid
    dir_name = f"tmp_{mar_name}_{rand_uuid}"
    tmp_dir = os.path.join(mar_output, dir_name)
    create_folder_if_not_exists(tmp_dir)
    return tmp_dir


def move_mar(gen_model: GenerateDataModel, tmp_dir: str) -> None:
    """
    This funtion moves MAR file from the temporary directory to model store.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass
        tmp_dir (str): Path of temporary directory.
    """
    old_filename = f"{gen_model.model_name}.mar"
    new_filename = f"{gen_model.mar_utils.mar_name}.mar"
    src = os.path.join(tmp_dir, old_filename)
    dst = os.path.join(gen_model.mar_utils.mar_output, new_filename)
    check_if_path_exists(src, "Generated mar file is missing", is_dir=False)
    mv_file(src, dst)


def read_config_for_download(gen_model: GenerateDataModel) -> GenerateDataModel:
    """
    This function reads repo id, version and handler name from
    model_config.json and sets values for the GenerateDataModel object.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass

    Raises:
        sys.exit(1): If model name, repo_id or repo_version is not valid, the
                     function will terminate the program with an exit code of 1.
    """
    check_if_path_exists(MODEL_CONFIG_PATH, "Model Config", is_dir=False)
    with open(MODEL_CONFIG_PATH, encoding="UTF-8") as config:
        models = json.loads(config.read())
        if gen_model.model_name in models:
            gen_model.is_custom_model = False
            try:
                # Read and validate the repo_id and repo_version
                gen_model.repo_info.repo_id = models[gen_model.model_name]["repo_id"]
                if not gen_model.repo_info.repo_version:
                    gen_model.repo_info.repo_version = models[gen_model.model_name][
                        "repo_version"
                    ]

                # Read handler file name
                if not gen_model.mar_utils.handler_path:
                    gen_model.mar_utils.handler_path = os.path.join(
                        os.path.dirname(__file__),
                        models[gen_model.model_name]["handler"],
                    )

                # Validate hf_token
                gen_model.validate_hf_token()

                # Validate repository info
                gen_model.validate_commit_info()

            except (KeyError, ValueError):
                print(
                    "## There seems to be an error in the model_config.json file. "
                    "Please check the same."
                )
                sys.exit(1)

        else:  # Custom model and HuggingFace model case
            gen_model.is_custom_model = True
            if gen_model.skip_download:
                if check_if_folder_empty(gen_model.mar_utils.model_path):
                    print("## Error: The given model path folder is empty\n")
                    sys.exit(1)

                if not gen_model.repo_info.repo_version:
                    gen_model.repo_info.repo_version = "1.0"

            else:
                if not gen_model.repo_info.repo_id:
                    print(
                        "## If you want to create a model archive file for supported models, "
                        "make sure you're model name is present in the below : "
                    )
                    print(list(models.keys()))
                    print(
                        "\nIf you want to create a model archive file for"
                        " either a Custom Model or other HuggingFace models, "
                        "refer to the official GPT-in-a-Box documentation: "
                        "https://opendocs.nutanix.com/gpt-in-a-box/overview/"
                    )
                    sys.exit(1)

                # Validate hf_token
                gen_model.validate_hf_token()

                # Validate repository info
                gen_model.validate_commit_info()

            if not gen_model.mar_utils.handler_path:
                gen_model.mar_utils.handler_path = os.path.join(
                    os.path.dirname(__file__), "handler.py"
                )

            print(
                f"\n## Generating MAR file for "
                f"custom model files: {gen_model.model_name}"
            )

    gen_model.mar_utils.mar_name = get_mar_name(
        gen_model.model_name,
        gen_model.repo_info.repo_version,
        gen_model.is_custom_model,
    )
    return gen_model


def run_download(gen_model: GenerateDataModel) -> GenerateDataModel:
    """
    This function checks if the given model path directory is empty and then
    downloads the given version's model files at that path.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass

    Returns:
        GenerateDataModel: An instance of the GenerateDataModel class.
    """
    if not check_if_folder_empty(gen_model.mar_utils.model_path):
        print(
            "## Make sure the model_path provided to download model files through is empty\n"
        )
        sys.exit(1)

    print(
        f"\n## Starting model files download from {gen_model.repo_info.repo_id}"
        f" with version {gen_model.repo_info.repo_version}\n"
    )

    hfh.snapshot_download(
        repo_id=gen_model.repo_info.repo_id,
        revision=gen_model.repo_info.repo_version,
        local_dir=gen_model.mar_utils.model_path,
        token=gen_model.repo_info.hf_token,
        ignore_patterns=get_ignore_pattern_list(gen_model),
    )
    print("## Successfully downloaded model_files\n")
    return gen_model


def create_mar(gen_model: GenerateDataModel) -> None:
    """
    This function checks if the Model Archive (MAR) file for the downloaded
    model exists in the specified model path otherwise generates the MAR file.
    The MAR file is generated in a temporary folder and then moved to model store
    to avoid conflicts.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass
    """
    if check_if_folder_empty(gen_model.mar_utils.model_path):
        print("## Model files not present in Model Path directory")
        sys.exit(1)

    # Creates a temporary directory with the mar_name inside model_store
    tmp_dir = create_tmp_model_store(
        gen_model.mar_utils.mar_output, gen_model.mar_utils.mar_name
    )

    generate_mars(
        gen_model=gen_model,
        mar_config=MODEL_CONFIG_PATH,
        model_store_dir=tmp_dir,
        debug=gen_model.debug,
    )

    # Move MAR file to model_store
    move_mar(gen_model, tmp_dir)
    # Delete temporary folder
    rm_dir(tmp_dir)
    print(
        f"\n## Mar file for {gen_model.model_name}"
        f" with version {gen_model.repo_info.repo_version} is generated!\n"
    )


def run_script(params: argparse.Namespace) -> bool:
    """
    This function validates input parameters, downloads model files and
    creates model archive file (MAR file) for the given model.

    Args:
        params (Namespace): An argparse.Namespace object containing command-line arguments.
                These are the necessary parameters and configurations for the script.
    Returns:
        bool: True for successful execution and False otherwise (used for testing)
    """
    gen_model = GenerateDataModel(params)
    gen_model = read_config_for_download(gen_model)

    check_if_path_exists(gen_model.mar_utils.model_path, "model_path", is_dir=True)
    check_if_path_exists(gen_model.mar_utils.mar_output, "mar_output", is_dir=True)
    gen_model.check_if_mar_exists()

    if not gen_model.skip_download:
        gen_model = run_download(gen_model)
    create_mar(gen_model)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        required=True,
        metavar="n",
        help="Name of model",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        metavar="ri",
        help="HuggingFace repository ID (In case of custom model download)",
    )
    parser.add_argument(
        "--repo_version",
        type=str,
        default=None,
        metavar="rv",
        help="Commit ID of models repo from HuggingFace repository",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Set flag to skip downloading the model files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        required=True,
        metavar="p",
        help="Absolute path of model files (should be empty if downloading)",
    )
    parser.add_argument(
        "--mar_output",
        type=str,
        default="",
        required=True,
        metavar="a",
        help="Absolute path of exported MAR file (.mar)",
    )
    parser.add_argument(
        "--handler_path",
        type=str,
        default="",
        metavar="hp",
        help="absolute path of handler",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        metavar="ht",
        help="HuggingFace Hub token to download LLAMA(2) models",
    )
    parser.add_argument("--debug", action="store_true", help="flag to debug")
    args = parser.parse_args()
    run_script(args)
