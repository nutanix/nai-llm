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
from collections import Counter
import re
import uuid
import huggingface_hub as hfh
from huggingface_hub.utils import HfHubHTTPError
from utils.marsgen import get_mar_name, generate_mars
from utils.system_utils import (
    check_if_path_exists,
    check_if_folder_empty,
    create_folder_if_not_exists,
    get_all_files_in_directory,
)
from utils.shell_utils import mv_file, rm_dir
from utils.generate_data_model import GenerateDataModel, set_values

FILE_EXTENSIONS_TO_IGNORE = [
    ".safetensors",
    ".safetensors.index.json",
    ".h5",
    ".ot",
    ".tflite",
    ".msgpack",
    ".onnx",
]

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")


def get_ignore_pattern_list(extension_list):
    """
    This function takes a list of file extensions and returns a list of patterns that
    can be used to filter out files with these extensions during model download.

    Args:
        extension_list (list(str)): A list of file extensions.

    Returns:
        list(str): A list of patterns with '*' prepended to each extension,
                    suitable for filtering files.
    """
    return ["*" + pattern for pattern in extension_list]


def compare_lists(list1, list2):
    """
    This function checks if two lists are equal by comparing their contents,
    regardless of the order.

    Args:
        list1 (list(str)): The first list to compare.
        list2 (list(str)): The second list to compare.

    Returns:
        bool: True if the lists have the same elements, False otherwise.
    """
    return Counter(list1) == Counter(list2)


def filter_files_by_extension(filenames, extensions_to_remove):
    """
    This function takes a list of filenames and a list of extensions to remove.
    It returns a new list of filenames after filtering out those with specified extensions.
    It uses regex patterns to filter filenames

    Args:
        filenames (list(str)): A list of filenames to be filtered.
        extensions_to_remove (list(str)): A list of file extensions to remove.

    Returns:
        list(str): A list of filenames after filtering.
    """
    pattern = "|".join([re.escape(suffix) + "$" for suffix in extensions_to_remove])
    # for FILE_EXTENSIONS_TO_IGNORE the pattern will be '\.safetensors$|\.safetensors\.index\.json$'
    filtered_filenames = [
        filename for filename in filenames if not re.search(pattern, filename)
    ]
    return filtered_filenames


def check_if_mar_exists(gen_model: GenerateDataModel):
    """
    This function checks if MAR file of a model already exists and skips
    generation if the MAR file already exists

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass
    """
    check_path = os.path.join(
        gen_model.mar_utils.mar_output, f"{gen_model.mar_utils.mar_name}.mar"
    )
    if os.path.exists(check_path):
        print(
            f"## Skipping MAR file generation as it already exists\n"
            f" Model name: {gen_model.model_name}\n Repository Version: "
            f"{gen_model.repo_info.repo_version}\n"
        )
        sys.exit(1)


def check_if_model_files_exist(gen_model: GenerateDataModel):
    """
    This function compares the list of files in the downloaded model directory with the
    list of files in the HuggingFace repository. It takes into account any files to
    ignore based on predefined extensions.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass

    Returns:
        bool: True if the downloaded model files match the expected
              repository files, False otherwise.
    """
    extra_files_list = get_all_files_in_directory(gen_model.mar_utils.model_path)
    hf_api = hfh.HfApi()
    repo_files = hf_api.list_repo_files(
        repo_id=gen_model.repo_info.repo_id,
        revision=gen_model.repo_info.repo_version,
        token=gen_model.repo_info.hf_token,
    )
    repo_files = filter_files_by_extension(repo_files, FILE_EXTENSIONS_TO_IGNORE)
    return compare_lists(extra_files_list, repo_files)


def create_tmp_model_store(mar_output, mar_name):
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


def move_mar(gen_model: GenerateDataModel, tmp_dir):
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


def read_config_for_download(gen_model: GenerateDataModel):
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

                # Make sure there is HF hub token for LLAMA(2)
                if (
                    gen_model.repo_info.repo_id.startswith("meta-llama")
                    and gen_model.repo_info.hf_token is None
                ):
                    print(
                        "## Error: HuggingFace Hub token is required for llama download."
                        " Please specify it using --hf_token=<your token>. "
                        "Refer https://huggingface.co/docs/hub/security-tokens"
                    )
                    sys.exit(1)

                # Validate downloaded files
                hf_api = hfh.HfApi()
                hf_api.list_repo_commits(
                    repo_id=gen_model.repo_info.repo_id,
                    revision=gen_model.repo_info.repo_version,
                    token=gen_model.repo_info.hf_token,
                )

                # Read handler file name
                if not gen_model.mar_utils.handler_path:
                    gen_model.mar_utils.handler_path = os.path.join(
                        os.path.dirname(__file__),
                        models[gen_model.model_name]["handler"],
                    )

            except (KeyError, HfHubHTTPError):
                print(
                    "## Error: Please check either repo_id, repo_version"
                    " or HuggingFace ID is not correct\n"
                )
                sys.exit(1)
        else:  # Custom model case
            if not gen_model.skip_download:
                print(
                    "## Please check your model name,"
                    " it should be one of the following : "
                )
                print(list(models.keys()))
                print(
                    "\n## If you want to use custom model files,"
                    " use the '--no_download' argument"
                )
                sys.exit(1)

            if check_if_folder_empty(gen_model.mar_utils.model_path):
                print("## Error: The given model path folder is empty\n")
                sys.exit(1)

            if not gen_model.mar_utils.handler_path:
                gen_model.mar_utils.handler_path = os.path.join(
                    os.path.dirname(__file__), "handler.py"
                )

            if not gen_model.repo_info.repo_version:
                gen_model.repo_info.repo_version = "1.0"

            gen_model.is_custom_model = True
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


def run_download(gen_model: GenerateDataModel):
    """
    This function checks if the given model path directory is empty and then
    downloads the given version's model files at that path.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass

    Returns:
        GenerateDataModel: An instance of the GenerateDataModel class.
    """
    if not check_if_folder_empty(gen_model.mar_utils.model_path):
        print("## Make sure the path provided to download model files is empty\n")
        sys.exit(1)

    print(
        f"\n## Starting model files download from {gen_model.repo_info.repo_id}"
        f" with version {gen_model.repo_info.repo_version}\n"
    )

    hfh.snapshot_download(
        repo_id=gen_model.repo_info.repo_id,
        revision=gen_model.repo_info.repo_version,
        local_dir=gen_model.mar_utils.model_path,
        local_dir_use_symlinks=False,
        token=gen_model.repo_info.hf_token,
        ignore_patterns=get_ignore_pattern_list(FILE_EXTENSIONS_TO_IGNORE),
    )
    print("## Successfully downloaded model_files\n")
    return gen_model


def create_mar(gen_model: GenerateDataModel):
    """
    This function checks if the Model Archive (MAR) file for the downloaded
    model exists in the specified model path otherwise generates the MAR file.
    The MAR file is generated in a temporary folder and then moved to model store
    to avoid conflicts.

    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel dataclass
    """
    if not gen_model.is_custom_model and not check_if_model_files_exist(gen_model):
        print("## Model files do not match HuggingFace repository files")
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


def run_script(params):
    """
    This function validates input parameters, downloads model files and
    creates model archive file (MAR file) for the given model.

    Args:
        params (Namespace): An argparse.Namespace object containing command-line arguments.
                These are the necessary parameters and configurations for the script.
    """
    gen_model = set_values(params)
    gen_model = read_config_for_download(gen_model)

    check_if_path_exists(gen_model.mar_utils.model_path, "model_path", is_dir=True)
    check_if_path_exists(gen_model.mar_utils.mar_output, "mar_output", is_dir=True)
    check_if_mar_exists(gen_model)

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
        metavar="mn",
        help="Name of model",
    )
    parser.add_argument(
        "--repo_version",
        type=str,
        default="",
        metavar="rv",
        help="Commit ID of models repo from HuggingFace repository",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Set flag to skip downloading the model files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        required=True,
        metavar="mp",
        help="Absolute path of model files (should be empty if downloading)",
    )
    parser.add_argument(
        "--mar_output",
        type=str,
        default="",
        required=True,
        metavar="mx",
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
        metavar="hft",
        help="HuggingFace Hub token to download LLAMA(2) models",
    )
    parser.add_argument("--debug", action="store_true", help="flag to debug")
    args = parser.parse_args()
    run_script(args)
