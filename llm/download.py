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
import dataclasses
import re
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


@dataclasses.dataclass
class MarUtils:
    """
    This dataclass that stores information regarding MAR file generation.

    Attributes:
        mar_output (str): Path of directory to export MAR file.
        mar_name (str): Name of MAR file.
        model_path (str): Path of model files directory.
        handler_path (str): Path of handler file of Torchserve.
    """

    mar_output = str()
    mar_name = str()
    model_path = str()
    handler_path = str()


@dataclasses.dataclass
class RepoInfo:
    """
    This dataclass stores information regarding the HuggingFace model repository.

    Attributes:
        repo_id (str): Repository ID of model in HuggingFace.
        repo_version (str): Commit ID of model's repo from HuggingFace repository.
        hf_token (str): Your HuggingFace token. Needed to download and verify LLAMA(2) models.
    """

    repo_id = str()
    repo_version = str()
    hf_token = str()


@dataclasses.dataclass
class DownloadDataModel:
    """
    This dataclass stores information regarding model download and MAR file generation.

    Attributes:
        model_name (str): Name of the model.
        skip_download (bool): Set to skip download model.
        mar_utils (MarUtils): Contains data regarding MAR file generation.
        repo_info (RepoInfo): Contains data regarding HuggingFace model repository.
        debug (bool): Flag to print debug statements.
    """

    model_name = str()
    skip_download = bool()
    mar_utils = MarUtils()
    repo_info = RepoInfo()
    debug = bool()


def set_values(params):
    """
    This function sets values for the DownloadDataModel object based on the command-line arguments.

    Args:
        params: An argparse.Namespace object containing command-line arguments.

    Returns:
        DownloadDataModel: An instance of the DownloadDataModel dataclass
    """
    dl_model = DownloadDataModel()
    dl_model.model_name = params.model_name
    dl_model.skip_download = params.no_download
    dl_model.debug = params.debug

    dl_model.repo_info.hf_token = params.hf_token
    dl_model.repo_info.repo_version = params.repo_version

    dl_model.mar_utils.handler_path = params.handler_path
    dl_model.mar_utils.model_path = params.model_path
    dl_model.mar_utils.mar_output = params.mar_output

    read_config_for_download(dl_model)

    # Sets MAR file name name as <model_name>_<first_n_chars_of_repo_version>
    dl_model.mar_utils.mar_name = get_mar_name(
        dl_model.model_name, dl_model.repo_info.repo_version
    )
    return dl_model


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


def check_if_mar_exists(dl_model):
    """
    This function checks if MAR file of a model already exists and skips
    generation if the MAR file already exists

    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel dataclass
    """
    check_path = os.path.join(
        dl_model.mar_utils.mar_output, f"{dl_model.mar_utils.mar_name}.mar"
    )
    if os.path.exists(check_path):
        print(
            f"## Skipping MAR file generation as it already exists\n"
            f" Model name: {dl_model.model_name}\n Repository Version: "
            f"{dl_model.repo_info.repo_version}\n"
        )
        sys.exit(1)


def check_if_model_files_exist(dl_model):
    """
    This function compares the list of files in the downloaded model directory with the
    list of files in the HuggingFace repository. It takes into account any files to
    ignore based on predefined extensions.

    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel dataclass

    Returns:
        bool: True if the downloaded model files match the expected
              repository files, False otherwise.
    """
    extra_files_list = get_all_files_in_directory(dl_model.mar_utils.model_path)
    hf_api = hfh.HfApi()
    repo_files = hf_api.list_repo_files(
        repo_id=dl_model.repo_info.repo_id,
        revision=dl_model.repo_info.repo_version,
        token=dl_model.repo_info.hf_token,
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
    dir_name = f"tmp_{mar_name}"
    tmp_dir = os.path.join(mar_output, dir_name)
    rm_dir(tmp_dir)  # delete existing tmp if it exists
    create_folder_if_not_exists(tmp_dir)
    return tmp_dir


def move_mar(dl_model, tmp_dir):
    """
    This funtion moves MAR file from the temporary directory to model store.

    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel dataclass
        tmp_dir (str): Path of temporary directory.
    """
    old_filename = f"{dl_model.model_name}.mar"
    new_filename = f"{dl_model.mar_utils.mar_name}.mar"
    src = os.path.join(tmp_dir, old_filename)
    dst = os.path.join(dl_model.mar_utils.mar_output, new_filename)
    check_if_path_exists(src, "Generated mar file is missing", is_dir=False)
    mv_file(src, dst)


def read_config_for_download(dl_model):
    """
    This function reads repo id, version and handler name from
    model_config.json and sets values for the DownloadDataModel object.

    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel dataclass

    Raises:
        sys.exit(1): If model name, repo_id or repo_version is not valid, the
                     function will terminate the program with an exit code of 1.
    """
    check_if_path_exists(MODEL_CONFIG_PATH, "Model Config", is_dir=False)
    with open(MODEL_CONFIG_PATH, encoding="UTF-8") as config:
        models = json.loads(config.read())
        if dl_model.model_name in models:
            try:
                # Read and validate the repo_id and repo_version
                dl_model.repo_info.repo_id = models[dl_model.model_name]["repo_id"]
                if not dl_model.repo_info.repo_version:
                    dl_model.repo_info.repo_version = models[dl_model.model_name][
                        "repo_version"
                    ]

                # Make sure there is HF hub token for LLAMA(2)
                if (
                    dl_model.repo_info.repo_id.startswith("meta-llama")
                    and dl_model.repo_info.hf_token is None
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
                    repo_id=dl_model.repo_info.repo_id,
                    revision=dl_model.repo_info.repo_version,
                    token=dl_model.repo_info.hf_token,
                )

                # Read handler file name
                if not dl_model.mar_utils.handler_path:
                    dl_model.mar_utils.handler_path = os.path.join(
                        os.path.dirname(__file__),
                        models[dl_model.model_name]["handler"],
                    )

            except (KeyError, HfHubHTTPError):
                print(
                    "## Error: Please check either repo_id, repo_version"
                    " or HuggingFace ID is not correct\n"
                )
                sys.exit(1)
        else:
            print(
                "## Please check your model name, it should be one of the following : "
            )
            print(list(models.keys()))
            sys.exit(1)


def run_download(dl_model):
    """
    This function checks if the given model path directory is empty and then
    downloads the given version's model files at that path.

    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel dataclass

    Returns:
        DownloadDataModel: An instance of the DownloadDataModel class.
    """
    if not check_if_folder_empty(dl_model.mar_utils.model_path):
        print("## Make sure the path provided to download model files is empty\n")
        sys.exit(1)

    print(
        f"\n## Starting model files download from {dl_model.repo_info.repo_id}"
        f" with version {dl_model.repo_info.repo_version}\n"
    )

    hfh.snapshot_download(
        repo_id=dl_model.repo_info.repo_id,
        revision=dl_model.repo_info.repo_version,
        local_dir=dl_model.mar_utils.model_path,
        local_dir_use_symlinks=False,
        token=dl_model.repo_info.hf_token,
        ignore_patterns=get_ignore_pattern_list(FILE_EXTENSIONS_TO_IGNORE),
    )
    print("## Successfully downloaded model_files\n")
    return dl_model


def create_mar(dl_model):
    """
    This function checks if the Model Archive (MAR) file for the downloaded
    model exists in the specified model path otherwise generates the MAR file.
    The MAR file is generated in a temporary folder and then moved to model store
    to avoid conflicts.

    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel dataclass
    """
    if not check_if_model_files_exist(dl_model):
        print("## Model files do not match HuggingFace repository files")
        sys.exit(1)

    # Creates a temporary directory with the mar_name inside model_store
    tmp_dir = create_tmp_model_store(
        dl_model.mar_utils.mar_output, dl_model.mar_utils.mar_name
    )

    generate_mars(
        dl_model=dl_model,
        mar_config=MODEL_CONFIG_PATH,
        model_store_dir=tmp_dir,
        debug=dl_model.debug,
    )

    # Move MAR file to model_store
    move_mar(dl_model, tmp_dir)
    # Delete temporary folder
    rm_dir(tmp_dir)
    print(
        f"\n## Mar file for {dl_model.model_name}"
        f" with version {dl_model.repo_info.repo_version} is generated!\n"
    )


def run_script(params):
    """
    This function validates input parameters and downloads model files,
    creates model archive file (MAR file) for the given model.

    Args:
        params (Namespace): An argparse.Namespace object containing command-line arguments.
                These are the necessary parameters and configurations for the script.
    """
    dl_model = set_values(params)
    check_if_path_exists(dl_model.mar_utils.model_path, "model_path", is_dir=True)
    check_if_path_exists(dl_model.mar_utils.mar_output, "mar_output", is_dir=True)
    check_if_mar_exists(dl_model)

    if not dl_model.skip_download:
        dl_model = run_download(dl_model)
    create_mar(dl_model)


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
