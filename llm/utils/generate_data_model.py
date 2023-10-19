"""
This module stores the dataclasses GenerateDataModel, MarUtils, RepoInfo
and function set_values that sets the GenerateDataModel attributes.
"""

import dataclasses


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
class GenerateDataModel:
    """
    This dataclass stores information regarding model download and MAR file generation.

    Attributes:
        model_name (str): Name of the model.
        skip_download (bool): Set to skip download model.
        is_custom_model (bool): Set if custom model is used.
        mar_utils (MarUtils): Contains data regarding MAR file generation.
        repo_info (RepoInfo): Contains data regarding HuggingFace model repository.
        debug (bool): Flag to print debug statements.
    """

    model_name = str()
    skip_download = bool()
    is_custom_model = bool()
    mar_utils = MarUtils()
    repo_info = RepoInfo()
    debug = bool()


def set_values(params):
    """
    This function sets values for the GenerateDataModel object based on the command-line arguments.

    Args:
        params: An argparse.Namespace object containing command-line arguments.

    Returns:
        GenerateDataModel: An instance of the GenerateDataModel dataclass
    """
    gen_model = GenerateDataModel()
    gen_model.model_name = params.model_name
    gen_model.skip_download = params.no_download
    gen_model.debug = params.debug

    gen_model.repo_info.hf_token = params.hf_token
    gen_model.repo_info.repo_version = params.repo_version

    gen_model.mar_utils.handler_path = params.handler_path
    gen_model.mar_utils.model_path = params.model_path
    gen_model.mar_utils.mar_output = params.mar_output
    return gen_model
