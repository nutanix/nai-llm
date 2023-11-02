"""
This module stores the dataclasses GenerateDataModel, MarUtils, RepoInfo
and function set_values that sets the GenerateDataModel attributes.
"""

import argparse
import os
import dataclasses
import sys


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


class GenerateDataModel:
    """
    This class stores information regarding model download and MAR file generation and
    related methods.

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

    def __init__(self, params: argparse.Namespace) -> None:
        """
        This is the init method that calls set_values method.

        Args:
            params: An argparse.Namespace object containing command-line arguments.
        """
        self.set_values(params)

    def set_values(self, params: argparse.Namespace) -> None:
        """
        This method sets values for the GenerateDataModel object based on the
        command-line arguments.

        Args:
            params: An argparse.Namespace object containing command-line arguments.
        """
        self.model_name = params.model_name
        self.skip_download = params.no_download
        self.debug = params.debug

        self.repo_info.hf_token = params.hf_token
        self.repo_info.repo_version = params.repo_version

        self.mar_utils.handler_path = params.handler_path
        self.mar_utils.model_path = params.model_path
        self.mar_utils.mar_output = params.mar_output

    def check_if_mar_exists(self) -> None:
        """
        This method checks if MAR file of a model already exists and skips
        generation if the MAR file already exists
        """
        check_path = os.path.join(
            self.mar_utils.mar_output, f"{self.mar_utils.mar_name}.mar"
        )
        if os.path.exists(check_path):
            print(
                f"## Skipping MAR file generation as it already exists\n"
                f" Model name: {self.model_name}\n Repository Version: "
                f"{self.repo_info.repo_version}\n"
            )
            sys.exit(1)
