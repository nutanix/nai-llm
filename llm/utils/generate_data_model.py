"""
This module stores the dataclasses GenerateDataModel, MarUtils, RepoInfo
and function set_values that sets the GenerateDataModel attributes.
"""

import argparse
import os
import dataclasses
import sys
import huggingface_hub as hfh
from huggingface_hub.utils import (
    HfHubHTTPError,
    HFValidationError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


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
        self.skip_download = params.skip_download
        self.debug = params.debug

        self.repo_info.hf_token = params.hf_token or os.environ.get("HF_TOKEN")
        self.repo_info.repo_id = params.repo_id
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

    def validate_hf_token(self) -> None:
        """
        This method makes sure there is HuggingFace token is valid
        for the Meta models (Llama models)
        """
        # Make sure there is HF hub token for LLAMA(2)
        if (
            self.repo_info.repo_id.startswith("meta-llama")
            and self.repo_info.hf_token is None
        ):
            print(
                (
                    "HuggingFace Hub token is required for llama download. "
                    "Please specify it using --hf_token=<your token> argument "
                    "or set it as an environment variable 'HF_TOKEN'. Refer "
                    "https://huggingface.co/docs/hub/security-tokens"
                )
            )
            sys.exit(1)

    def validate_commit_info(self) -> None:
        """
        This method validates the HuggingFace repository information and
        sets the latest commit ID of the model if repo_version is None.
        """
        try:
            hf_api = hfh.HfApi()
            commit_info = hf_api.list_repo_commits(
                repo_id=self.repo_info.repo_id,
                revision=self.repo_info.repo_version,
                token=self.repo_info.hf_token,
            )

            # Set repo_version to latest commit ID if it is None
            if not self.repo_info.repo_version:
                self.repo_info.repo_version = commit_info[0].commit_id

        except (HfHubHTTPError, HFValidationError):
            print(
                "## Error: Please check either repo_id, repo_version"
                " or HuggingFace ID is not correct\n"
            )
            sys.exit(1)

    def get_repo_file_extensions(self) -> set:
        """
        This function returns set of all file extensions in the Hugging Face repo of
        the model.
        Returns:
            repo_file_extension (set): The set of all file extensions in the
                                       Hugging Face repo of the model
        Raises:
            sys.exit(1): If repo_id, repo_version or huggingface token
                        is not valid, the function will terminate
                        the program with an exit code of 1.
        """
        try:
            hf_api = hfh.HfApi()
            repo_files = hf_api.list_repo_files(
                repo_id=self.repo_info.repo_id,
                revision=self.repo_info.repo_version,
                token=self.repo_info.hf_token,
            )
            return {os.path.splitext(file_name)[1] for file_name in repo_files}
        except (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
            HfHubHTTPError,
            HFValidationError,
            ValueError,
            KeyError,
        ):
            print(
                "## Error: Please check either repo_id, repo_version"
                " or HuggingFace ID is not correct\n"
            )
            sys.exit(1)
