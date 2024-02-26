"""
This module stores the dataclasses InferenceDataModel, TorchserveStartData
and function prepare_settings to set the InferenceDataModel's ts_data.
"""

import argparse
import os
import dataclasses


@dataclasses.dataclass
class TorchserveStartData:
    """
    This dataclass stores information about logs, config file and model store
    required to start torchserve.

    Attributes:
        ts_log_file (str): Path to log file.
        ts_log_config (str): Path to log config file.
        ts_config_file (str): Path to Torchserve config file.
        ts_model_store (str): Path to model store.
    """

    ts_log_file = str()
    ts_log_config = str()
    ts_config_file = str()
    ts_model_store = str()


class InferenceDataModel:
    """
    This class stores information necessary to start Torchserve and run inference and
    it's related functions.

    Attributes:
        model_name (str): Model name.
        repo_version (str): Commit ID of model's repository from HuggingFace.
        input_path (str): Path of input data folder.
        gen_folder (str): Path of temporary gen folder to store logs.
        mar_filepath (str): Absolute path to the MAR file.
        ts_data (TorchserveStartData): Instance of TorchserveStartData dataclass.
    """

    model_name = str()
    repo_version = str()
    input_path = str()
    gen_folder = str()
    mar_filepath = str()
    ts_data = TorchserveStartData()

    def __init__(self, params: argparse.Namespace) -> None:
        """
        This is the init method that calls set_data_model method.

        Args:
            params: An argparse.Namespace object containing command-line arguments.
        """
        self.set_data_model(params)

    def set_data_model(self, args: argparse.Namespace) -> None:
        """
        This method sets model_name, input_path, gen_folder, mar_filepath,
        repo_version attributes of the InferenceDataModel class.

        Args:
            args (Namespace): Arguments returned by the ArgumentParser.
        """
        self.model_name = args.model_name
        self.input_path = args.data
        self.gen_folder = args.gen_folder_name
        self.mar_filepath = args.mar
        self.repo_version = args.repo_version

    def prepare_settings(self) -> None:
        """
        This method sets ts_data attribute of InferenceDataModel class,
        sets environment variables LOG_LOCATION and METRICS_LOCATION and makes gen folder.

        Args:
            data_model (InferenceDataModel): Instance of InferenceDataModel.
        """
        dir_path = os.path.dirname(__file__)
        self.ts_data.ts_log_file = os.path.join(
            dir_path, self.gen_folder, "logs/ts_console.log"
        )
        self.ts_data.ts_log_config = os.path.join(dir_path, "../log4j2.xml")
        self.ts_data.ts_config_file = os.path.join(dir_path, "../config.properties")
        self.ts_data.ts_model_store = os.path.dirname(self.mar_filepath)
        gen_path = os.path.join(dir_path, self.gen_folder, "logs")
        os.environ["LOG_LOCATION"] = gen_path
        os.environ["METRICS_LOCATION"] = gen_path
        os.makedirs(os.path.join(dir_path, self.gen_folder, "logs"), exist_ok=True)
