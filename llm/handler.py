"""
Serves as a handler for a LLM, allowing it to be used in an inference service.
The handler provides functions to preprocess input data, make predictions using the model, 
and post-process the output for a particular use case.
"""
import logging
import os
from abc import ABC
from collections import defaultdict
import torch
import transformers
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class LLMHandler(BaseHandler, ABC):
    """
    This is a derived class that inherits from BaseHandler class.
    It provides functions to initialize handler attributes,
    preprocess input data, run inference using the model,
    and post-process the output for the deployed model.
    Attributes:
        initialized (bool): Flag indicating that Torchserve is initialized.
        device (torch.device): The device that is being used for generation.
        map_location(str): The type of device where the model is
                            currently loaded (e.g., 'cpu' or 'cuda').
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        model (transformers.PreTrainedModel): The loaded Hugging Face model instance.
    Methods:
        __init__():
            This method initializes some attributes for an instance of LLMHandler
            for the specified model.
        initialize():
            This method loads the Hugging Face model and tokenizer based on
            the provided model name and model files present in MAR file.
        preprocess(text: str) -> Tensor:
            This method tokenizes input text using the associated tokenizer.
            Args:
                text (str): The input text to be tokenized.
            Returns:
                Tensor: Tokenized input data
        inference(data: Tensor) -> list(str):
            This method reads the generation parameters set as environment vairables
            and uses the preprocessed tokens and generation parameters to generate a
            output text.
            Args:
                data (Tensor): The input Tensor of encoded tokens for which generation is run.
            Returns:
                list(str): A list containing model's generated output.
        postprocess(data: list(str)) -> list(str):
            This method returns the list of generated text recieved.
            Args:
                data (list(str)): A list containing the output text of model generation.
            Returns:
                list(str): A list containing model's generated output.
        _batch_to_json(data: list(str)) -> list(str):
        _to_json(data: (str)) -> json(str):
        get_env_value(str) -> float:
            This method reads the inputed environment variable and converts it to float
            and returns it. This is used for reading model generation parameters.
            Args:
                env_var (str): Environment variable to be read.
            Returns:
                float: Value of the enviroment variable read.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.request = {
            "request_list": defaultdict(int),
            "request_ids": defaultdict(int),
            "request_type": defaultdict(int),
        }
        self.tokenizer = None
        self.map_location = None
        self.device = None
        self.model = None
        self.device_map = None

    def initialize(self, context):
        """
        This method loads the Hugging Face model and tokenizer based on
        the provided model name and model files present in MAR file.
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.device = torch.device("cuda")
            self.device_map = "auto"
        else:
            self.device = self.device_map = torch.device("cpu")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, device_map=self.device_map
        )
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # to avoid an error during batching
        self.tokenizer.padding_side = "left"
        logger.info("Tokenizer loaded successfully")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
            device_map=self.device_map,
            local_files_only=True,
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully")
        self.initialized = True
        logger.info("Initialized TorchServe Server!")

    def preprocess(self, data):
        """
        This method tookenizes input text using the associated tokenizer.
        Args:
            text (str): The input text to be tokenized.
        Returns:
            Tensor: Tokenized input data
        """
        input_list = []

        for idx, input_data in enumerate(data):
            # Pre-process for Kserve v2 format
            if isinstance(input_data, dict):
                if "inputs" in input_data:
                    self.request["request_type"][idx] = "kservev2"
                    self.request["request_ids"][idx] = (
                        input_data.get("id") if input_data.get("id") else ""
                    )
                    # Kserve wrapper validates ID, setting empty if not sent in request
                    # To handle multiple inputs inside a single request use-case
                    for row_data in input_data.get("inputs"):
                        self.request["request_list"][idx] += 1
                        input_text = row_data.get("data")[0]

                        if isinstance(input_text, (bytes, bytearray)):
                            input_text = input_text.decode("utf-8")
                        input_list.append(input_text)

            else:
                if isinstance(input_data, (bytes, bytearray)):
                    row_input = input_data.decode("utf-8")

                # Set as raw for non kserve requests
                self.request["request_type"][idx] = "raw"
                input_list.append(row_input)

        logger.info("Received text: %s", ", ".join(map(str, input_list)))
        encoded_input = self.tokenizer(input_list, padding=True, return_tensors="pt")[
            "input_ids"
        ].to(self.device)

        return encoded_input

    def inference(self, data, *args, **kwargs):
        """
        This method reads the generation parameters set as environment vairables
        and uses the preprocessed tokens and generation parameters to generate a
        output text.
        Args:
            data (Tensor): The input Tensor of encoded tokens for which generation is run.
        Returns:
            list(str): A list containing model's generated output.
        """
        logger.info("Running Inference")
        encoding = data
        logger.info("Generating text")
        param_dict = {}
        if os.environ.get("NAI_TEMPERATURE"):
            param_dict["temperature"] = self.get_env_value("NAI_TEMPERATURE")

        if os.environ.get("NAI_REP_PENALTY"):
            param_dict["repetition_penalty"] = self.get_env_value("NAI_REP_PENALTY")

        if os.environ.get("NAI_TOP_P"):
            param_dict["top_p"] = self.get_env_value("NAI_TOP_P")

        if os.environ.get("NAI_MAX_TOKENS"):
            param_dict["max_new_tokens"] = self.get_env_value("NAI_MAX_TOKENS")
        else:
            param_dict["max_new_tokens"] = 200

        param_dict["pad_token_id"] = self.tokenizer.eos_token_id
        param_dict["eos_token_id"] = self.tokenizer.eos_token_id
        param_dict["do_sample"] = True

        generated_ids = self.model.generate(encoding, **param_dict)

        inference = []
        inference = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logger.info("Generated text is: %s", ", ".join(map(str, inference)))
        return inference

    def postprocess(self, data):
        """
        This method returns the list of generated text recieved.
        Args:
            data (list(str)): A list containing the output text of model generation.
        Returns:
            list(str): A list containing model's generated output.
        """
        response_list = []
        idx = 0
        inference_output = []

        for result in data:
            # For raw request - response
            if self.request["request_type"][idx] == "raw":
                response_list.append(result)
                continue

            # For Kserve v2 response
            inference_output.append(result)
            self.request["request_list"][idx] -= 1
            if self.request["request_list"][idx]:
                continue

            response = {}
            response["id"] = self.request["request_ids"][idx]
            response["model_name"] = self.context.manifest.get("model").get("modelName")
            response["model_version"] = self.context.manifest.get("model").get(
                "modelVersion"
            )
            response["outputs"] = self._batch_to_json(inference_output)

            response_list.append(response)
            idx += 1
            inference_output = []

        return response_list

    def _batch_to_json(self, data):
        """
        Splits batch output to json objects
        """
        output = []
        for item in data:
            output.append(self._to_json(item))
        return output

    def _to_json(self, data):
        """
        Constructs JSON object from data
        """
        output_data = {}
        output_data["name"] = (
            "explain"
            if self.context.get_request_header(0, "explain") == "True"
            else "predict"
        )
        output_data["shape"] = [-1]
        output_data["datatype"] = "BYTES"
        output_data["data"] = [data]
        return output_data

    def get_env_value(self, env_var):
        """
        This function gets the value of an environment variable as a float.
        Args:
            env_var (str): The name of the environment variable to retrieve.

        Returns:
            float or None: The float value of the environment variable
                           if conversion is successful, or None otherwise.
        """
        value = os.environ.get(env_var)
        if value is not None and value.strip():
            try:
                value = float(value)
            except ValueError:
                print(f"Warning: Unable to convert {env_var} {value} to an float.")
        return value
