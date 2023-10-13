"""
handler
This module serves as a handler for a LLM, allowing it to be used in an inference service.
The handler provides functions to preprocess input data, make predictions using the model,
and post-process the output for a particular use case.

"""
import logging
import os
from abc import ABC
import torch
import transformers

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


def get_env_value(env_var):
    """
    get_env_value
    This function reads the inputed environment variable and converts it to float
    and returns it. This is used for reading model generation parameters.

    Args:
        env_var (str): Environment variable to be read.

    Returns:
        float: Value of the enviroment variable read.

    Raises:
        ValueError: Throws an error if the environment variable read can't be
                    converted to float
    """
    value = os.environ.get(env_var)
    if value is not None and value.strip():
        try:
            value = float(value)
        except ValueError:
            print(f"Warning: Unable to convert {env_var} {value} to an float.")
    return value


class LLMHandler(BaseHandler, ABC):
    """
    This is a derived class that inherits from BaseHandler class.
    It provides functions to initialize handler attributes,
    preprocess input data, run inference using the model,
    and post-process the output for the deployed model.

    Attributes:
        initialized (bool): Flag indicating that Torchserve is initialized.

        request_list (dict):

        request_ids (dict):

        request_type (dict):

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
            This method tookenizes input text using the associated tokenizer.
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
        _batch_to_json():
        _to_json():
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
        self.device = None
        self.tokenizer = None
        self.map_location = None
        self.model = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        # torchserve sets gpu_id in round robin fashion for workers
        if (
            os.environ.get("TS_NUMBER_OF_GPU")
            and torch.cuda.is_available()
            and properties.get("gpu_id") is not None
        ):
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, device_map=self.device
        )
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # to avoid an error during batching
        self.tokenizer.padding_side = "left"
        logger.info("Tokenizer loaded successfully")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
            device_map=self.device,
            local_files_only=True,
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully")
        self.initialized = True
        logger.info("Initialized TorchServe Server!")

    def preprocess(self, data):
        input_list = []
        for _, request_data in enumerate(data):
            input_text = request_data.get("data")
            if input_text is None:
                input_text = request_data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            input_list.append(input_text)

        logger.info("Recieved text: %s", ", ".join(map(str, input_list)))
        encoded_input = self.tokenizer(input_list, padding=True, return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        return encoded_input

    def inference(self, data, *args, **kwargs):
        logger.info("Running Inference")
        encoding = data
        logger.info("Generating text")
        param_dict = {}
        if os.environ.get("NAI_TEMPERATURE"):
            param_dict["temperature"] = get_env_value("NAI_TEMPERATURE")

        if os.environ.get("NAI_REP_PENALTY"):
            param_dict["repetition_penalty"] = get_env_value("NAI_REP_PENALTY")

        if os.environ.get("NAI_TOP_P"):
            param_dict["top_p"] = get_env_value("NAI_TOP_P")

        if os.environ.get("NAI_MAX_TOKENS"):
            param_dict["max_new_tokens"] = get_env_value("NAI_MAX_TOKENS")
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
        self.initialized = True
        return data
