import logging
import os
from abc import ABC
import torch
import transformers
import json

from collections import defaultdict
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

class LLMHandler(BaseHandler, ABC):

    def __init__(self):
        super(LLMHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        if os.environ.get('TS_NUMBER_OF_GPU') and torch.cuda.is_available() and properties.get("gpu_id") is not None: # torchserve sets gpu_id in round robin fashion for workers
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir,
                                                       local_files_only = True,
                                                       device_map = self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token # to avoid an error during batching
        self.tokenizer.padding_side = "left"
        logger.info("Tokenizer loaded successfully")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype = torch.bfloat16, # Load model weights in bfloat16
            device_map = self.device,
            local_files_only = True,
            trust_remote_code = True
        )

        logger.info("Model loaded successfully")
        self.initialized = True
        logger.info("Initialized TorchServe Server!")

    def preprocess(self, requests):
        self.request_list = defaultdict(int)
        self.request_ids = defaultdict(int)
        self.request_type = defaultdict(int)
        input_list = []

        for idx, data in enumerate(requests):

            # Pre-process for Kserve v2 format
            if isinstance(data, dict):
                if "inputs" in data:
                    self.request_type[idx] = "kservev2"
                    self.request_ids[idx] = data.get("id") if data.get("id") else "" # Kserve wrapper validates ID, setting empty if not sent in request
                    
                    for row_data in data.get("inputs"):
                        self.request_list[idx] += 1                                  # To handle multiple inputs inside a single request use-case
                        input_text = row_data.get('data')[0]
                        
                        if isinstance(input_text, (bytes, bytearray)):
                            input_text = input_text.decode("utf-8")
                        input_list.append(input_text)
                
                    
            else:
                if isinstance(data, (bytes, bytearray)):
                    row_input = data.decode("utf-8")
                
                # Set as raw for non kserve requests
                self.request_type[idx] = "raw"
                input_list.append(row_input)
        
        logger.info("Recieved text: {}".format(', '.join(map(str, input_list))))
        encoded_input = self.tokenizer(input_list, padding=True, return_tensors='pt')["input_ids"].to(self.device)
        return encoded_input

    def inference(self, input_batch):
        logger.info("Running Inference")
        encoding = input_batch
        logger.info("Generating text")
        generated_ids = self.model.generate(encoding, max_new_tokens = 200)
        inference=[]
        inference = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logger.info("Generated text is: {}".format(', '.join(map(str, inference))))
        return inference

    def postprocess(self, inference_output):
        response_list = []
        idx = 0
        data = []

        for result in inference_output:
            #For raw request - response
            if self.request_type[idx] == 'raw':
                response_list.append(result)
                continue

            #For Kserve v2 response
            data.append(result)
            self.request_list[idx] -= 1
            if self.request_list[idx]:
                continue

            else:
                response = {}
                response["id"] = self.request_ids[idx]
                response["model_name"] = self.context.manifest.get("model").get(
                    "modelName")
                response["model_version"] = self.context.manifest.get("model").get(
                    "modelVersion")
                response["outputs"] = self._batch_to_json(data)
                
                response_list.append(response)
                idx += 1
                data = []

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
        output_data["name"] = ("explain" if self.context.get_request_header(
            0, "explain") == "True" else "predict")
        output_data["shape"] = [-1]
        output_data["datatype"] = "BYTES"
        output_data["data"] = [data]
        return output_data

    def _get_json(self, jsonData):
        try:
            logger.info(jsonData)
            print(jsonData)
            data = json.loads(jsonData)
        except ValueError as err:
            return False
        return data