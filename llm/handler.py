import logging
import os
from abc import ABC
import torch
import transformers

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
        input_list = []
        for _, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            input_list.append(input_text)

        logger.info("Recieved text: {}".format(', '.join(map(str, input_list))))
        encoded_input = self.tokenizer(input_list, padding=True, return_tensors='pt')["input_ids"].to(self.device)
        return encoded_input
    
    def getenv(self, param_dict):
        generation_params ={"NAI_TEMPERATURE":"temperature", 
                            "NAI_REP_PENALTY":"repetition_penalty",
                            "NAI_TOP_P":"top_p", 
                            "NAI_MAX_TOKENS":"max_new_tokens"}
        
        for var_name, var_value in generation_params.items():
            param_dict[var_value]=os.environ.get(var_name)
            if param_dict[var_value] is not None and param_dict[var_value].strip():
                try:
                    param_dict[var_value]= float(param_dict[var_value])
                except ValueError:
                    print(f"Warning: Unable to convert {var_value} {param_dict[var_value]} to an float.")
        
        
        new_param_dict = {}
        for key, value in param_dict.items():
            if value is not None:
                new_param_dict[key] = value
            else:
                if key=="max_new_tokens":
                    new_param_dict[key]=200
            
        return new_param_dict

    def inference(self, input_batch):
        logger.info("Running Inference")
        encoding = input_batch
        logger.info("Generating text")
        param_dict={}
        param_dict=self.getenv(param_dict)
        param_dict['pad_token_id']=self.tokenizer.eos_token_id
        param_dict['eos_token_id']=self.tokenizer.eos_token_id
        param_dict['do_sample']=True
        
        generated_ids = self.model.generate(encoding, **param_dict)
        
        inference=[]
        inference = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logger.info("Generated text is: {}".format(', '.join(map(str, inference))))
        return inference

    def postprocess(self, inference_output):
        return inference_output