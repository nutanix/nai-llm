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

    def inference(self, input_batch):
        logger.info("Running Inference")
        encoding = input_batch
        logger.info("Generating text")
        temperature = os.environ.get("TS_TEMPERATURE")
        if temperature is not None and temperature.strip():
            try:
                temperature= float(temperature)
            except ValueError:
                print(f"Warning: Unable to convert 'temperature' to an float.")
        else:
            temperature=1.0
        print("temp "+str(temperature))
        rep_penalty= os.environ.get("TS_REP_PENALTY")
        if rep_penalty is not None and rep_penalty.strip():
            try:
                rep_penalty= float(rep_penalty)
            except ValueError:
                print(f"Warning: Unable to convert 'repition_penalty' to an float.")
        else:
            rep_penalty=1.0
        print(rep_penalty)
        top_p=os.environ.get("TS_TOP_P")
        if top_p is not None and top_p.strip():
            try:
                top_p= float(top_p)
            except ValueError:
                print(f"Warning: Unable to convert 'top_p' to an float.")
        else:
            top_p=1.0
        print(top_p)
        max_tokens=os.environ.get("TS_MAX_TOKENS")
        if max_tokens is not None and max_tokens.strip():
            try:
                max_tokens= int(max_tokens)
            except ValueError:
                print(f"Warning: Unable to convert 'max_tokens' to an integer.")
        else:
            max_tokens=200
        print(max_tokens)
        if temperature==1.0 and rep_penalty==1.0 and top_p==1.0:
            generated_ids = self.model.generate(encoding, max_new_tokens = max_tokens)
        else:
            print("non default")
            generated_ids = self.model.generate(encoding, max_new_tokens = max_tokens, pad_token_id = self.tokenizer.eos_token_id, eos_token_id = self.tokenizer.eos_token_id, repetition_penalty=rep_penalty, do_sample = True, temperature=temperature, top_p=top_p)
        
        inference=[]
        inference = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logger.info("Generated text is: {}".format(', '.join(map(str, inference))))
        return inference

    def postprocess(self, inference_output):
        return inference_output