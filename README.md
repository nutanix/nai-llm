# NAI-LLM

## Setup

Install openjdk, pip3:
```
sudo apt-get install openjdk-17-jdk python3-pip
```

Install required packages:

```
pip install -r requirements.txt
```

Install NVIDIA Drivers:

[Installation Reference](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#runfile) <br />

Download the latest [Datacenter Nvidia drivers](https://www.nvidia.com/download/index.aspx) for the GPU type

For Nvidia A100, Select A100 in Datacenter Tesla for Linux 64 bit with cuda toolkit 11.7, latest driver is 515.105.01

```
curl -fSsl -O https://us.download.nvidia.com/tesla/515.105.01/NVIDIA-Linux-x86_64-515.105.01.run
sudo sh NVIDIA-Linux-x86_64-515.105.01.run -s
```

Note: We don’t need to install CUDA toolkit separately as it is bundled with PyTorch installation. Just Nvidia driver installation is enough. 


## Scripts

### Download model files and Generate MAR file
Run the following command for downloading model files and/or generating MAR file: 
```
python3 download.py [--no_download --repo_version <REPO_VERSION>] --model_name <MODEL_NAME> --model_path <MODEL_PATH> --mar_output <MAR_EXPORT_PATH> --hf_token <Your_HuggingFace_Hub_Token>
```
- no_download:      Set flag to skip downloading the model files
- model_name:       Name of model
- repo_version:     Commit ID of model's repo from HuggingFace repository (optional, if not provided default set in model_config will be used)
- model_path:       Absolute path of model files (should be empty if downloading)
- mar_output:       Absolute path of export of MAR file (.mar)
- hf_token:         Your HuggingFace token. Needed to download and verify LLAMA(2) models.

The available LLMs are mpt_7b, falcon_7b, llama2_7b

#### Examples
Download MPT-7B model files and generate model archive for it:
```
python3 llm/download.py --model_name mpt_7b --model_path /home/ubuntu/models/mpt_7b/model_files --mar_output /home/ubuntu/models/model_store
```
Download Falcon-7B model files and generate model archive for it:
```
python3 llm/download.py --model_name falcon_7b --model_path /home/ubuntu/models/falcon_7b/model_files --mar_output /home/ubuntu/models/model_store
```
Download Llama2-7B model files and generate model archive for it:
```
python3 llm/download.py --model_name llama2_7b --model_path /home/ubuntu/models/llama2_7b/model_files --mar_output /home/ubuntu/models/model_store --hf_token <Your_HuggingFace_Hub_Token>
```

### Start Torchserve and run inference
Run the following command for starting Torchserve and running inference on the given input:
```
bash run.sh -n <MODEL_NAME> -a <MAR_EXPORT_PATH> [OPTIONAL -d <INPUT_PATH> -v <REPO_VERSION>]
```
- n:    Name of model
- v:    Commit ID of model's repo from HuggingFace repository (optional, if not provided default set in model_config will be used)
- d:    Absolute path of input data folder (optional)
- a:    Absolute path to the Model Store directory

For model names, we support MPT-7B, Falcon-7b and Llama2-7B.
Should print "Ready For Inferencing" as a message at the end

#### Examples
For 1 GPU Inference with official MPT-7B model:
```
bash llm/run.sh -n mpt_7b -d data/translate -a /home/ubuntu/models/model_store
```
For 1 GPU Inference with official Falcon-7B model:
```
bash llm/run.sh -n falcon_7b -d data/qa -a /home/ubuntu/models/model_store
```
For 1 GPU Inference with official Llama2-7B model:
```
bash llm/run.sh -n llama2_7b -d data/summarize -a /home/ubuntu/models/model_store
```

### Describe registered model
curl http://{inference_server_endpoint}:{management_port}/models/{model_name} <br />

For MPT-7B model
```
curl http://localhost:8081/models/mpt_7b
```
For Falcon-7B model
```
curl http://localhost:8081/models/falcon_7b
```
For Llama2-7B model
```
curl http://localhost:8081/models/llama2_7b
```

### Inference Check
curl http://{inference_server_endpoint}:{inference_port}/predictions/{model_name} -T {input_file} <br />

Test input file can be found in the data folder. <br />

For MPT-7B model
```
curl http://localhost:8080/predictions/mpt_7b -T data/qa/sample_test1.txt
```
For Falcon-7B model
```
curl http://localhost:8080/predictions/falcon_7b -T data/summarize/sample_test1.txt
```
For Llama2-7B model
```
curl http://localhost:8080/predictions/llama2_7b -T data/translate/sample_test1.txt
```
### Register additional models
For loading multiple unique models, make sure that the MAR files (.mar) for the concerned models are stored in the same directory <br />

curl -X POST "http://{inference_server_endpoint}:{management_port}/models?url={mar_name}.mar&initial_workers=1&synchronous=true"
Test input file can be found in the data folder. <br />

For MPT-7B model
```
curl -X POST "http://localhost:8081/models?url=mpt_7b.mar&initial_workers=1&synchronous=true"
```
For Falcon-7B model
```
curl -X POST "http://localhost:8081/models?url=falcon_7b.mar&initial_workers=1&synchronous=true"
```
For Llama2-7B model
```
curl -X POST "http://localhost:8081/models?url=llama2_7b.mar&initial_workers=1&synchronous=true"
```

### Edit registered model configuration
curl -v -X PUT "http://{inference_server_endpoint}:{management_port}/models/{model_name}?min_workers={number}&max_workers={number}&batch_size={number}&max_batch_delay={delay_in_ms}"

For MPT-7B model
```
curl -v -X PUT "http://localhost:8081/models/mpt_7b?min_worker=3&max_worker=6"
```
For Falcon-7B model
```
curl -v -X PUT "http://localhost:8081/models/falcon_7b?min_worker=3&max_worker=6"
```
For Llama2-7B model
```
curl -v -X PUT "http://localhost:8081/models/llama2_7b?min_worker=3&max_worker=6"
```

### Unregister a model
curl -X DELETE "http://{inference_server_endpoint}:{management_port}/models/{model_name}/{repo_version}"

### Stop Torchserve and Cleanup
If keep alive flag was set in the bash script, then you can run the following command to stop the server and clean up temporary files
```
python3 llm/cleanup.py
```

## Model Version Support

We provide the capability to download and register various commits of the single model from HuggingFace. By specifying the commit ID as "repo_version", you can produce MAR files for multiple iterations of the same model and register them simultaneously. To transition between these versions, you can set a default version within Torchserve while it is running.

### Set Default Model Version
If multiple versions of the same model are registered, we can set a particular version as the default for inferencing<br />

curl -v -X PUT "http://{inference_server_endpoint}:{management_port}/{model_name}/{repo_version}/set-default"

## Custom Model Support

We provide the capability to generate a MAR file with custom models and start an inference server using it with Torchserve.<br />

### Generate MAR file for custom model
To generate the MAR file, run the following:
```
python3 download.py --no_download [--repo_version <REPO_VERSION> --handler <CUSTOM_HANDLER_PATH>] --model_name <CUSTOM_MODEL_NAME> --model_path <MODEL_PATH> --mar_output <MAR_EXPORT_PATH>
```
- no_download:      Set flag to skip downloading the model files, must be set for custom models
- model_name:       Name of custom model
- repo_version:     Any model version, defaults to "1.0" (optional)
- model_path:       Absolute path of custom model files (should be empty non empty)
- mar_output:       Absolute path of export of MAR file (.mar) 
- handler:          Path to custom handler, defaults to llm/handler.py (optional)<br />

### Start Torchserve and run inference for custom model
To start Torchserve and run inference on the given input with a custom MAR file, run the following:
```
bash run.sh -n <CUSTOM_MODEL_NAME> -a <MAR_EXPORT_PATH> [OPTIONAL -d <INPUT_PATH>]
```
- n:    Name of custom model 
- d:    Absolute path of input data folder (optional)
- a:    Absolute path to the Model Store directory