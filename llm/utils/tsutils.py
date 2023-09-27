import os
import platform
import time
import json
import requests

torchserve_command = {
    "Windows": "torchserve.exe",
    "Darwin": "torchserve",
    "Linux": "torchserve"
}

torch_model_archiver_command = {
        "Windows": "torch-model-archiver.exe",
        "Darwin": "torch-model-archiver",
        "Linux": "torch-model-archiver"
    }

def generate_ts_start_cmd(ncs, model_store,
                          config_file, log_file,
                          log_config_file, gpus, debug):
    cmd = f"TS_SERVICE_ENVELOPE=body TS_NUMBER_OF_GPU={gpus} {torchserve_command[platform.system()]} --start --model-store={model_store}"
    if ncs:
        cmd += " --ncs"
    if config_file:
        cmd += f" --ts-config={config_file}"
    if log_config_file:
        cmd += f" --log-config {log_config_file}"
    if log_file:
        print(f"## Console logs redirected to file: {log_file} \n")
        dirpath = os.path.dirname(log_file)
        cmd += f" >> {os.path.join(dirpath,log_file)}"
    debug and print(f"## In directory: {os.getcwd()} | Executing command: {cmd} \n")
    return cmd


def start_torchserve(ncs=False, model_store="model_store",
        config_file="", log_file="", log_config_file="",
        wait_for=10, gpus=0, debug=False):

    print("## Starting TorchServe \n")
    cmd = generate_ts_start_cmd(ncs, model_store,
                                config_file, log_file,
                                log_config_file, gpus,
                                debug)
    print(cmd)
    status=os.system(cmd)
    if status == 0:
        print("## Successfully started TorchServe \n")
        time.sleep(wait_for)
        return (True)
    else:
        print("## TorchServe failed to start ! Make sure it's not running already\n")
        return (False)


def stop_torchserve(wait_for=10):
    try:
        requests.get('http://localhost:8080/ping')
    except Exception as e:
        return

    print("## Stopping TorchServe \n")
    cmd = f"{torchserve_command[platform.system()]} --stop"
    status = os.system(cmd)
    if status == 0:
        print("## Successfully stopped TorchServe \n")
        time.sleep(wait_for)
        return True
    else:
        print("## TorchServe failed to stop ! \n")
        return False


def get_params_for_registration(model_name):
    dirpath = os.path.dirname(__file__)
    initial_workers = batch_size = max_batch_delay = response_timeout = None

    with open(os.path.join(dirpath, '../model_config.json'), 'r') as f:
        model_config = json.loads(f.read())
        if model_name in model_config:
            param_config = model_config[model_name]["registration_params"]
            if "initial_workers" in param_config:
                initial_workers = param_config['initial_workers']

            if "batch_size" in param_config:
                batch_size = param_config['batch_size']

            if "max_batch_delay" in param_config:
                max_batch_delay = param_config['max_batch_delay']

            if "response_timeout" in param_config:
                response_timeout = param_config['response_timeout']

    return initial_workers, batch_size, max_batch_delay, response_timeout


def register_model(model_name, marfile, gpus, 
                   protocol="http", host="localhost",
                   port="8081"):
    print(f"\n## Registering {marfile} model, this might take a while \n")
    initial_workers, batch_size, max_batch_delay, response_timeout = get_params_for_registration(model_name)
    if(initial_workers is None): #setting the default value of workers to number of gpus
        initial_workers = gpus

    params = (
        ("url", marfile),
        ("initial_workers", initial_workers or 1),
        ("batch_size", batch_size or 1),
        ("max_batch_delay", max_batch_delay or 200),
        ("response_timeout", response_timeout or 2000),
        ("synchronous", "true"),
    )

    url = f"{protocol}://{host}:{port}/models"
    response = requests.post(url, params=params, verify=False)
    return response


def run_inference(model_name, file_name, protocol="http",
                  host="localhost", port="8080", timeout=120):
    print(f"## Running inference on {model_name} model \n")
    url = f"{protocol}://{host}:{port}/predictions/{model_name}"        
    files = {"data": (file_name, open(file_name, "rb"))}
    response = requests.post(url, files=files, timeout=timeout)
    print(response)
    return response

def run_inference_v2(model_name, file_name, protocol="http", 
                  host="localhost", port="8080", timeout=120, headers=None):
    print(f"## Running inference on {model_name} model \n")

    url = f"{protocol}://{host}:{port}/v2/models/{model_name}/infer"

    print("Url", url)
    with open(file_name, 'r') as f:
        data = json.load(f)
        print("Data", data, "\n")

    response = requests.post(url, json=data, headers=headers, timeout=timeout)
    print(response, "\n")
    return response


def unregister_model(model_name, protocol="http", host="localhost", port="8081"):
    print(f"## Unregistering {model_name} model \n")
    url = f"{protocol}://{host}:{port}/models/{model_name}"
    response = requests.delete(url, verify=False)
    return response
