import os
import sys
import subprocess
import traceback

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import tsutils as ts
import system_utils

def error_msg_print():
    print("\n**************************************")
    print("*\n*\n*  Error found - Unsuccessful ")
    print("*\n*\n**************************************")
    ts.stop_torchserve()

def get_model_name(input_dict, model_url):
    for entry in input_dict['models']:
        if model_url == entry['modelUrl']:
            return entry['modelName']

    print('\n model not found among registered models')
    error_msg_print()
    sys.exit(1)


def get_inputs_from_folder(input_path):
    return [os.path.join(input_path, item) for item in os.listdir(input_path)] if input_path else []


def set_compute_setting(gpus):
    if gpus > 0 and system_utils.is_gpu_instance():
        import torch

        if not torch.cuda.is_available():
            sys.exit("## Ohh its NOT running on GPU ! \n")
        print(f'\n## Running on {gpus} GPU(s) \n')

    else:
        print('\n## Running on CPU \n')
        gpus=0


def prepare_settings(data_model):
    data_model.dir_path = os.path.dirname(__file__)
    data_model.ts_log_file = os.path.join(data_model.dir_path,
                                          data_model.gen_folder,
                                          "logs/ts_console.log")
    data_model.ts_log_config = os.path.join(data_model.dir_path, "../log4j2.xml")
    data_model.ts_config_file = os.path.join(data_model.dir_path, '../config.properties')
    data_model.ts_model_store = os.path.dirname(data_model.mar_filepath)
    gen_path = os.path.join(data_model.dir_path, data_model.gen_folder, "logs")
    os.environ["LOG_LOCATION"] = gen_path
    os.environ["METRICS_LOCATION"] = gen_path
    os.makedirs(os.path.join(data_model.dir_path, data_model.gen_folder, "logs"), exist_ok=True)


def ts_health_check():
    os.system("curl localhost:8080/ping")


def start_ts_server(ts_model_store, ts_log_file, ts_log_config, ts_config_file, gpus, debug):
    started = ts.start_torchserve(model_store=ts_model_store,
                                        log_file=ts_log_file, 
                                        log_config_file=ts_log_config, 
                                        config_file=ts_config_file, 
                                        gpus=gpus,
                                        debug=debug)
    if not started:
        error_msg_print()
        sys.exit(1)

def execute_inference_on_inputs(model_inputs, model_name):
    for input in model_inputs:
        print(input)
        print(model_name)
        response = ts.run_inference(model_name, input)
        if response and response.status_code == 200:
            print(f"## Successfully ran inference on {model_name} model. \n\n Output - {response.text}\n\n")
        else:
            print(f"## Failed to run inference on {model_name} model \n")
            error_msg_print()
            sys.exit(1)


def register_model(model_name, input_mar, gpus):
    response = ts.register_model(model_name, input_mar, gpus)
    if response and response.status_code == 200:
        print(f"## Successfully registered {input_mar} model with torchserve \n")
    else:
        print("## Failed to register model with torchserve \n")
        error_msg_print()
        sys.exit(1)


def unregister_model(model_name):
    response = ts.unregister_model(model_name)
    if response and response.status_code == 200:
        print(f"## Successfully unregistered {model_name} \n")
    else:
        print(f"## Failed to unregister {model_name} \n")
        error_msg_print()
        sys.exit(1)


def validate_inference_model(models_to_validate, debug):
    for model in models_to_validate:
        model_name = model["name"]
        model_inputs = model["inputs"]
        
        execute_inference_on_inputs(model_inputs, model_name)

        debug and os.system(f"curl http://localhost:8081/models/{model_name}")
        print(f"## {model_name} Handler is stable. \n")


def get_inference_internal(data_model, debug):
    dm = data_model
    set_compute_setting(dm.gpus)

    start_ts_server(ts_model_store=dm.ts_model_store,
                        ts_log_file=dm.ts_log_file, 
                        ts_log_config=dm.ts_log_config, 
                        ts_config_file=dm.ts_config_file,
                        gpus=dm.gpus,
                        debug=debug)
    ts_health_check()
    register_model(dm.model_name, dm.mar_filepath, dm.gpus)

    if dm.input_path:
        inputs = get_inputs_from_folder(dm.input_path)
        inference_model = {
            "name": dm.model_name,
            "inputs": inputs,
        }

        models_to_validate = [
            inference_model
        ]

        if inputs:
            validate_inference_model(models_to_validate, debug)


def get_inference_with_mar(data_model, debug=False):
    try:
        prepare_settings(data_model)

        # copy mar file to model_store
        mar_dest = os.path.join(data_model.ts_model_store, data_model.mar_filepath.split('/')[-1])
        mar_name = data_model.mar_filepath.split('/')[-1]
        print(mar_dest)
        print(mar_name)
        if data_model.mar_filepath != mar_dest:
            subprocess.check_output(f'cp {data_model.mar_filepath} {mar_dest}', shell=True)

        get_inference_internal(data_model, debug=debug)

    except Exception as e:
        error_msg_print()
        print(e)
        debug and traceback.print_exc()
        sys.exit(1)
