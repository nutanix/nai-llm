import os
import argparse
import json
import sys
from huggingface_hub import snapshot_download
import utils.inference_utils
import utils.marsgen as mg
from utils.system_utils import check_if_path_exists, create_folder_if_not_exits

MODEL_STORE = 'model-store'

class DownloadDataModel(object):
    model_name = str()
    model_path = str()
    download_model = bool()
    gen_mar = bool()
    for_k8s = bool()
    mar_output = str()
    repo_id = str()
    handler_path = str()
    hf_token = str()
    debug = bool()


def set_values(args):
    dl_model = DownloadDataModel()
    dl_model.model_name = args.model_name
    dl_model.model_path = args.model_path
    dl_model.download_model = args.no_download
    dl_model.gen_mar = args.no_generate
    dl_model.for_k8s = args.k8s
    dl_model.mar_output = args.mar_output
    dl_model.handler_path = args.handler_path
    dl_model.hf_token = args.hf_token
    dl_model.debug = args.debug
    return dl_model


def run_download(dl_model):
    check_if_path_exists(dl_model.model_path, "model_path")
    mar_config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    check_if_path_exists(mar_config_path)

    with open(mar_config_path) as f:
        models = json.loads(f.read())
        if dl_model.model_name in models:
            dl_model.repo_id = models[dl_model.model_name]['repo_id']
        else:
            print("## Please check your model name, it should be one of the following : ")
            print(list(models.keys()))
            sys.exit(1)
    
    if dl_model.repo_id.startswith("meta-llama") and dl_model.hf_token is None: # Make sure there is HF hub token for LLAMA(2)
        print(f"HuggingFace Hub token is required for llama download. Please specify it using --hf_token=<your token>. Refer https://huggingface.co/docs/hub/security-tokens")
        sys.exit(1)
    
    print("## Starting model files download\n")
    snapshot_download(repo_id=dl_model.repo_id,
                      local_dir=dl_model.model_path,
                      local_dir_use_symlinks=False,
                      token=dl_model.hf_token)
    print("## Successfully downloaded model_files\n")
    return dl_model


def create_mar(dl_model):
    check_if_path_exists(dl_model.model_path, "model_path")
    check_if_path_exists(dl_model.mar_output, "mar_output")
    mar_config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    check_if_path_exists(mar_config_path)
    if dl_model.handler_path == "":
        with open(mar_config_path) as f:
            models = json.loads(f.read())
            if dl_model.model_name in models:
                dl_model.handler_path = os.path.join(os.path.dirname(__file__), models[dl_model.model_name]["handler"])
    mg.generate_mars(dl_model=dl_model, 
                     mar_config=mar_config_path,
                     model_store_dir=dl_model.mar_output,
                     debug=dl_model.debug)


def run_script(args):
    dl_model = set_values(args)
    dl_model.download_model and check_if_path_exists(dl_model.model_path, "model_path")
    dl_model.gen_mar and check_if_path_exists(dl_model.mar_output, "mar_output")

    if dl_model.download_model:
        dl_model = run_download(dl_model)
    
    if dl_model.gen_mar:
        if dl_model.for_k8s:
            path = os.path.join(dl_model.mar_output, dl_model.model_name, MODEL_STORE)
            create_folder_if_not_exits(path)
            dl_model.mar_output = path

        create_mar(dl_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download script')
    parser.add_argument('--model_name', type=str, default="", required=True,
                        metavar='mn', help='name of the model')
    parser.add_argument('--no_download', action='store_false',
                        help='flag to not download')
    parser.add_argument('--model_path', type=str, default="",
                        metavar='mp', help='absolute path to model folder')
    parser.add_argument('--no_generate', action='store_false',
                        help='flag to not generating mar')
    parser.add_argument('--k8s', action='store_true',
                        help='flag to handle request for k8s')
    parser.add_argument('--mar_output', type=str, default="",
                        metavar='mx', help='absolute path of output mar')
    parser.add_argument('--handler_path', type=str, default="",
                        metavar='hp', help='absolute path of handler')
    parser.add_argument('--hf_token', type=str, default=None,
                        metavar='hft', help='HuggingFace Hub token to download LLAMA(2) models')
    parser.add_argument('--debug', action='store_true',
                        help='flag to debug')
    args = parser.parse_args()
    run_script(args)