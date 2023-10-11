import os
import argparse
import json
import sys
from huggingface_hub import snapshot_download, HfApi
import utils.inference_utils
import utils.marsgen as mg
from utils.system_utils import check_if_path_exists, check_if_folder_empty, create_folder_if_not_exists
from utils.shell_utils import mv_file, rm_dir
from collections import Counter
import re

FILE_EXTENSIONS_TO_IGNORE = [".safetensors", ".safetensors.index.json"]

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'model_config.json')

MIN_REPO_VERSION_LEN = MAR_NAME_LEN = 7
# MIN_REPO_VERSION_LEN - Minimum number of characters allowed in 'repo_version' argument
# MAR_NAME_LEN - Number of characters to include from repo_version in MAR name

class DownloadDataModel(object):
    model_name = str()
    model_path = str()
    download_model = bool()
    mar_output = str()
    mar_name = str()
    repo_id = str()
    repo_version = str()
    handler_path = str()
    hf_token = str()
    debug = bool()


def set_values(args):
    dl_model = DownloadDataModel()
    dl_model.model_name = args.model_name
    dl_model.repo_version = args.repo_version
    dl_model.model_path = args.model_path
    dl_model.download_model = args.no_download
    dl_model.mar_output = args.mar_output
    dl_model.handler_path = args.handler_path
    dl_model.hf_token = args.hf_token
    dl_model.debug = args.debug
    read_config_for_download(dl_model)
    # Sets MAR file name name as <model_name>_<first_6_chars_of_repo_version>
    dl_model.mar_name = f"{dl_model.model_name}_{dl_model.repo_version[0:MAR_NAME_LEN]}"
    return dl_model


def get_ignore_pattern_list(extension_list):
    return ["*" + pattern for pattern in extension_list]


def compare_lists(list1, list2):
    return Counter(list1) == Counter(list2)


def filter_files_by_extension(filenames, extensions_to_remove):
    pattern = '|'.join([re.escape(suffix) + '$' for suffix in extensions_to_remove]) 
    # for the extensions in FILE_EXTENSIONS_TO_IGNORE pattern will be '\.safetensors$|\.safetensors\.index\.json$'
    filtered_filenames = [filename for filename in filenames if not re.search(pattern, filename)]
    return filtered_filenames


def check_if_mar_exists(dl_model):
    check_path = os.path.join(dl_model.mar_output, f"{dl_model.mar_name}.mar")
    if os.path.exists(check_path):
        print(f"## Skipping MAR file generation as it already exists\nModel name: {dl_model.model_name}\nRepository Version: {dl_model.repo_version}\n")
        sys.exit(1)


def check_if_model_files_exist(dl_model):
    extra_files_list = os.listdir(dl_model.model_path)
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files(repo_id=dl_model.repo_id, revision=dl_model.repo_version, token=dl_model.hf_token)
    repo_files = filter_files_by_extension(repo_files, FILE_EXTENSIONS_TO_IGNORE)
    return compare_lists(extra_files_list, repo_files)


def create_tmp_model_store(mar_output, mar_name):
    dir_name = f"tmp_{mar_name}"
    tmp_dir = os.path.join(mar_output, dir_name)
    rm_dir(tmp_dir) # delete existing tmp if it exists
    create_folder_if_not_exists(tmp_dir)
    return tmp_dir


def move_mar(dl_model, tmp_dir):
    old_filename = f"{dl_model.model_name}.mar"
    # using first six characters of repo_version in the MAR filename
    new_filename = f"{dl_model.mar_name}.mar"
    src = os.path.join(tmp_dir, old_filename)
    dst = os.path.join(dl_model.mar_output, new_filename)
    check_if_path_exists(src, "Generated mar file is missing")
    mv_file(src, dst)


def read_config_for_download(dl_model):
    # Read and validate the repo_id and repo_version
    check_if_path_exists(MODEL_CONFIG_PATH)
    with open(MODEL_CONFIG_PATH) as f:
        models = json.loads(f.read())
        if dl_model.model_name in models:
            try:
                dl_model.repo_id = models[dl_model.model_name]['repo_id']
                if not dl_model.repo_version:
                    dl_model.repo_version = models[dl_model.model_name]['repo_version']
                
                if len(dl_model.repo_version) < MIN_REPO_VERSION_LEN:
                    print(f"## Error: 'repo_version' should be at least {MIN_REPO_VERSION_LEN} characters long")
                    sys.exit(1)

                # Make sure there is HF hub token for LLAMA(2)
                if dl_model.repo_id.startswith("meta-llama") and dl_model.hf_token is None:
                    print(f"## Error: HuggingFace Hub token is required for llama download. Please specify it using --hf_token=<your token>. Refer https://huggingface.co/docs/hub/security-tokens")
                    sys.exit(1)

                # Validate downloaded files
                hf_api = HfApi()
                hf_api.list_repo_commits(repo_id=dl_model.repo_id, revision=dl_model.repo_version, token=dl_model.hf_token)

                # Read handler file name
                if not dl_model.handler_path:
                    dl_model.handler_path = os.path.join(os.path.dirname(__file__),
                                                     models[dl_model.model_name]["handler"])
            except Exception:
                print(f"## Error: Please check either repo_id, repo_version or HuggingFace ID is not correct\n")
                sys.exit(1)
        else:
            print("## Please check your model name, it should be one of the following : ")
            print(list(models.keys()))
            sys.exit(1)


def run_download(dl_model):
    if not check_if_folder_empty(dl_model.model_path):
        print("## Make sure the path provided to download model files is empty\n")
        sys.exit(1)

    print(f"\n## Starting model files download from {dl_model.repo_id} with version {dl_model.repo_version}\n")
    snapshot_download(repo_id=dl_model.repo_id,
                      revision = dl_model.repo_version,
                      local_dir=dl_model.model_path,
                      local_dir_use_symlinks=False,
                      token=dl_model.hf_token,
                      ignore_patterns=get_ignore_pattern_list(FILE_EXTENSIONS_TO_IGNORE))
    print("## Successfully downloaded model_files\n")
    return dl_model


def create_mar(dl_model):
    if not check_if_model_files_exist(dl_model):
        print("## Model files do not match HuggingFace repository files")
        sys.exit(1)

    # Creates a temporary directory with the name "tmp_<model-name>_<repo-version>" inside model_store
    tmp_dir = create_tmp_model_store(dl_model.mar_output, dl_model.mar_name)

    mg.generate_mars(dl_model=dl_model, 
                     mar_config=MODEL_CONFIG_PATH,
                     model_store_dir=tmp_dir,
                     debug=dl_model.debug)

    # Move MAR file to model_store
    move_mar(dl_model, tmp_dir)
    # Delete temporary folder
    rm_dir(tmp_dir)
    print(f"\n## Mar file for {dl_model.model_name} with version {dl_model.repo_version} is generated!\n")


def run_script(args):
    dl_model = set_values(args)
    check_if_path_exists(dl_model.model_path, "model_path")
    check_if_path_exists(dl_model.mar_output, "mar_output")
    check_if_mar_exists(dl_model)

    if dl_model.download_model:
        dl_model = run_download(dl_model)
    create_mar(dl_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download script')
    parser.add_argument('--model_name', type=str, default="", required=True,
                        metavar='mn', help='Name of model')
    parser.add_argument('--repo_version', type=str, default="",
                        metavar='rv', help='Commit ID of models repo from HuggingFace repository')
    parser.add_argument('--no_download', action='store_false',
                        help='Set flag to skip downloading the model files')
    parser.add_argument('--model_path', type=str, default="", required=True,
                        metavar='mp', help='Absolute path of model files (should be empty if downloading)')
    parser.add_argument('--mar_output', type=str, default="", required=True,
                        metavar='mx', help='Absolute path of export of MAR file (.mar)')
    parser.add_argument('--handler_path', type=str, default="",
                        metavar='hp', help='absolute path of handler')
    parser.add_argument('--hf_token', type=str, default=None,
                        metavar='hft', help='HuggingFace Hub token to download LLAMA(2) models')
    parser.add_argument('--debug', action='store_true',
                        help='flag to debug')
    args = parser.parse_args()
    run_script(args)