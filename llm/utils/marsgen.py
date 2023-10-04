import json
import os
import sys
import subprocess
from system_utils import check_if_path_exists
from huggingface_hub import HfApi
from collections import Counter

def compare_lists(list1, list2):
    return Counter(list1)==Counter(list2)

def filter_files_by_extension(filenames, extensions_to_remove):
    filtered_filenames = [filename for filename in filenames if not any(filename.endswith(ext) for ext in extensions_to_remove)]
    return filtered_filenames

def generate_mars(dl_model, mar_config, model_store_dir, debug=False):
    debug and print(f"## Starting generate_mars, mar_config:{mar_config}, model_store_dir:{model_store_dir}\n")
    cwd = os.getcwd()
    os.chdir(os.path.dirname(mar_config))

    with open(mar_config) as f:
        models = json.loads(f.read())
        if dl_model.model_name not in models:
            print("## Please check your model name, it should be one of the following : ")
            print(list(models.keys()))
            sys.exit(1)

        model = models[dl_model.model_name]

        handler = None
        if model.get("handler") and model["handler"]:
            handler = dl_model.handler_path
            check_if_path_exists(handler)

        extra_files_list = os.listdir(dl_model.model_path)
        hf_api = HfApi()
        repo_files = hf_api.list_repo_files(repo_id=dl_model.repo_id, token=dl_model.hf_token)
        repo_files = filter_files_by_extension(repo_files, [".safetensors", ".safetensors.index.json"])
        if not compare_lists(extra_files_list, repo_files): #checking if local model files are equal to the repository files
            print("## Model files do not match HuggingFace repository files")
            sys.exit(1)
        extra_files=""
        for file in extra_files_list:
            abs_file_path = os.path.join(dl_model.model_path, file)
            extra_files = extra_files + abs_file_path + ','
        extra_files = extra_files[:len(extra_files)-1]

        runtime = None
        if model.get("runtime") and model["runtime"]:
            runtime = model["runtime"]

        archive_format = None
        if model.get("archive_format") and model["archive_format"]:
            archive_format = model["archive_format"]

        requirements_file = None
        if model.get("requirements_file") and model["requirements_file"]:
            requirements_file = model["requirements_file"]

        export_path = model_store_dir
        if model.get("export_path") and model["export_path"]:
            export_path = model["export_path"]
            check_if_path_exists(export_path)

        model_file_input = None
        if model.get("model_file") and model["model_file"]:
            model_file_input = model["model_file"]

        cmd = model_archiver_command_builder(model["model_name"],
                                             model["version"],
                                             model_file_input,
                                             handler, extra_files,
                                             runtime, archive_format, 
                                             requirements_file,
                                             export_path,
                                             debug=debug)

        debug and print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")

        try:
            subprocess.check_call(cmd, shell=True)
            marfile = "{}.mar".format(model["model_name"])
            print(f"## {marfile} is generated.\n")
        except subprocess.CalledProcessError as exc:
            print("## Creation failed !\n")
            debug and print("## {} creation failed !, error: {}\n".format(model["model_name"], exc))
            sys.exit(1)

    os.chdir(cwd)


def model_archiver_command_builder(model_name=None, version=None, model_file=None, handler=None, extra_files=None,
                                   runtime=None, archive_format=None, requirements_file=None,
                                   export_path=None, force=True, debug=False):
    cmd = "torch-model-archiver"
    if model_name:
        cmd += f" --model-name {model_name}"
    if version:
        cmd += f" --version {version}"
    if model_file:
        cmd += f" --model-file {model_file}"
    if handler:
        cmd += f" --handler {handler}"
    if extra_files:
        cmd += " --extra-files \"{0}\"".format(extra_files)
    if runtime:
        cmd += f" --runtime {runtime}"
    if archive_format:
        cmd += f" --archive-format {archive_format}"
    if requirements_file:
        cmd += f" --requirements-file {requirements_file}"
    if export_path:
        cmd += f" --export-path {export_path}"
    if force:
        cmd += " --force"
    print(f"\n## Generating mar file, will take few mins.\n")
    debug and print(cmd)
    return cmd