import json
import os
import sys
import subprocess
from system_utils import check_if_path_exists


def generate_mars(dl_model, mar_config, model_store_dir, debug=False):
    debug and print(f"## Starting generate_mars, mar_config:{mar_config}, model_store_dir:{model_store_dir}\n")
    cwd = os.getcwd()
    os.chdir(os.path.dirname(mar_config))

    handler = dl_model.handler_path
    check_if_path_exists(handler)

    extra_files_list = os.listdir(dl_model.model_path)
    extra_files_list = [os.path.join(dl_model.model_path, file) for file in extra_files_list]
    extra_files = ','.join(extra_files_list)

    export_path = model_store_dir
    check_if_path_exists(export_path)

    runtime = None
    archive_format = None
    requirements_file = None
    model_file_input = None

    cmd = model_archiver_command_builder(dl_model.model_name,
                                            dl_model.repo_version,
                                            model_file_input,
                                            handler, extra_files,
                                            runtime, archive_format, 
                                            requirements_file,
                                            export_path,
                                            debug=debug)

    debug and print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")

    try:
        subprocess.check_call(cmd, shell=True)
        debug and print(f"## Model {dl_model.model_name} with version {dl_model.repo_version} is generated.\n")
    except subprocess.CalledProcessError as exc:
        print("## Creation failed !\n")
        debug and print("## {} creation failed !, error: {}\n".format(dl_model.model_name, exc))
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