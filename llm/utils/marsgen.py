import os
import sys
import subprocess

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from system_utils import check_if_path_exists

# MAR_NAME_LEN - Number of characters to include from repo_version in MAR name
MAR_NAME_LEN = 7

def get_mar_name(model_name, repo_version):
    mar_name = f"{model_name}_{repo_version[0:MAR_NAME_LEN]}"
    return mar_name


def generate_mars(dl_model, mar_config, model_store_dir, debug=False):
    debug and print(f"## Starting generate_mars, mar_config:{mar_config}, model_store_dir:{model_store_dir}\n")
    cwd = os.getcwd()
    os.chdir(os.path.dirname(mar_config))

    handler = dl_model.handler_path
    check_if_path_exists(handler)

    # Reading all files in model_path to make extra_files string
    extra_files_list = os.listdir(dl_model.model_path)
    extra_files_list = [os.path.join(dl_model.model_path, file) for file in extra_files_list]
    extra_files = ','.join(extra_files_list)

    export_path = model_store_dir
    check_if_path_exists(export_path)

    cmd = model_archiver_command_builder(model_name = dl_model.model_name,
                                            version=dl_model.repo_version,
                                            handler=handler, extra_files=extra_files,
                                            export_path=export_path,
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


def model_archiver_command_builder(model_name=None, version=None, model_file=None,
                                   handler=None, extra_files=None,
                                   runtime=None, archive_format=None,
                                   requirements_file=None, export_path=None,
                                   force=True, debug=False):
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
        cmd += f" --extra-files \"{extra_files}\""
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
    print("\n## Generating mar file, will take few mins.\n")
    debug and print(cmd)
    return cmd
