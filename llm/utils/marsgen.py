import json
import os
import sys
import subprocess
from system_utils import check_if_path_exists

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

        serialized_file_path = None
        if model.get("serialized_file_local") and model["serialized_file_local"]:
            serialized_file_path = os.path.join(dl_model.model_path, model["serialized_file_local"])
            check_if_path_exists(serialized_file_path)

        handler = None
        if model.get("handler") and model["handler"]:
            handler = dl_model.handler_path
            check_if_path_exists(handler)

        extra_files = None
        if model.get("extra_files") and model["extra_files"]:
            # prefix each extra file with model_path
            extra_files_list = model["extra_files"].split(',')
            extra_files=""
            for file in extra_files_list:
                abs_file_path = os.path.join(dl_model.model_path, file)
                check_if_path_exists(abs_file_path)
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
                                             serialized_file_path,
                                             handler, extra_files,
                                             runtime, archive_format, 
                                             requirements_file,
                                             export_path)

        debug and print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")

        try:
            subprocess.check_call(cmd, shell=True)
            marfile = "{}.mar".format(model["model_name"])
            print(f"## {marfile} is generated.\n")
        except subprocess.CalledProcessError as exc:
            print("## {} creation failed !, error: {}\n".format(model["model_name"], exc))
            sys.exit(1)

        if model.get("serialized_file_remote") and \
                model["serialized_file_remote"] and \
                os.path.exists(serialized_file_path):
            os.remove(serialized_file_path)

    os.chdir(cwd)


def model_archiver_command_builder(model_name=None, version=None, model_file=None,
                                   serialized_file=None, handler=None, extra_files=None,
                                   runtime=None, archive_format=None, requirements_file=None,
                                   export_path=None, force=True):
    cmd = "torch-model-archiver"
    if model_name:
        cmd += f" --model-name {model_name}"
    if version:
        cmd += f" --version {version}"
    if model_file:
        cmd += f" --model-file {model_file}"
    if serialized_file:
        cmd += f" --serialized-file {serialized_file}"
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
    print(cmd)
    return cmd