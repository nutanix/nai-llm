import os
import platform
import subprocess
import sys
import json

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

nvidia_smi_cmd = {
    "Windows": "nvidia-smi.exe",
    "Darwin": "nvidia-smi",
    "Linux": "nvidia-smi",
}

def is_gpu_instance():
    try:
        subprocess.check_output(nvidia_smi_cmd[platform.system()])
        print('\n## Nvidia GPU detected!')
        return True
    except Exception:
        print('\n## No Nvidia GPU in system!')
        return False

def is_conda_build_env():
    return True if os.system("conda-build") == 0 else False

def is_conda_env():
    return True if os.system("conda") == 0 else False

def check_python_version():
    req_version = (3, 8)
    cur_version = sys.version_info

    if not (
        cur_version.major == req_version[0] and cur_version.minor >= req_version[1]
    ):
        print("System version" + str(cur_version))
        print(
            f"TorchServe supports Python {req_version[0]}.{req_version[1]} and higher only. Please upgrade"
        )
        exit(1)

def check_ts_version():
    from ts.version import __version__

    return __version__

def try_and_handle(cmd, dry_run=False):
    if dry_run:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise e

def check_if_path_exists(filepath, param = ""):
    if not os.path.exists(filepath):
        print(f"Filepath does not exist {param} - {filepath}")
        sys.exit(1)

def create_folder_if_not_exits(path):
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path}")

def check_file_extension(file_path, extension):
    return file_path.endswith(extension)

def remove_suffix_if_starts_with(string, suffix):
    if string.startswith(suffix):
        return string[len(suffix):]  
    else:
        return string  
    
def set_model_params(model_config_path, model_name):
    # Clear existing environment variables if they exist
    generation_params = {"TS_TEMPERATURE":"temperature", 
                         "TS_REP_PENALTY":"repetition_penalty",
                         "TS_TOP_P":"top_p", 
                         "TS_MAX_TOKENS":"max_new_tokens"}
    for var_name in generation_params.keys():
        if var_name in os.environ:
            del os.environ[var_name]
    
    # Set the new environment variables with the provided values in model_config
    with open(model_config_path) as f:
        model_config = json.loads(f.read())
        if model_name in model_config:
            param_config = model_config[model_name]["model_params"]
            for var_name, var_value in generation_params.items():
                if var_value in param_config:
                   os.environ[var_name] = str(param_config[var_value])
