import os
import argparse
import sys
import json
from utils.inference_utils import get_inference_with_mar, error_msg_print
from utils.shell_utils import rm_dir
from utils import tsutils as ts
from utils.system_utils import check_if_path_exists
from utils.system_utils import create_folder_if_not_exists, remove_suffix_if_starts_with
import utils.inference_data_model as dm
from download import MAR_NAME_LEN

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'model_config.json')

def read_config_for_inference(args):
    with open(MODEL_CONFIG_PATH) as f:
        models = json.loads(f.read())
        if args.model_name not in models:
            print("## Please check your model name, it should be one of the following : ")
            print(list(models.keys()))
            error_msg_print()
            sys.exit(1)

        if args.gpus > 0:
            gpu_type_list = models[args.model_name]["gpu_type"]
            gpu_type = remove_suffix_if_starts_with(args.gpu_type, "NVIDIA")
            if gpu_type not in gpu_type_list:
                print("WARNING: This GPU Type is not validated, the validated GPU Types are:")
                for gpu in gpu_type_list:
                    print(gpu)

        if models[args.model_name]["repo_version"] and not args.repo_version:
            args.repo_version = models[args.model_name]["repo_version"]
    return args


def set_mar_filepath(model_store, model_name, repo_version):
    mar_name = f"{model_name}_{repo_version[0:MAR_NAME_LEN]}.mar"
    return os.path.join(model_store, mar_name)


def run_inference_with_mar(args):
    check_if_path_exists(args.mar)
    data_model = dm.set_data_model(data=args.data, gpus=args.gpus,
                                   gen_folder=args.gen_folder_name,
                                   model_name=args.model_name,
                                   mar_filepath=args.mar,
                                   repo_version=args.repo_version)
    get_inference_with_mar(data_model, args.debug_mode)


def run_inference(args):
    check_if_path_exists(args.model_store, "Model Store")

    args.mar = set_mar_filepath(args.model_store, args.model_name, args.repo_version)
    check_if_path_exists(args.mar, "MAR file")

    create_folder_if_not_exists(os.path.join(os.path.dirname(__file__),
                               'utils', args.gen_folder_name))
    
    ts.set_model_params(args.model_name)
    run_inference_with_mar(args)


def torchserve_run(args):
    try:
        # Stop the server if anything is running
        cleanup(args.gen_folder_name, True, False)

        check_if_path_exists(MODEL_CONFIG_PATH, "Model Config")
        args = read_config_for_inference(args)
        
        run_inference(args)

        print("\n**************************************")
        print("*\n*\n*  Ready For Inferencing  ")
        print("*\n*\n**************************************")

    finally:
        cleanup(args.gen_folder_name, args.stop_server, args.ts_cleanup)


def cleanup(gen_folder, ts_stop = True, ts_cleanup = True):
    if ts_stop:
        ts.stop_torchserve()
        dirpath = os.path.dirname(__file__)
        # clean up the logs folder to reset logs before the next run
        # TODO - To reduce logs from taking a lot of
        # storage it is being cleared everytime it is stopped
        # Understand on how this can be handled better by rolling file approach
        rm_dir(os.path.join(dirpath, 'utils', gen_folder, 'logs'))

        if ts_cleanup:
            # clean up the entire generate folder
            rm_dir(os.path.join(dirpath, 'utils', gen_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference run script')
    parser.add_argument('--data', type=str, default="",
                        metavar='d',
                        help='absolute path to the inputs folder that contains data to be predicted.')
    parser.add_argument('--model_name', type=str, default="",
                        metavar='n', help='name of the model file')
    parser.add_argument('--repo_version', type=str, default="",
                        metavar='n', help='HuggingFace repository version')
    parser.add_argument('--gpus', type=int, default=0,
                        metavar='g', help='number of gpus to use for execution')
    parser.add_argument('--gpu_type', type=str, default="",
                        metavar='gn', help='type of gpus to use for execution')
    parser.add_argument('--gen_folder_name', type=str, default="gen",
                        metavar='f', help='Name for generate folder used to create temp files')
    parser.add_argument('--stop_server', type=int, default=0,
                        metavar='stop', help='Stop torchserve after run completion')
    parser.add_argument('--ts_cleanup', type=int, default=0,
                        metavar='cleanup',
                        help='clean up torchserve temp files after run completion')
    parser.add_argument('--debug_mode', type=int, default=0,
                        metavar='debug', help='run debug mode')
    parser.add_argument('--model_store', type=str, default="",
                        metavar='model_store', help='absolute path to the model store directory')
    args = parser.parse_args()
    torchserve_run(args)
