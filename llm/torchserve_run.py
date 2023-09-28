import os
import argparse
import sys
import json
from utils.inference_utils import get_inference_with_mar, error_msg_print
from utils.shell_utils import rm_dir
from utils import tsutils as ts
from utils.system_utils import check_if_path_exists
from utils.system_utils import create_folder_if_not_exits, check_file_extension, remove_suffix_if_starts_with
import utils.inference_data_model as dm



def run_inference_with_mar(args):
    check_if_path_exists(args.mar)
    
    data_model = dm.set_data_model(data=args.data, gpus=args.gpus,
                                   gen_folder=args.gen_folder_name,
                                   model_name=args.model_name,
                                   mar_filepath=args.mar)
    get_inference_with_mar(data_model, args.debug_mode)


def run_inference(args, model_config_path):
    # validate gen folder
    create_folder_if_not_exits(os.path.join(os.path.dirname(__file__),
                               'utils', args.gen_folder_name))

    if args.mar:
        if check_file_extension(args.mar, ".mar"):
            print("The model archive file has the correct extension.")
            if args.gpus > 0:
                check_if_path_exists(model_config_path)
                with open(model_config_path) as f:
                    gpu_config = json.loads(f.read())
                    gpu_type_list = gpu_config[args.model_name]["gpu_type"]
                    gpu_type = remove_suffix_if_starts_with(args.gpu_type, "NVIDIA")
                    if gpu_type not in gpu_type_list:
                        print("This GPU Type is not supported, the supported GPU Types are:")
                        for gpu in gpu_type_list:
                            print(gpu)
                        error_msg_print()
                        sys.exit(1)
            ts.set_model_params(args.model_name)
            run_inference_with_mar(args)
        else:
            print("The model archive file does not have the correct extension.")
            error_msg_print()
            sys.exit(1)
    else:
        print("Absolute path to the model archive file (.mar) not provided")
        error_msg_print()
        sys.exit(1)


def torchserve_run(args):
    try:
        # Stop the server if anything is running
        cleanup(args.gen_folder_name, True, False)
        # data folder exists check
        if args.data:
            check_if_path_exists(args.data)
        
        model_config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
        model_name=args.model_name
        check_if_path_exists(model_config_path)
        with open(model_config_path) as f:
            models = json.loads(f.read())
            if model_name not in models:
                print("## Please check your model name, it should be one of the following : ")
                print(list(models.keys()))
                error_msg_print()
                sys.exit(1)
        
        run_inference(args, model_config_path)

        print("\n**************************************")
        print("*\n*\n*  Inference Run Successful  ")
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
    parser.add_argument('--gpus', type=int, default=0,
                        metavar='g', help='number of gpus to use for execution')
    parser.add_argument('--gpu_type', type=str, default="",
                        metavar='gn', help='type of gpus to use for execution')
    parser.add_argument('--gen_folder_name', type=str, default="gen",
                        metavar='f', help='Name for generate folder used to create temp files')
    parser.add_argument('--stop_server', type=int, default=1,
                        metavar='stop', help='Stop torchserve after run completion')
    parser.add_argument('--ts_cleanup', type=int, default=1,
                        metavar='cleanup',
                        help='clean up torchserve temp files after run completion')
    parser.add_argument('--debug_mode', type=int, default=0,
                        metavar='debug', help='run debug mode')
    parser.add_argument('--mar', type=str, default="",
                        metavar='mar', help='absolute path to the model archive file')
    args = parser.parse_args()
    torchserve_run(args)
