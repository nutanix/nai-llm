#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

helpFunction()
{
   echo ""
   echo "Usage: $0 -n <MODEL_NAME> -a <MAR_EXPORT_PATH> -g <NUM_GPUS> [OPTIONAL -d <INPUT_PATH> -v <REPO_VERSION>]"
   echo -e "\t-n Name of the Model"
   echo -e "\t-v HuggingFace repository version (optional)"
   echo -e "\t-d Absolute path of input data folder (optional)"
   echo -e "\t-g Number of gpus to be used to execute (Set 0 to use cpu)"
   echo -e "\t-a Absolute path to the Model Store directory"
   exit 1 # Exit script after printing help
}

while getopts ":n:v:d:g:a:o:r" opt;
do
   case "$opt" in
        n ) model_name="$OPTARG" ;;
        v ) repo_version="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        a ) model_store="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

function create_execution_cmd()
{
    gen_folder="gen"
    
    cmd="python3 $wdir/torchserve_run.py"
    
    cmd+=" --gen_folder_name $gen_folder"

    if [ ! -z $model_name ] ; then
        cmd+=" --model_name $model_name"
     else
        echo "Model Name not provided"
        helpFunction
    fi

    if [ ! -z $repo_version ] ; then
        cmd+=" --repo_version $repo_version"
    fi

    if [ ! -z $model_store ] ; then
        cmd+=" --model_store $model_store"
    else
        echo "Model store path not provided"
        helpFunction
    fi

    if [ -z "$gpus" ] ; then
        echo "GPU parameter missing"
        helpFunction
    elif [ "$gpus" -gt 0 ]; then

        # Check if nvidia-smi is available
        if ! command -v nvidia-smi &> /dev/null; then
            echo "NVIDIA GPU drivers or nvidia-smi are not installed."
            helpFunction
        fi
        # Run nvidia-smi to get GPU information and extract the GPU name
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
        gpu_name=$(echo "$gpu_info" | tr -d '[:space:]')
        sys_gpus=$(nvidia-smi --list-gpus | wc -l)
        if [ "$gpus" -gt "$sys_gpus" ]; then
            echo "Machine has fewer GPUs ($sys_gpus) then input provided - $gpus";
            helpFunction  
        fi

        cmd+=" --gpu_type $gpu_name"
    fi
    cmd+=" --gpus $gpus"

    if [ ! -z "$data" ] ; then
        cmd+=" --data $data"
    fi
}

function inference_exec_vm(){
    echo "Running the Inference script";
    echo "";
    echo "$cmd";
    echo "";
    $cmd
}

# Entry Point
create_execution_cmd

inference_exec_vm