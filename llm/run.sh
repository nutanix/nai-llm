#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

helpFunction()
{
   echo ""
   echo "Usage: $0 -n <MODEL_NAME> -a <MAR_EXPORT_PATH> [OPTIONAL -d <INPUT_PATH> -v <REPO_VERSION>]"
   echo -e "\t-n Name of the Model"
   echo -e "\t-v HuggingFace repository version (optional)"
   echo -e "\t-d Absolute path of input data folder (optional)"
   echo -e "\t-a Absolute path to the Model Store directory"
   echo -e "\t-q BitsAndBytes Quantization Precision (4 or 8) (optional)"
   exit 1 # Exit script after printing help
}

while getopts ":n:v:d:q:a:" opt;
do
   case "$opt" in
        n ) model_name="$OPTARG" ;;
        v ) repo_version="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        a ) model_store="$OPTARG" ;;
        q ) quantize_bits="$OPTARG" ;;
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

    if [ ! -z "$data" ] ; then
        cmd+=" --data $data"
    fi

    if [ ! -z "$quantize_bits" ] ; then
        cmd+=" --quantize_bits $quantize_bits"
    fi
}

function inference_exec_vm(){
    echo "";
    echo "Running the Inference script";
    echo "";
    $cmd
    exit_status=$?
    # Checks exit status of python file
    if [ "${exit_status}" -ne 0 ];
    then
        exit 1
    else
        exit 0
    fi
}

# Entry Point
create_execution_cmd

inference_exec_vm