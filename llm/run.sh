#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

helpFunction()
{
   echo ""
   echo "Usage: $0 -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH>  -g <NUM_OF_GPUS> -a <ABOSULTE_PATH_MODEL_ARCHIVE_FILE> [OPTIONAL -k]"
   echo -e "\t-o Choice of compute infra to be run on"
   echo -e "\t-n Name of the Model"
   echo -e "\t-d Absolute path to the inputs folder that contains data to be predicted."
   echo -e "\t-g Number of gpus to be used to execute. Set 0 to use cpu"
   echo -e "\t-a Absolute path to the model archive file (.mar)"
   echo -e "\t-k Keep the torchserve server alive after run completion. Default stops the server if not set"
   exit 1 # Exit script after printing help
}

while getopts ":n:d:g:a:o:r:k" opt;
do
   case "$opt" in
        o ) compute_choice="$OPTARG" ;;
        n ) model_name="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        a ) mar_file_path="$OPTARG" ;;
        k ) stop_server=0 ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

function create_execution_cmd()
{
    echo $model_name
    gen_folder="gen"
    if [ $compute_choice = "vm" ] ;
    then
        cmd="python3 $wdir/torchserve_run.py"
    else
        cmd="python3 code/torchserve/torchserve_run.py"
    fi
    
    cmd+=" --gen_folder_name $gen_folder"

    if [ ! -z $model_name ] ; then
        cmd+=" --model_name $model_name"
     else
        echo "Model Name not provided"
        helpFunction
    fi

    if [ ! -z $mar_file_path ] ; then
        cmd+=" --mar $mar_file_path"
    else
        echo "Model Archive File path not provided"
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
    
    if [ ! -z $stop_server ] ; then
        cmd+=" --stop_server $stop_server"
    fi

    if [ ! -z "$data" ] ; then
        cmd+=" --data $data"
    fi
}

function inference_exec_vm(){
    echo "Running the Inference script";
    echo "$cmd"
    $cmd
}

# Entry Point
if [ -z "$compute_choice"  ] 
then
    compute_choice="vm"
fi

create_execution_cmd
case $compute_choice in
    "vm")
        echo "Compute choice is VM."
        inference_exec_vm
        ;;
    *)
        echo "Invalid choice. Exiting."
        echo "Please select a valid option:"
        #echo "1. k8s for Kubernetes env"
        echo " vm for virtual machine env"
        #echo "3. docker for docker env"
        exit 1
        ;;
esac

