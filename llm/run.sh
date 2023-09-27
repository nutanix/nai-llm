#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

function helpFunction()
{
    if [ $compute_choice = "vm" ] ;
    then
        vmHelpFunction
    else
        kubernetesHelpFunction
    fi
    echo -e "\t-o Choice of compute infra to be run on"
    echo -e "\t-n Name of the Model"
    echo -e "\t-d Absolute path to the inputs folder that contains data to be predicted."
    echo -e "\t-g Number of gpus to be used to execute. Set 0 to use cpu"
    echo -e "\t-k Keep the torchserve server alive after run completion. Default stops the server if not set"
    
    exit 1 # Exit script after printing help
}

function vmHelpFunction()
{
    echo "Usage: $0 -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH>  -g <NUM_OF_GPUS> -a <ABOSULTE_PATH_MODEL_ARCHIVE_FILE> [OPTIONAL -k]"
    echo -e "\t-a Absolute path to the model archive file (.mar)"
}

function kubernetesHelpFunction()
{
    echo "Usage: $0 -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH>  -g <NUM_OF_GPUS> -m <NFS_LOCAL_MOUNT_LOCATION> -f <NFS_ADDRESS_WITH_SHARE_PATH> -e <KUBE_DEPLOYMENT_NAME> [OPTIONAL -k]"
    echo -e "\t-m Absolute path to the NFS local mount location"
    echo -e "\t-f NFS server address with share path information"
    echo -e "\t-e Name of the deployment metadata"
}

function create_execution_cmd()
{
    echo $model_name
    gen_folder="gen"
    cmd="python3 $wdir/torchserve_run.py"
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

function inference_exec_kubernetes()
{   
    echo "Config $KUBECONFIG"

    if [ -z "$KUBECONFIG" ]; then
        echo "Kube config environment variable is not set - KUBECONFIG"
        exit 1 
    fi

    if [ -z "$gpus"  ] || [ "$gpus" -eq 0 ] 
    then
        gpus="0"
    fi

    if [ -z $mount_path ] ; then
        echo "Local mount path not provided"
        helpFunction
    fi

    if [ -z $nfs ] ; then
        echo "NFS info not provided"
        helpFunction
    fi

    if [ -z $deploy_name ] ; then
        echo "deployment metadata name not provided"
        helpFunction
    fi

    mkdir $mount_path/$model_name/config
    cp $wdir/config.properties $mount_path/$model_name/config/

    export INGRESS_HOST=$(kubectl get po -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].status.hostIP}')
    export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')

    echo "Running the Inference script";
    python3 $wdir/kserve_run.py --gpu $gpus --cpu 8 --mem 32Gi --model_name $model_name --mount_path $mount_path --nfs $nfs --deploy_name $deploy_name --data $data

    if [ -z $stop_server ] ; then
        python3 $wdir/utils/cleanup.py --k8s --deploy_name $deploy_name
    fi
}

# Entry Point
while getopts ":n:d:g:o:a:m:f:e:k" opt;
do
   case "$opt" in
        n ) model_name="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        o ) compute_choice="$OPTARG" ;;
        k ) stop_server=0 ;;
        a ) mar_file_path="$OPTARG" ;;
        m ) mount_path="$OPTARG" ;;
        f ) nfs="$OPTARG" ;;
        e ) deploy_name="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

shift $((OPTIND-3))
OPTIND=2

if [ -z "$compute_choice"  ] 
then
    compute_choice="vm"
fi


case $compute_choice in
    "k8s")
        echo "Compute choice is Kubernetes."

        inference_exec_kubernetes
        ;;
    "vm")
        echo "Compute choice is VM."

        create_execution_cmd
        inference_exec_vm
        ;;
    *)
        echo "Invalid choice. Exiting."
        echo "Please select a valid option:"
        echo "k8s for Kubernetes env"
        echo "vm for virtual machine env"
        #echo "3. docker for docker env"
        exit 1
        ;;
esac

