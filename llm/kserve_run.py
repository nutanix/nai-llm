import argparse
import sys
import os
import time
import utils.tsutils as ts
from utils.system_utils import check_if_path_exists
from utils.inference_utils import get_inputs_from_folder
from kubernetes import client, config
from kserve import KServeClient, constants, V1beta1PredictorSpec, V1beta1TorchServeSpec, V1beta1InferenceServiceSpec, V1beta1InferenceService

CONFIG_DIR = 'config'
CONFIG_FILE = 'config.properties'
MODEL_STORE_DIR = 'model-store'

kubMemUnits = ['Ei', 'Pi', 'Ti', 'Gi', 'Mi', 'Ki']

def set_config(model_name, mount_path):
    model_spec_path = os.path.join(mount_path, model_name)
    config_file_path = os.path.join(model_spec_path, CONFIG_DIR, CONFIG_FILE)
    check_if_path_exists(config_file_path, 'Config')
    check_if_path_exists(os.path.join(model_spec_path, MODEL_STORE_DIR), 'Model store')

    config_info = ['\ninstall_py_dep_per_model=true\n', 'model_store=/mnt/models/model-store\n','model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"'+model_name+'":{"1.0":{"defaultVersion":true,"marName":"'+model_name+'.mar","minWorkers":1,"maxWorkers":1,"batchSize":1,"maxBatchDelay":500,"responseTimeout":60}}}}']

    with open(config_file_path, "a") as config_file:
        config_file.writelines(config_info)

def create_pv(core_api, deploy_name, storage, nfs_server, nfs_path):
    # Create Persistent Volume
    persistent_volume = client.V1PersistentVolume(
        api_version='v1',
        kind='PersistentVolume',
        metadata=client.V1ObjectMeta(
            name=deploy_name, 
            labels={
                "storage": "nfs"
            }
        ),
        spec=client.V1PersistentVolumeSpec(
            capacity={
                "storage": storage
            },
            access_modes=["ReadWriteMany"],
            persistent_volume_reclaim_policy='Retain',
            nfs=client.V1NFSVolumeSource(
                path=nfs_path,
                server=nfs_server
            )
        )
    )

    core_api.create_persistent_volume(body=persistent_volume)


def create_pvc(core_api, deploy_name, storage):
    # Create Persistent Volume Claim
    persistent_volume_claim = client.V1PersistentVolumeClaim(
        api_version='v1',
        kind='PersistentVolumeClaim',
        metadata=client.V1ObjectMeta(
            name=deploy_name
        ),
        spec=client.V1PersistentVolumeClaimSpec(
            storage_class_name="",
            access_modes=["ReadWriteMany"],
            resources=client.V1ResourceRequirements(
                requests={
                    "storage": storage
                }
            ),
            selector=client.V1LabelSelector(
                match_labels={
                    "storage": "nfs"
                }
            )
        )
    )

    core_api.create_namespaced_persistent_volume_claim(body=persistent_volume_claim, namespace='default')


def create_isvc(deploy_name, model_name, cpus, memory, gpus):
    storageuri = 'pvc://'+ deploy_name + '/' + model_name

    default_model_spec = V1beta1InferenceServiceSpec(
        predictor=V1beta1PredictorSpec(
            pytorch=V1beta1TorchServeSpec(
                protocol_version='v2',
                storage_uri=storageuri,
                env=[
                    client.V1EnvVar(
                        name='TS_INFERENCE_ADDRESS',
                        value='http://0.0.0.0:8085'
                    ),
                    client.V1EnvVar(
                        name='TS_MANAGEMENT_ADDRESS',
                        value='http://0.0.0.0:8090'
                    ),
                    client.V1EnvVar(
                        name='TS_METRICS_ADDRESS',
                        value='http://0.0.0.0:8091'
                    ),
                    client.V1EnvVar(
                        name='TS_SERVICE_ENVELOPE',
                        value='body'
                    ),
                    client.V1EnvVar(
                        name='TS_NUMBER_OF_GPU',
                        value=str(gpus)
                    )
                ],
                resources=client.V1ResourceRequirements(
                    limits={
                        "cpu": cpus,
                        "memory": memory,
                        "nvidia.com/gpu": gpus
                    },
                    requests={
                        "cpu": cpus,
                        "memory": memory,
                        "nvidia.com/gpu": gpus
                    }
                )
            )
        )
    )

    isvc = V1beta1InferenceService(api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(name=deploy_name, namespace='default'),
        spec=default_model_spec)


    kserve = KServeClient(client_configuration=config.load_kube_config())
    kserve.create(isvc, watch=True)

def execute_inference_on_inputs(model_inputs, model_name, deploy_name):
    for input in model_inputs:    
        host = os.environ.get('INGRESS_HOST')
        port = os.environ.get('INGRESS_PORT')

        kserve = KServeClient(client_configuration=config.load_kube_config())
        obj=kserve.get(name=deploy_name, namespace='default')
        service_hostname = obj['status']['url'].split('/')[2:][0]
        headers = {"Content-Type": "application/json; charset=utf-8", "Host": service_hostname}

        response = ts.run_inference_v2(model_name, input, protocol="http", host=host, port=port, headers=headers)
        if response and response.status_code == 200:
            print(f"## Successfully ran inference on {model_name} model. \n\n Output - {response.text}\n\n")
        else:
            print(f"## Failed to run inference on {model_name} - model \n")
            sys.exit(1)

def execute(args):
    if not any(unit in args.mem for unit in kubMemUnits):
        print("container memory unit has to be one of", kubMemUnits)
        sys.exit(1)

    gpus = args.gpu
    cpus = args.cpu
    memory = args.mem
    nfs_server, nfs_path = args.nfs.split(':')
    deploy_name = args.deploy_name
    model_name = args.model_name
    input_path = args.data

    if not nfs_path or not nfs_server:
        print("NFS server and share path was not provided in accepted format - <address>:<share_path>")
        sys.exit(1)

    storage = '100Gi'

    set_config(model_name, args.mount_path)

    config.load_kube_config()
    core_api = client.CoreV1Api()

    create_pv(core_api, deploy_name, storage, nfs_server, nfs_path)

    create_pvc(core_api, deploy_name, storage)

    create_isvc(deploy_name, model_name, cpus, memory, gpus)

    print("wait for model registration to complete, will take some time")
    time.sleep(240)

    if input_path:
        check_if_path_exists(input_path, 'Input')
        model_inputs = get_inputs_from_folder(input_path)
        execute_inference_on_inputs(model_inputs, model_name, deploy_name)


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to generate the yaml.')

    # Add arguments
    parser.add_argument('--nfs', type=str, help='nfs ip address with mount path')
    parser.add_argument('--gpu', type=int, help='number of gpus')
    parser.add_argument('--cpu', type=int, help='number of cpus')
    parser.add_argument('--mem', type=str, help='memory required by the container')
    parser.add_argument('--model_name', type=str, help='name of the model to deploy')
    parser.add_argument('--mount_path', type=str, help='local path to the nfs mount location')
    parser.add_argument('--deploy_name', type=str, help='name of the deployment')
    parser.add_argument('--data', type=str, help='data folder for the deployment validation')

    # Parse the command-line arguments
    args = parser.parse_args()
    execute(args)