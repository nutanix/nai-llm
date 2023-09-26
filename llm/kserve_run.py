import argparse
import sys
import os
from utils.system_utils import check_if_path_exists
from kubernetes import client, config
from kserve import KServeClient, constants, V1beta1PredictorSpec, V1beta1TorchServeSpec, V1beta1InferenceServiceSpec, V1beta1InferenceService

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


CONFIG_DIR = 'config'
CONFIG_FILE = 'config.properties'
MODEL_STORE_DIR = 'model-store'

kubMemUnits = ['Ei', 'Pi', 'Ti', 'Gi', 'Mi', 'Ki']

# Parse the command-line arguments
args = parser.parse_args()

if not any(unit in args.mem for unit in kubMemUnits):
    print("container memory unit has to be one of", kubMemUnits)
    sys.exit(1)

gpus = args.gpu
cpus = args.cpu
memory = args.mem
nfs_server, nfs_path = args.nfs.split(':')
deploy_name = args.deploy_name

if not nfs_path or not nfs_server:
    print("NFS server and share path was not provided in accepted format - <address>:<share_path>")
    sys.exit(1)

storage = '100Gi'

model_spec_path = os.path.join(args.mount_path, args.model_name)
config_file_path = os.path.join(model_spec_path, CONFIG_DIR, CONFIG_FILE)
check_if_path_exists(config_file_path, 'Config')
check_if_path_exists(os.path.join(model_spec_path, MODEL_STORE_DIR), 'Model store')

config_info = ['\ninstall_py_dep_per_model=true\n', 'model_store=/mnt/models/model-store\n','model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"'+args.model_name+'":{"1.0":{"defaultVersion":true,"marName":"'+args.model_name+'.mar","minWorkers":1,"maxWorkers":1,"batchSize":1,"maxBatchDelay":500,"responseTimeout":60}}}}']

with open(config_file_path, "a") as config_file:
    config_file.writelines(config_info)

config.load_kube_config()
core_api = client.CoreV1Api()

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

storageuri = 'pvc://'+ deploy_name + '/' + args.model_name

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
isvc_object = kserve.create(isvc, watch=True)

