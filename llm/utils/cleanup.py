import os
import argparse
import sys
from shell_utils import rm_dir
import tsutils as ts

from kubernetes import client, config
from kserve import KServeClient


def vm():
    ts.stop_torchserve()
    dirpath = os.path.dirname(__file__)
    # clean up the logs folder to reset logs before the next run
    # TODO - To reduce logs from taking a lot of storage it is being cleared everytime it is stopped
    # Understand on how this can be handled better by rolling file approach
    rm_dir(os.path.join(dirpath, 'gen', 'logs'))
    # clean up the entire generate folder
    rm_dir(os.path.join(dirpath, 'gen'))

def kubernetes(deploy_name):
    print("Clean up triggered for all the deployments under -", deploy_name)
    kube_config = config.load_kube_config()
    kserve = KServeClient(client_configuration=kube_config)
    try:
        kserve.delete(name=deploy_name, namespace='default')
    except:
        return

    core_api = client.CoreV1Api()
    try:
        core_api.delete_persistent_volume(name=deploy_name)
    except:
        return
    
    try:
        core_api.delete_namespaced_persistent_volume_claim(name=deploy_name, namespace='default')
    except:
        return


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to cleanup existing deployment.')

    # Add arguments
    parser.add_argument('--k8s', action='store_true', help='flag to handle request for k8s')
    parser.add_argument('--deploy_name', type=str, help='name of the deployment')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.k8s:
        if not args.deploy_name:
            print("Deployment name not provided")
            sys.exit(1)
        kubernetes(args.deploy_name)

    else:
        vm()