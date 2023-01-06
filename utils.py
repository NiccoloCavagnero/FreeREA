import platform
import os
import re
import numpy as np
import random
import torch
import torch.nn as nn


def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


def avg_arr(arr):
    sum = 0.
    count = 0.
    for i in range(len(arr)):
        current_count = np.prod([dim for dim in arr[i].shape])
        count += current_count
        sum += torch.sum(arr[i]) * current_count
    return sum.item() / count


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads

    if type(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


# Try considering bn too.
def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array


def get_default_data_root():
    if 'DATA_PATH' in os.environ:
        return os.environ['DATA_PATH']

    node = platform.node()

    sysname = node
    # To recognize clusters
    """
    if re.match(r'compute-\d+-\d+', sysname):
        sysname = 'cluster1'
    elif re.match(r'node\d', sysname):
        sysname = 'cluster2'
    """

    paths = {
        # Map in the form
        # 'hostname': '/path/to/datasets'
        # Where "datasets" directory is expected to contain CIFAR10, CIFAR100 and ImageNet16 directories
    }

    return paths.get(sysname, None)


def get_nas_archive_path(nasbench_name: str):
    nasbench_name = nasbench_name.lower()
    if nasbench_name not in ('nats', 'nasbench101'):
        raise Exception(f"Unknown NAS benchmark {nasbench_name}")

    # Use NATS_PATH or NASBENCH101_PATH env vars to override
    env_var = f'{nasbench_name.upper()}_PATH'
    if env_var in os.environ:
        return os.environ[env_var]

    # Get default path depending on the local system
    node = platform.node()

    sysname = node
    # To recognize clusters
    """
    if re.match(r'compute-\d+-\d+', sysname):
        sysname = 'cluster1'
    elif re.match(r'node\d', sysname):
        sysname = 'cluster2'
    """

    if nasbench_name == 'nats':
        paths = {
            # Map in the form
            # 'hostname': '/path/to/NATS-tss-v1_0-3ffb9-simple'
        }
    elif nasbench_name == 'nasbench101':
        paths = {
            # Map in the form
            # 'hostname': '/path/to/nb101_pkl_file.pkl'
        }

    return paths.get(sysname, None)
