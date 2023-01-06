#!/usr/bin/env python3

import torch
import time
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from datasets import DATASET_BUILDERS
from nas_spaces import NATSBench, NASBench101
from nas_utils import compute_tfm, METRIC_NAME_MAP
from utils import get_nas_archive_path, get_default_data_root, seed_all

# Computable metrics
metric_names = tuple(METRIC_NAME_MAP.keys())

parser = ArgumentParser()
parser.add_argument('--space', type=str, default='nats', choices=('nats', 'nasbench101'))
parser.add_argument('--dataset', choices=('cifar10', 'cifar100', 'ImageNet16-120'), default='cifar10')
parser.add_argument('--metrics', nargs='+', choices=metric_names, default=None)
parser.add_argument('--metrics-root', default='cached_metrics')
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--ignore-errors', action='store_true')
parser.add_argument('--resume', type=int, default=None)
parser.add_argument('--show-correlation', type=int, default=None)

parser.add_argument('--bs', type=int, default=64)
group = parser.add_mutually_exclusive_group()
group.add_argument('--save-batch', type=str, default=None)
group.add_argument('--load-batch', type=str, default=None)
group.add_argument('--rand-batch', action='store_true')
args = parser.parse_args()

metric_names = tuple(args.metrics or metric_names)

archive_path = get_nas_archive_path(args.space)
dataset = args.dataset
out_path = os.path.join(args.metrics_root, args.space, dataset)
os.makedirs(out_path, exist_ok=True)
seed = args.seed

seed_all(seed)

# Load a new batch
if not args.load_batch:
    data, _ = DATASET_BUILDERS[dataset](get_default_data_root())

    loader = DataLoader(data, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True,
                        persistent_workers=True, multiprocessing_context='fork')

    if not args.rand_batch:
        batch = next(iter(loader))

        # Save it if required
        if args.save_batch:
            torch.save(batch, args.save_batch)
            print("Batch saved.")
else:
    # Load a batch (useful to resume... we use seeds, but you are never sure)
    batch = torch.load(args.load_batch)
    assert batch[0].shape[0] == args.bs, "Wrong batch size"
    print("Batch loaded.")

# Select device and move batch to the device
device = torch.device(f'cuda:{args.device}')
if not args.rand_batch:
    batch = batch[0].to(device), batch[1].to(device)

# Load NAS space
if args.space == 'nats':
    # NO metrics for NATS! We compute them here!
    api = NATSBench(archive_path, args.dataset, metric_root=None)
elif args.space == 'nasbench101':
    api = NASBench101(archive_path, verbose=True, progress=True)
    # Only CIFAR10 is supported by NASBench101
    assert args.dataset == 'cifar10', "Wrong dataset"
else:
    raise ValueError("This should never happen.")

# Dict: metric_name -> file_handler
file_handlers = {}

# Dict: metric_name -> list(vals...)
# Used only if --show-correlation is specified
metric_vals = {}
accuracies = []

# For each metric
for m_name in metric_names:
    metric_vals[m_name] = []

    # Check if the file doesn't exist. If it exists and --resume was not specified, exit.
    fpath = os.path.join(out_path, f'{m_name}.csv')
    if os.path.exists(fpath) and args.resume is None:
        print("Metrics already exist and --resume option was not specified. Refusing to continue")
        exit(-1)

    # Create the file and open it in "append" mode
    file_handlers[m_name] = open(fpath, 'a')
    # If we are resuming, don't write the header again!
    if args.resume is None:
        # index: index of the network
        # time=[s]: time to init the network and compute the metric, 3 times
        # time_pure=[s]: time to compute the metric 3 times. Network initialization is excluded
        file_handlers[m_name].write(f'index,{m_name},time=[s],time_pure=[s]\n')

# Same thing for the accuracy file. It's not a metric, but a CSV file comes handy
accuracy_handler = open(os.path.join(out_path, f'accuracies.csv'), 'w')
if args.resume is None:
    accuracy_handler.write('index,accuracy\n')

tot_exemplars = len(api)
starting_time = time.time()

# Iterate all the architectures
for index, exemplar in enumerate(api):
    # If resuming, just skip already done architectures.
    # The ugly way, but simpler than split the iterator. Who cares
    if args.resume is not None and index < args.resume:
        continue

    ctime = time.time()
    time_per_ex = (ctime - starting_time) / ((index - (args.resume or 0)) + 1)
    print(
        f'\rExemplar {index + 1} / {tot_exemplars}, {round((ctime - starting_time) / 60, 3)} minutes... average: {round(time_per_ex, 3)} s per exemplar. Expected remaining time: {round(time_per_ex * (tot_exemplars - (index - (args.resume or 0))) / 60, 3)} minutes')
    seed_all(seed)
    start = time.time()
    # compute the metrics
    # Each metric is computed 3 times.
    # metrics: dict metric name -> metric
    # times: dict metric name -> time for 3 metric computations
    # init_time: float time to init model 3 times
    start_time = time.time()

    if args.rand_batch:
        bl_arg = loader
    else:
        bl_arg = batch
    metrics, times, init_time = compute_tfm(exemplar, bl_arg, device, metrics=metric_names,
                                            ignore_errors=args.ignore_errors)

    accuracy = api.get_accuracy(exemplar)
    end_time = time.time()
    print(f"     |||| Done in {round(end_time - start_time, 3)} s")

    # Let's try to prevent CUDA Out-of-memory errors but not to slow down everything

    if index % 10 == 0:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    if args.show_correlation is not None and index % args.show_correlation == 0:
        accuracies.append(accuracy)
        for m_name in metric_names:
            metric_vals[m_name].append(metrics[m_name])

            corr, p = spearmanr(metric_vals[m_name], accuracies)
            print(f"     |||| Spearman of {m_name}: {corr:0.4f} (p = {p:0.4f})")

    # Write stuff and flush (so we don't lose data anyway)
    accuracy_handler.write(f'{index},{accuracy}\n')
    accuracy_handler.flush()
    for m_name in metric_names:
        line = f'{index},{metrics[m_name]},{(times[m_name] or 0) + init_time},{times[m_name]}\n'
        file_handlers[m_name].write(line)
        file_handlers[m_name].flush()

# Close everything
accuracy_handler.close()
for m_name in metric_names:
    file_handlers[m_name].close()

# Done :D
print("Done! :D")
