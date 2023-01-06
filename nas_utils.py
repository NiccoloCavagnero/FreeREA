from typing import Union, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import *
from scipy.stats import mode

from nas_spaces import NASSpaceBase, Exemplar

"""
    Dictionary mapping metric names to their implementation.
    
    Each metric function is expected to accept 4 arguments:
        - The model (torch.nn.Module)
        - An input batch (torch.Tenor)
        - A target tensor (labels for the input batch (torch.Tensor)
        - The device (torch.device)
    
    If the actual implementation doesn't need some of the args a lambda function can be used
"""
METRIC_NAME_MAP = {
    # log(x)
    'logsynflow': compute_synflow_per_weight,
    # x
    'synflow': lambda n, inputs, targets, dev: compute_synflow_per_weight(n, inputs, targets, dev, remap=None),

    'params': lambda n, _1, _2, _3: count_params(n) / 1e6,
    'macs': lambda n, inp, _1, _2: get_macs_and_params(n, inp.shape)[0],
    'naswot': compute_naswot_score,
}


def get_random_population(api: NASSpaceBase, N, generation=0):
    exemplars = [api.random().set_generation(generation) for _ in range(N)]
    return exemplars


def population_init(api, N, max_flops=float('inf'), max_params=float('inf'), analyzer=None, start=0.0, metrics=None):
    population = []
    while len(population) < N:
        new = get_random_population(api, 1)[0]
        if is_feasible(new, max_flops, max_params):
            population.append(new)
        if analyzer:
            analyzer.update(population, start, metrics)
    return population


def is_feasible(exemplar, max_flops=float('inf'), max_params=float('inf')):
    cost = exemplar.get_cost_info()
    if cost['flops'] <= max_flops and cost['params'] <= max_params:
        return True
    return False


def compute_tfm(exemplar: Exemplar,
                batch_or_loader: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
                device: torch.device,
                metrics: Tuple = tuple(METRIC_NAME_MAP.keys()),
                ignore_errors: bool = False):
    if not exemplar.born:
        network = exemplar.space.get_network(exemplar, device=device)
        network = network.to(device)

        metric_trials = {}
        metric_times = {}

        init_time = 0
        for i in range(3):
            if isinstance(batch_or_loader, tuple):
                inputs, targets = batch_or_loader
            elif isinstance(batch_or_loader, DataLoader):
                inputs, targets = next(iter(batch_or_loader))
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                raise ValueError("Invalid argument")

            start_time = time.time()
            network = init_model(network)
            end_time = time.time()
            init_time += (end_time - start_time)

            for metric_name in metrics:
                start_time = time.time()
                try:
                    val = METRIC_NAME_MAP[metric_name](network, inputs, targets, device)
                except RuntimeError as ex:
                    if ignore_errors:
                        # In case of errors set the value to None but keep running!
                        val = None
                    else:
                        raise ex
                end_time = time.time()
                if metric_name not in metric_trials:
                    metric_trials[metric_name] = []
                    metric_times[metric_name] = []
                metric_trials[metric_name].append(val)
                metric_times[metric_name].append(end_time - start_time)

        for metric_name in metrics:
            if None in metric_trials[metric_name]:
                metric_trials[metric_name] = None
                metric_times[metric_name] = None
            else:
                metric_trials[metric_name] = np.mean(metric_trials[metric_name])
                metric_times[metric_name] = np.sum(metric_times[metric_name])

        return metric_trials, metric_times, init_time


def return_top_k(exemplars, K=3, metric_names=[]):
    exemplars = [exemplar for exemplar in exemplars if exemplar.born]

    values_dict = {}
    for metric_n in metric_names:
        if metric_n == 'skip':
            values_dict['skip'] = np.array([exemplar.skip() for exemplar in exemplars])
        else:
            values_dict[metric_n] = np.array([exemplar.get_metric(metric_n) for exemplar in exemplars])

    scores = np.zeros(len(exemplars))
    scores_dict = {}
    for metric_n in metric_names:
        if metric_n == 'ntk':
            values_dict[metric_n] = 1 - values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        else:
            values_dict[metric_n] = values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        scores_dict[metric_n] = values_dict[metric_n]
        scores += values_dict[metric_n]

    for idx, (exemplar, rank) in enumerate(zip(exemplars, scores)):
        exemplar.rank = rank

    exemplars.sort(key=lambda x: -x.rank)
    return exemplars[:K]


def get_max_accuracy(api: NASSpaceBase, max_flops: float, max_params: float):
    best, best_accuracy = None, 0.0
    for idx in range(len(api)):
        info = api.get_cost_info(idx)
        if info['flops'] <= max_flops and info['params'] <= max_params:
            accuracy = api.get_accuracy(api[idx])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best = idx
                print(best, best_accuracy)
    return best_accuracy


def kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def init_model(model):
    model.apply(kaiming_normal)
    return model


def dictionary_update(exemplars, history, replace=True):
    if replace:
        # Update already seen genotypes
        history.update({exemplar.genotype: exemplar for exemplar in exemplars})
    else:
        # Add new genotypes
        history.update({exemplar.genotype: exemplar for exemplar in exemplars if exemplar.genotype not in history})
        # For already seen exemplars, just update the generation
        for exemplar in exemplars:
            history[exemplar.genotype].generation = exemplar.generation

    # This is also removing exemplars with same genotype from population
    current_genotypes = set([exemplar.genotype for exemplar in exemplars])
    exemplars = [history[genotype] for genotype in current_genotypes]
    return exemplars, history


def clean_history(history, max_params, max_flops):
    return {genotype: exemplar for genotype, exemplar in history.items()
            if exemplar.get_cost_info()['params'] <= max_params
            and exemplar.get_cost_info()['flops'] <= max_flops}


def get_top_k_accuracies(exemplars, K=3, metrics=[]):
    best_K, acc = return_top_k(exemplars, K, metrics), []
    for exemplar in best_K:
        idx = exemplar.idx
        acc.append((idx, round(exemplar.get_accuracy(), 3)))
    return acc, [exemplar.genotype for exemplar in best_K], [exemplar.rank for exemplar in best_K]


def info(api, exemplars, history, start, metrics, max_flops, max_params):
    history_feasible = clean_history(history, max_params, max_flops).values()
    acc_history, genotypes_hist, ranks_hist = get_top_k_accuracies(history_feasible, K=3, metrics=metrics)
    exemplars_feasible = [exemplar for exemplar in exemplars if exemplar.born and
                          exemplar.get_cost_info()['flops'] <= max_flops and
                          exemplar.get_cost_info()['params'] <= max_params]
    acc_pop, _, _ = get_top_k_accuracies(exemplars_feasible, K=3, metrics=metrics)
    accuracies = [exemplar.get_accuracy() for exemplar in history_feasible]

    metrics_time = api.total_metrics_time(history_feasible, metrics)
    search_time = time.time() - start
    total_time = metrics_time + search_time

    # This is not a correct number if we loose the constraints at the beginning
    print(f'\n|||| {len(history_feasible)} different cells explored..')
    print(f'|||| Max acc: {round(max(accuracies), 2)}, Avg acc: {round(np.mean(accuracies), 2)}, Std acc: {round(np.std(accuracies), 2)}..')
    print(f'|||| Accuracies of top 3 visited cells: {acc_history}')
    print(f'|||| Ranks of top 3 visited cells: {ranks_hist}')
    print(f'|||| Genotype of top 3 visited cells:')
    for index, genotype in enumerate(genotypes_hist):
        print(f'     |||| {acc_history[index][0]} : {genotype}')
    print(f'|||| Accuracies of top 3 surviving cells: {acc_pop}')
    print(f'|||| Search Time: {round(search_time / 60, 2)} minutes..')
    print(f'|||| Metrics Time: {round(metrics_time / 60, 2)} minutes..')
    print(f'|||| Total Time: {round(total_time / 60, 2)} minutes..')

    return acc_history[0][1], total_time / 60


def edit_distance(exemplar1, exemplar2):
    genes1 = genotype_to_gene_list(exemplar1.genotype)
    genes2 = genotype_to_gene_list(exemplar2.genotype)

    count = 0
    for gene1, gene2 in zip(genes1, genes2):
        if gene1 != gene2:
            count += 1
    return count


def genotype_to_gene_list(genotype):
    out = []
    levels = genotype.split('+')
    for level in levels:
        level = level.split('|')[1:-1]
        for i in range(len(level)):
            out.append(level[i])
    return out


def gene_list_to_genotype(gene_list):
    gene_list = gene_list[0][0]
    out = '|' + gene_list[0] + '|+|'
    for idx, gene in enumerate(gene_list[1:]):
        out += gene + '|'
        if idx == 1:
            out += '+|'
    return out


def mean_exemplar(exemplars, K=5, metrics=[]):
    _, genotypes_hist, _ = get_top_k_accuracies(exemplars, K=K, metrics=metrics)

    top_genotypes = [genotype_to_gene_list(genotype) for genotype in genotypes_hist]

    mean_genes = mode(top_genotypes, axis=0)
    mean_genotype = gene_list_to_genotype(mean_genes)
    return mean_genotype


class EarlyStop:
    def __init__(self, patience=20):
        super().__init__()
        self.patience = patience
        self.waiting = 0
        self.n_pop = 0

    def stop(self, history):
        n = len(history)
        if n >= self.n_pop:
            self.n_pop = n
            self.waiting = 0
        else:
            self.waiting += 1
            if self.patience >= self.waiting:
                print('\n   Early Stop...')
                return True
        return False
