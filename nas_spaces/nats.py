from copy import deepcopy as copy
import os.path
from typing import Union, Text, Sequence

import numpy as np
import torch
from torch.nn import Module
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

from .utils import load_metrics
from . import NASSpaceBase, Exemplar

__all__ = ['NATSBench']


class _NetWrapper(Module):

    def __init__(self, net: Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, get_ints=False):
        x = self.net(x)
        if get_ints:
            return x[1], x[0]
        else:
            return x[1]


class NATSBench(NASSpaceBase):

    _all_ops = {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3'}

    def __init__(self, path: str, dataset: str, verbose: bool = False, metric_root=None):
        super().__init__(path, verbose)

        # Value check the dataset
        known_datasets = ('cifar10', 'cifar100', 'ImageNet16-120')
        if dataset not in known_datasets:
            raise ValueError(f"Unknown dataset {dataset}. Dataset must be one of: {', '.join(known_datasets)}.")
        self._dataset = dataset

        # Init wrapped API
        self._official_api = create(path, 'tss', fast_mode=True, verbose=verbose)

        # Load metrics
        if metric_root is not None:
            self.metrics_cache = load_metrics(os.path.join(metric_root, dataset))

    @staticmethod
    def get_api_name():
        return 'nats'

    def __len__(self):
        return len(self._official_api)

    def __iter__(self):
        # Returns an iterator
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item: int) -> Exemplar:
        # In the original wrapped API __getitem__ returns just the architecture string, without configuration
        # depending on the dataset

        config = self._official_api.get_net_config(item, self._dataset)
        return Exemplar(self, item, genotype=config['arch_str'])

    def get_index(self, x: Union[Exemplar, Text]):
        if isinstance(x, Exemplar):
            genotype = x.genotype
        elif isinstance(x, str):
            genotype = x
        else:
            raise ValueError(f"Invalid argument {x}")

        index = self._official_api.query_index_by_arch(genotype)
        if index < 0:
            raise KeyError(f"Unknown architeture: {genotype}")

        return index

    def get_accuracy(self, spec: Exemplar, epochs: int = 200) -> float:
        index = spec.idx
        info = self._official_api.get_more_info(index, self._dataset, hp=epochs, is_random=False)
        return info['test-accuracy']

    def get_val_accuracy(self, spec: Exemplar, epochs: int = 12) -> float:  # or 90 epochs
        index = spec.idx
        validation_accuracy, latency, time_cost, current_total_time_cost = \
            self._official_api.simulate_train_eval(index, dataset=self._dataset, hp=epochs)
        return validation_accuracy

    def get_cost_info(self, spec: Union[Exemplar, int]) -> dict:
        metrics = {
            'params': None,
            'latency': None,
            'flops': None
        }

        if isinstance(spec, Exemplar):
            index = spec.idx
        elif isinstance(spec, int):
            index = spec
        else:
            raise ValueError(f"Invalid argument {spec}. Type: {type(spec)}")

        full_cost_info = self._official_api.get_cost_info(index, self._dataset)

        for k in metrics:
            metrics[k] = full_cost_info[k]

        return metrics

    def total_metrics_time(self, exemplars: Sequence[Exemplar], metrics: Sequence[str]) -> float:
        indexes = [exemplar.idx for exemplar in exemplars]
        metrics = [metric + '_time=[s]' for metric in metrics if metric != 'skip' and metric != 'params']
        return self.metrics_cache.loc[indexes, metrics].sum(axis=1).sum(axis=0)

    def total_train_and_eval_time(self, exemplars: Sequence[Exemplar], epochs: int = 12) -> float:
        indexes = [exemplar.idx for exemplar in exemplars]
        time = 0.0
        for index in indexes:
            _, _, time_cost, _ = self._official_api.simulate_train_eval(index, dataset=self._dataset, hp=epochs)
            time += time_cost
        return time

    def get_network(self, spec: Exemplar, device: torch.device = None) -> Module:
        net = get_cell_based_tiny_net(self._official_api.get_net_config(spec.idx, self._dataset))

        net = _NetWrapper(net)

        if device is not None:
            net = net.to(device)

        return net

    def _get_metric_val(self, index: int, metric_name: str):
        return self.metrics_cache.at[index, metric_name]

    @staticmethod
    def _skip(exemplar: Exemplar) -> int:
        levels = exemplar.genotype.split('+')
        max_len = 0
        counter = 0

        for idx, level in enumerate(levels):
            level = level.split('|')[1:-1]
            n_genes = len(level)

            for i in range(n_genes):
                if 'skip' in level[i]:
                    counter += 1
                    min_edge = idx - i
                    max_len += min_edge
        if counter:
            return max_len / counter
        return 0

    @staticmethod
    def _get_different_gene(gene):
        suitable = copy(NATSBench._all_ops)
        suitable.remove(gene)
        gene = np.random.choice(list(suitable))
        return gene + '~'

    def mutation(self, exemplar: Exemplar, R: int = 1) -> Exemplar:
        genotype = exemplar.genotype.split('+')
        levels = []
        for index, level in enumerate(genotype):
            level = level.split('|')[1:-1]
            levels.append(level)

        # modify a gene
        for _ in range(R):
            # Level choice weighted by the number of genes per level
            chosen_level = np.argmax(np.random.multinomial(1, [1 / 6, 2 / 6, 3 / 6]))
            mutating_gene = np.random.randint(chosen_level + 1)
            levels[chosen_level][mutating_gene] = self._get_different_gene(
                levels[chosen_level][mutating_gene][:-2]) + str(
                mutating_gene)

        for idx, level in enumerate(levels):
            levels[idx] = '|' + '|'.join(level) + '|'
        arch_string = '+'.join(levels)

        index = self.get_index(arch_string)
        return self[index]

    def crossover(self, exemplars: Sequence[Exemplar]):
        assert len(exemplars) == 2

        genotype1 = exemplars[0].genotype
        genotype2 = exemplars[1].genotype

        levels1 = genotype1.split('+')
        levels2 = genotype2.split('+')

        new_genotype = ''
        for level_1, level_2 in zip(levels1, levels2):
            level_1 = level_1.split('|')[1:-1]
            level_2 = level_2.split('|')[1:-1]
            n_genes = len(level_1)

            new_genotype += '|'
            for i in range(n_genes):
                choice = np.random.binomial(1, 0.5)
                if choice == 0:
                    new_genotype += level_1[i] + '|'
                else:
                    new_genotype += level_2[i] + '|'
            new_genotype += '+'

        index = self.get_index(new_genotype[:-1])
        return self[index]