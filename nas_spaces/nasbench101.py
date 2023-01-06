import os
import random
from copy import deepcopy as copy
import itertools
from typing import Union, Text, Tuple, Sequence
import numpy as np
import torch
from torch.nn import Module

from .utils import load_metrics
from . import NASSpaceBase, Exemplar
from ._nasbench101.original_api import api as n101api
from ._nasbench101.model_spec import ModelSpec
from ._nasbench101.model import Network
from ._nasbench101.base_ops import OP_MAP
from ._nasbench101.graph_util import vertex_min_path, vertex_max_path


__all__ = ['NASBench101']


class NASBench101(NASSpaceBase):

    _all_ops = list(OP_MAP.keys())

    def __init__(self, path: str, verbose: bool = False, progress: bool = False, epochs: int = 108, metric_root=None):
        super().__init__(path, verbose)

        self._original_api = n101api.NASBench(path, verbose=verbose, progress=progress)
        config = {
            'stem_out_channels': 128,
            'num_stacks': 3,
            'num_modules_per_stack': 3,
            'num_labels': 10
        }
        self._config = config
        self._epochs = epochs

        # Load metrics
        if metric_root is not None:
            self.metrics_cache = load_metrics(os.path.join(metric_root, 'cifar10'))

    @staticmethod
    def get_api_name():
        return 'nasbench101'

    @staticmethod
    def _get_different_gene(gene):
        suitable = copy(NASBench101._all_ops)
        suitable.remove(gene)
        gene = np.random.choice(list(suitable))
        return gene + ':'

    def __iter__(self):
        it = self._original_api.hash_iterator()
        for i, h in enumerate(it):
            genotype = self._get_genotype(self._get_spec(i))
            yield Exemplar(self, i, genotype=genotype)

    def __getitem__(self, item: int) -> Exemplar:
        spec = self._get_spec(item)
        genotype = self._get_genotype(spec)
        return Exemplar(self, item, genotype=genotype)

    def _get_spec(self, ex: Union[int, Exemplar]):
        if isinstance(ex, int):
            i = ex
        elif isinstance(ex, Exemplar):
            i = ex.idx
        else:
            raise ValueError(f"Invalid argument {ex}")

        unique_hash = next(itertools.islice(self._original_api.hash_iterator(), i, None))
        matrix = self._original_api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self._original_api.fixed_statistics[unique_hash]['module_operations']
        spec = ModelSpec(matrix, operations)

        return spec

    def get_accuracy(self, ex: Exemplar) -> float:
        spec = self._get_spec(ex.idx)

        _, computed_stats = self._original_api.get_metrics_from_spec(spec)
        computed_stats = computed_stats[self._epochs]
        accuracies = np.array([x[f'final_test_accuracy'] for x in computed_stats])

        return accuracies.mean().item() * 100

    def get_val_accuracy(self, ex: Exemplar, epoch: int = 36) -> float:
        spec = self._get_spec(ex.idx)

        _, computed_stats = self._original_api.get_metrics_from_spec(spec)
        computed_stats = computed_stats[epoch]
        accuracies = np.array([x[f'final_validation_accuracy'] for x in computed_stats])

        return np.random.choice(accuracies) * 100

    def get_cost_info(self, spec: Union[Exemplar, int]) -> dict:
        metrics = {
            'params': None,
            'latency': None,  # Not available
            'flops': 0   # We will get this from thop
        }

        if isinstance(spec, Exemplar):
            index = spec.idx
        elif isinstance(spec, int):
            index = spec
        else:
            raise ValueError(f"Invalid argument {spec}")

        unique_hash = next(itertools.islice(self._original_api.hash_iterator(), index, None))
        metrics['params'] = self._original_api.fixed_statistics[unique_hash]['trainable_parameters'] / 1e6

        return metrics

    def total_metrics_time(self, exemplars: Sequence[Exemplar], metrics: Sequence[str]) -> float:
        indexes = [exemplar.idx for exemplar in exemplars]
        metrics = [metric + '_time=[s]' for metric in metrics if metric != 'skip' and metric != 'params']
        return self.metrics_cache.loc[indexes, metrics].sum(axis=1).sum(axis=0)

    def total_train_and_eval_time(self, exemplars: Sequence[Exemplar], epochs: int = 36) -> float:
        indexes = [exemplar.idx for exemplar in exemplars]
        time = 0.0
        for index in indexes:
            spec = self._get_spec(index)
            _, computed_stats = self._original_api.get_metrics_from_spec(spec)
            avg_run_time = 0.0
            for run in range(3):
                avg_run_time += computed_stats[epochs][run]['final_training_time']
            time += avg_run_time / 3
        return time

    def get_network(self, ex: Exemplar, device: torch.device = None) -> Module:
        sp = self._get_spec(ex.idx)
        net = Network(sp, self._config)

        if device is not None:
            net = net.to(device)

        return net

    def get_index(self, x: Union[Exemplar, Text]) -> int:
        if isinstance(x, Exemplar):
            return x.idx
        elif isinstance(x, str):
            index, _ = self._from_genotype(x)
            return index
        else:
            raise ValueError(f"Invalid argument {x}")

    def _get_metric_val(self, index: int, metric_name: str):
        return self.metrics_cache.at[index, metric_name]

    def __len__(self):
        return len(self._original_api.hash_iterator())

    def _has_skip(self, exemplar: Exemplar) -> bool:
        matrix = self._get_spec(exemplar.idx).matrix
        return vertex_min_path(matrix)[-1] != vertex_max_path(matrix)[-1]

    @staticmethod
    def _skip(exemplar: Exemplar) -> int:
        ops = exemplar.genotype.split(';')
        max_len = 0
        counter = 0
        for idx, op in enumerate(ops[1:]):
            edges = op.split(':')[1]
            if len(edges) > 1:
                for edge in edges.split(','):
                    edge = int(edge)
                    skipped += idx - edge
                    if skipped:
                        counter += 1
                        max_len += skipped
            else:
                min_edge = int(edges)
                skipped = idx - min_edge
                if skipped:
                    counter += 1
                    max_len += skipped  # total skipped layers
        if counter:
            return max_len / counter
        return 0

    @staticmethod
    def _get_genotype(spec: ModelSpec) -> str:
        assert spec.ops[0] == 'input'
        assert spec.ops[-1] == 'output'

        l = []
        num_nodes = spec.matrix.shape[0]
        for i in range(num_nodes):
            links = (map(str, spec.matrix[:, i].nonzero()[0]))
            links = (','.join(links))
            l.append(spec.ops[i] + (':' + links if i != 0 else ''))
        return ';'.join(l)

    def _from_genotype(self, genotype: str) -> Tuple[int, Text]:
        node_strs = genotype.split(';')
        assert node_strs[0] == 'input'

        num_nodes = len(node_strs)
        m = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
        ops = [node_strs[0]]

        for i, node_str in enumerate(node_strs[1:]):
            node_op, node_inputs = node_str.split(':')
            ops.append(node_op)
            node_inputs = list(map(int, node_inputs.split(',')))
            m[node_inputs, i + 1] = 1

        spec = ModelSpec(m, ops)
        if not self._original_api.is_valid(spec):
            raise ValueError(f"Invalid genotype {genotype}")

        # Recompute genotype
        arch_hash = self._original_api._hash_spec(spec)
        genotype = self._get_genotype(spec)
        index = self._original_api.fixed_statistics[arch_hash]['index']

        return index, genotype

    def from_genotype(self, genotype: str) -> Exemplar:
        index, genotype2 = self._from_genotype(genotype)
        return Exemplar(self, index, genotype2)

    def mutation(self, exemplar: Exemplar, R: int = 1) -> Exemplar:
        genotype = exemplar.genotype

        for _ in range(R):
            edges = self._count_edges(genotype)
            ops = genotype.split(';')
            nodes = len(ops)

            if edges < 9 and nodes < 7:
                # we can add a node or an edge
                if np.random.binomial(1, 0.5):
                    if np.random.binomial(1, 0.5):   # 0.75
                        # add a node
                        ops = self._add_node(ops)
                    else:
                        # add an edge
                        ops = self._add_edge(ops)
                else:
                    # modify a node or an edge
                    ops = self._modify_gene(ops, nodes, edges)
            elif nodes < 7:
                # we can add a node
                if np.random.binomial(1, 0.5):
                    # add a node
                    ops = self._add_node(ops)
                else:
                    # modify a node or an edge
                    ops = self._modify_gene(ops, nodes, edges)
            elif edges < 9:
                # we can add or remove an edge
                if edges >= len(ops):
                    if np.random.binomial(1, 0.5):
                        # add an edge
                        ops = self._add_edge(ops)
                    else:
                        # modify a node or an edge
                        ops = self._modify_gene(ops, nodes, edges)
                # we can only add an edge
                else:
                    # add an edge
                    ops = self._add_edge(ops)
            else:
                # we can just modify a gene
                ops = self._modify_gene(ops, nodes, edges)
            genotype = ';'.join(ops)
        return self.from_genotype(genotype)

    def crossover(self, exemplars: Sequence[Exemplar]) -> Exemplar:
        assert len(exemplars) == 2
        genotype1 = exemplars[0].genotype.split(';')[1:]
        genotype2 = exemplars[1].genotype.split(';')[1:]
        out1, len1 = genotype1[-1], len(genotype1)
        out2, len2 = genotype2[-1], len(genotype2)

        new_ops = ['input']

        for gene1, gene2 in zip(genotype1, genotype2):
            choice = np.random.binomial(1, 0.5)
            if choice == 0:
                new_ops.append(gene1)
            else:
                new_ops.append(gene2)

        if 'output' not in new_ops[-1]:
            if len1 > len2:
                new_ops[-1] = out2
            elif len2 > len1:
                new_ops[-1] = out1

        new_genotype = ';'.join(new_ops)
        while self._count_edges(new_genotype) > 9:
            new_ops = self._remove_edge(new_genotype.split(';'))
            new_genotype = ';'.join(new_ops)
        return self.from_genotype(new_genotype)

    @staticmethod
    def _count_edges(genotype: str) -> int:
        counter = 0
        ops = genotype.split(';')[1:]
        for op in ops:
            op = op.split(':')[1:]
            for inputs in op:
                inputs = inputs.split(',')
                counter += len(inputs)
        return counter

    @staticmethod
    def _modify_node(ops: Sequence[str]) -> Sequence[str]:
        if len(ops) - 1 <= 1:
            return ops
        index = np.random.randint(1, len(ops)-1)
        to_be_modified = ops[index].split(':')
        ops[index] = NASBench101._get_different_gene(to_be_modified[0]) + to_be_modified[1]
        return ops

    def modify_edge(self, ops: Sequence[str]) -> Sequence[str]:
        ops = self._remove_edge(ops)
        ops = self._add_edge(ops)
        return ops

    @staticmethod
    def _remove_node(ops: Sequence[str]) -> Sequence[str]:
        new_ops = ['input']
        to_remove = np.random.randint(1, len(ops)-1)
        ops.pop(to_remove)
        for idx, node in enumerate(ops[1:]):
            node = node.split(':')
            edges = [int(edge) for edge in node[1].split(',')]
            node = node[0] + ':' + ','.join(set([str(np.clip(edge, 0, idx)) for edge in edges]))
            new_ops.append(node)
        return new_ops

    @staticmethod
    def _remove_edge(ops: Sequence[str]) -> Sequence[str]:
        indexes = []
        all_edges = []
        for idx, op in enumerate(ops[1:]):
            edges = op.split(':')[1]
            edges = list(map(lambda x: int(x), edges.split(',')))
            all_edges.append(edges)
            if len(edges) > 1:
                indexes.append(idx)
        # cannot remove an edge, mutation failed
        if not len(indexes):
            return ops
        chosen_node = np.random.choice(indexes)
        chosen_edge = np.random.choice(all_edges[chosen_node])
        all_edges[chosen_node].remove(chosen_edge)

        ops[chosen_node + 1] = ops[chosen_node + 1].split(':')[0] + ':' + ','.join(
            list(map(str, all_edges[chosen_node])))

        return ops

    def _modify_gene(self, ops: Sequence[str], nodes: int, edges: int) -> Sequence[str]:
        if np.random.binomial(1, 0.5):    # 0.25
            # node ops
            if nodes <= 3:
                # we can only modify a node
                ops = self._modify_node(ops)
            else:
                # we can modify or remove a node
                if np.random.binomial(1, 0.5):
                    # modify the node
                    ops = self._modify_node(ops)
                else:
                    ops = self._remove_node(ops)
        else:
            # edge ops
            if edges <= nodes - 1:
                # we can only modify the edge
                ops = self.modify_edge(ops)
            else:
                # we can modify or remove the edge
                if np.random.binomial(1, 0.5):
                    # modify the edge
                    ops = self.modify_edge(ops)
                else:
                    ops = self._remove_edge(ops)
        return ops

    @staticmethod
    def _add_node(ops: Sequence[str]) -> Sequence[str]:
        if len(ops) - 1 > 1:
            position = np.random.randint(1, len(ops)-1)
        else:
            position = len(ops) - 2
        if position > 1:
            input_node = np.random.randint(0, position)
            new_node = random.choice(NASBench101._all_ops)
            ops.insert(position, f'{new_node}:{input_node}')
        return ops

    @staticmethod
    def _add_edge(ops: Sequence[str]) -> Sequence[str]:
        # find possible nodes
        possible_nodes = []
        for idx, op in enumerate(ops[2:]):
            existing_edges = list(map(lambda x: int(x), op.split(':')[1].split(',')))
            if len(existing_edges) < idx + 2:
                possible_nodes.append(idx + 2)

        # there are no suitable nodes, return ops
        if not len(possible_nodes):
            return ops

        # find possible edges
        chosen_node = np.random.choice(possible_nodes)
        existing_edges = list(map(lambda x: int(x), ops[chosen_node].split(':')[1].split(',')))
        possible_edges = [edge for edge in np.arange(0, chosen_node) if edge not in existing_edges]

        # no possible edges, mutation failed
        if not len(possible_edges):
            return ops

        chosen_edge = np.random.choice(possible_edges)
        existing_edges.append(chosen_edge)
        existing_edges.sort()

        ops[chosen_node] = ops[chosen_node].split(':')[0] + ':' + ','.join(list(map(str, existing_edges)))
        return ops
