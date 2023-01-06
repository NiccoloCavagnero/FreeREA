import abc
from typing import Union, Text, Sequence
import torch
import random as rnd
from torch.nn import Module


class Exemplar:
    def __init__(self, space, idx, genotype, gen=0):
        super().__init__()

        self.space: NASSpaceBase = space
        self.idx = idx
        self.generation = gen
        self.genotype = genotype

        self.rank = None

        self.born = False

        self._cost_info = None
        self._metrics = None
        self.val_accuracy = None

    def get_metric(self, metric_name: str) -> float:
        return self.space._get_metric_val(self.idx, metric_name)

    def get_cost_info(self):
        if self._cost_info is not None:
            return self._cost_info
        else:
            self._cost_info = self.space.get_cost_info(self.idx)
        return self._cost_info

    def skip(self) -> int:
        return self.space._skip(self)

    def set_generation(self, gen):
        self.generation = gen

        return self

    def get_accuracy(self):
        return self.space.get_accuracy(self)

    def get_val_accuracy(self):
        if self.val_accuracy is None:
            self.val_accuracy = self.space.get_val_accuracy(self)
        return self.val_accuracy

    def get_network(self, device: torch.device) -> Module:
        return self.space.get_network(self, device=device)


class NASSpaceBase:

    def __init__(self, path: str, verbose: bool = False):
        ...

    @abc.abstractmethod
    def __len__(self):
        """
        :return:        The number of architectures in the search space
        """
        ...

    @abc.abstractmethod
    def __iter__(self):
        """
        :return:        An iterator to iterate across all the architectures
        """
        ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> Exemplar:
        """
        :param item:    Index of the architecture
        :return:        A representation of the architecture
        """
        ...

    @abc.abstractmethod
    def get_accuracy(self, spec: Exemplar) -> float:
        """
        :param spec:    Representation of the architecture
        :return:        Final test accuracy
        """
        ...

    @abc.abstractmethod
    def get_cost_info(self, spec: Exemplar) -> dict:
        """
        :param spec:    Representation of the architecture
        :return:        Dict containing cost info (latency, params)
        """
        ...

    @abc.abstractmethod
    def get_network(self, spec: Exemplar, device: torch.device = None) -> Module:
        """
        Build the network
        :param spec:        Network architecture specification
        :param device:      Device (CPU if not specified)
        :return:
        """
        ...

    @abc.abstractmethod
    def get_index(self, x: Union[Exemplar, Text]) -> int:
        """
        Get network index from exemplar or genotype
        :param x:
        :return:
        """
        ...

    @abc.abstractmethod
    def _get_metric_val(self, index: int, metric_name: str):
        ...

    @abc.abstractmethod
    def _skip(self, exemplar: Exemplar) -> bool:
        ...

    @abc.abstractmethod
    def mutation(self, exemplar: Exemplar, R: int = 1) -> Exemplar:
        """
        Returns an exemplar defining a mutated architecture
        :param exemplar:
        :param R:   consecutive mutations
        :return:
        """
        ...

    @abc.abstractmethod
    def crossover(self, exemplars: Sequence[Exemplar]) -> Exemplar:
        """
        For just two exemplars
        :param exemplars:
        :return:
        """
        ...

    def random(self) -> Exemplar:
        """
        Get a random architecture
        :return:
        """
        return self[rnd.randint(0, len(self) - 1)]

    def total_metrics_time(self, exemplars: Sequence[Exemplar], metrics: Sequence[str]) -> float:
        """
        Get total computation time for the metrics
        :return:
        """
        ...

    def total_train_and_eval_time(self, exemplars: Sequence[Exemplar], epochs: int) -> float:
        """
        Get total computation time for training and evaluation
        :return:
        """
        ...


METRIC_NAMES = {'nregions', 'ntk', 'path_norm', 'synflow', 'snip', 'jacob', 'grad_norm', 'fisher', 'grasp', 'grad_sign',
                'sensitivity'}
