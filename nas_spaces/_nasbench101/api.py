from .original_api import api as nasbench101api
from .model import Network
from .model_spec import ModelSpec
import itertools
import random
import numpy as np


class NASBench101:
    def __init__(self, dataset, data_path, config=None, verbose=True, progress=True):
        self.dataset = dataset
        self.api = nasbench101api.NASBench(data_path, verbose=verbose, progress=progress)

        config = config or {
            'stem_out_channels': 128,
            'num_stacks': 3,
            'num_modules_per_stack': 3,
            'num_labels': 10
        }
        self.config = config

    def get_max_accuracy(self, unique_hash):
        spec = self.get_spec(unique_hash)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = 0.
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
        return maxacc

    """
        def query(self, unique_hash, epochs=108, stop_halfway=False):
        return self.api.query(self.get_spec(unique_hash), epochs, stop_halfway)
    """

    def get_accuracy(self, unique_hash, epochs=108, stop_halfway=False, trial=None):

        """
        Get the test accuracy of the architecture identified by the hash
        :param unique_hash:     Hash of the architecture
        :param epochs:          Training epochs: 4, 12, 36, 108
        :param stop_halfway:    Halfway or final accuracy?
        :param trial:           Trial. 0 to 2. If not specified, you get the mean. Pass 'all' to get all of them
        :return:
        """

        spec = self.get_spec(unique_hash)

        _, computed_stats = self.api.get_metrics_from_spec(spec)
        computed_stats = computed_stats[epochs]
        prefix = 'halfway' if stop_halfway else 'final'
        accuracies = np.array([x[f'{prefix}_test_accuracy'] for x in computed_stats])

        if trial is None:
            return accuracies.mean()
        elif trial == 'all':
            return accuracies
        else:
            return accuracies[trial].item()

    def get_training_time(self, unique_hash):
        spec = self.get_spec(unique_hash)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = -1.
        maxtime = 0.
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
                    maxtime = statmap['final_training_time']
        return maxtime

    def get_network(self, unique_hash):
        spec = self.get_spec(unique_hash)
        network = Network(spec, self.config)
        return network

    def get_spec(self, unique_hash):
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        spec = ModelSpec(matrix, operations)
        return spec

    def __iter__(self):
        for unique_hash in self.api.hash_iterator():
            network = self.get_network(unique_hash)
            yield unique_hash, network

    def __getitem__(self, index):
        return next(itertools.islice(self.api.hash_iterator(), index, None))

    def __len__(self):
        return len(self.api.hash_iterator())

    def num_activations(self):
        for unique_hash in self.api.hash_iterator():
            network = self.get_network(unique_hash)
            return network.classifier.in_features

    def train_and_eval(self, arch, dataname, acc_type, trainval=True, traincifar10=False):
        unique_hash = self.__getitem__(arch)
        time = 12. * self.get_training_time(unique_hash) / 108.
        acc = self.get_max_accuracy(unique_hash, acc_type, trainval)
        return acc, acc, time

    def random_arch(self):
        return random.randint(0, len(self) - 1)

    def mutate_arch(self, arch):
        unique_hash = self.__getitem__(arch)
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        coords = [(i, j) for i in range(matrix.shape[0]) for j in range(i + 1, matrix.shape[1])]
        random.shuffle(coords)
        # loop through changes until we find change thats allowed
        for i, j in coords:
            # try the ops in a particular order
            for k in [m for m in np.unique(matrix) if m != matrix[i, j]]:
                newmatrix = matrix.copy()
                newmatrix[i, j] = k
                spec = ModelSpec(newmatrix, operations)
                try:
                    newhash = self.api._hash_spec(spec)
                    if newhash in self.api.fixed_statistics:
                        return [n for n, m in enumerate(self.api.fixed_statistics.keys()) if m == newhash][0]
                except:
                    pass
