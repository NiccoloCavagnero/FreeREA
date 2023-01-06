import time
import pandas as pd

from nas_utils import get_top_k_accuracies


class Analyzer:
    def __init__(self, api, dataset, algorithm):
        super().__init__()

        self.api = api
        self.dataset = dataset
        self.algorithm = algorithm
        self.accuracies = []
        self.times = []
        self.current_run = - 1

    def new_run(self):
        self.accuracies.append([])
        self.times.append([])
        self.current_run += 1

    def update(self, history, start, metrics):
        if type(history) == dict:
            population = list(history.values())
        else:
            population = history
        if 'val_accuracy' not in metrics:
            accuracy = get_top_k_accuracies(population, 1, metrics)[0][0][1]
            total_time = self.api.total_metrics_time(population, metrics=metrics)
        else:
            population.sort(key=lambda x: x.get_val_accuracy())
            accuracy = population[-1].get_accuracy()
            total_time = self.api.total_train_and_eval_time(population)
        total_time += time.time() - start  # search time

        self.accuracies[self.current_run].append(accuracy)
        self.times[self.current_run].append(total_time)

    def save(self):
        df = pd.DataFrame()

        for idx in range(self.current_run+1):
            other = pd.DataFrame()
            other['accuracy_'+str(idx)] = self.accuracies[idx]
            other['time_' + str(idx)] = self.times[idx]

            df = pd.concat([df, other], axis=1)

        df.index.name = 'index'
        df.to_csv(f'results/{self.api.get_api_name()}/{self.dataset}/{self.algorithm}')








