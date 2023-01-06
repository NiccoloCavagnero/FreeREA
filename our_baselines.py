import random

from nas_utils import *
from genotypes import mutate


def freeREAMinus(api, max_time, N=25, n=5, max_flops=float('inf'), max_params=float('inf'), analyzer=None):
    metrics = ['naswot', 'logsynflow', 'skip']
    start = time.time()
    population = population_init(api, N, max_flops, max_params)
    for exemplar in population:
        exemplar.born = True
    history = {exemplar.genotype: exemplar for exemplar in population}

    step = 0
    total_time = (api.total_metrics_time(history.values(), metrics=metrics) + time.time() - start) / 60
    earlyStop = EarlyStop(5)
    while total_time <= max_time:

        while True:
            sampled = random.choices(population, k=n)
            parent = return_top_k(sampled, 1, metrics)
            offspring = mutate(api, parent, R=1, P=1, cross=False, generation=step+1)[0]
            if is_feasible(offspring, max_flops, max_params):
                break
        offspring.born = True
        history[offspring.genotype] = offspring
        population.append(offspring)
        population.pop(0)
        step += 1
        total_time = (api.total_metrics_time(history.values(), metrics=metrics) + time.time() - start) / 60
        analyzer.update(history, start, metrics)

        if earlyStop.stop(history):
            break

    top1_hist = info(api, population, history, start, metrics=metrics, max_flops=max_flops, max_params=max_params)
    return top1_hist