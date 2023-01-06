import random

from pruning import prune, prune_flops_params
from nas_utils import *
from genotypes import mutate


def search(api, N=25, n=5, P=[1], R=[1], max_flops=0.1, max_params=0.4, max_time=2, n_random=0, analyzer=None):

    """
    :param api:             Search space API
    :param N:               Size of initial random population
    :param K:               Size of surviving population at each step
    :param P:               Parallel mutations
    :param R:               Consecutive mutations
    :param max_flops:       Maximum flops
    :param max_params:      Maximum parameters
    :param steps:           Number number of steps
    :return:
    """

    start = time.time()

    metrics = ['naswot', 'logsynflow', 'skip'] # 'shiftlogsynflow',

    # Generate initial population
    exemplars = population_init(api, N, max_flops, max_params, analyzer=None)
    # Update history
    history = {exemplar.genotype: exemplar for exemplar in exemplars}
    # Prune according to feasibility, ageing and training free metrics
    exemplars, history = prune(exemplars, history, max_flops=max_flops,
                               max_params=max_params, K=N, metrics=metrics)

    # Maybe a termination condition based on relative improvement could be adopted
    total_time = 0
    step = 0
    while total_time <= max_time:

        # Keep generating exemplars until a suitable number of feasible exemplars is generated
        while True:
            # Mutation (Tournament Selection with Ageing)
            sampled = random.sample(exemplars, n)
            top = return_top_k(sampled, 2, metric_names=metrics)
            top_mutated = []
            for r, p in zip(R, P):
                top_mutated += mutate(api, top, R=r, P=p, cross=True, generation=step+1)
            exemplars += top_mutated
            # add random samples
            if n_random:
                exemplars += get_random_population(api, n_random, step+1)
            # update history
            exemplars, history = dictionary_update(exemplars, history, replace=False)
            # feasibility pruning
            exemplars, history = prune_flops_params(exemplars, history, max_flops=max_flops,
                                                    max_params=max_params)
            if len(exemplars) > N:
                break

        # ageing and training free metrics pruning
        exemplars, history = prune(exemplars, history, max_flops=max_flops, max_params=max_params,
                                   K=N, metrics=metrics, feasibility=False)
        total_time = (api.total_metrics_time(history.values(), metrics=metrics) + time.time() - start) / 60
        step += 1
        analyzer.update(history, start, metrics)

    top1_hist = info(api, exemplars, history, start, metrics=metrics, max_flops=max_flops, max_params=max_params)

    return top1_hist
