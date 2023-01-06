from nas_utils import *


def prune_flops_params(exemplars, history, max_flops, max_params):
    exemplars, history = dictionary_update(exemplars, history)
    history = clean_history(history, max_params, max_flops)

    # Return only exemplars with flops and params below threshold
    exemplars = [exemplar for exemplar in exemplars if
                 exemplar.get_cost_info()['flops'] <= max_flops
                 and exemplar.get_cost_info()['params'] <= max_params]
    return exemplars, history


def prune_tfm(exemplars=None, history=None, K=5, metrics=[]):
    for exemplar in exemplars:
        exemplar.born = True

    exemplars, history = dictionary_update(exemplars, history)
    if K >= len(exemplars):
        return exemplars, history

    return return_top_k(exemplars, K, metric_names=metrics), history


def prune_oldest(exemplars, K, R=1):
    # first iteration, do not perform ageing
    gen_list = [exemplar.generation for exemplar in exemplars]
    if len(set(gen_list)) == 1:
        return exemplars

    for _ in range(R):
        if len(exemplars) > K + R - 1:
            oldest = np.argmin(gen_list)
            exemplars.pop(oldest)
            gen_list = [exemplar.generation for exemplar in exemplars]
    return exemplars


def prune(exemplars, history, max_flops=0, max_params=0, K=0, metrics=[], feasibility=True):
    if feasibility:
        exemplars, history = prune_flops_params(exemplars, history, max_flops, max_params)
    exemplars = prune_oldest(exemplars, K)
    exemplars, history = prune_tfm(exemplars, history, K, metrics)
    return exemplars, history
