from typing import Sequence
from nas_spaces import Exemplar


# Returns the string defining the mutated architecture
# R: consecutive mutations
def mutation(exemplar: Exemplar, R=1) -> Exemplar:
    return exemplar.space.mutation(exemplar, R)


# just two exemplars, for more use a multinomial
def crossover(exemplars: Sequence[Exemplar]) -> Exemplar:
    return exemplars[0].space.crossover(exemplars)


def mutate(api, exemplars: Sequence[Exemplar], R=1, P=1, cross=False, generation=0):
    new_exemplars = []

    for _ in range(P):
        new_exemplars += [mutation(exemplar, R=R).set_generation(generation) for exemplar in exemplars]

    # This highly enhance the search
    if cross and len(exemplars) == 2:
        for _ in range(1):
            new_exemplars.append(crossover(exemplars).set_generation(generation))

    return new_exemplars
