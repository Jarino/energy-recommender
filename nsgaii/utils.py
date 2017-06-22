import random


def uniform(low, up, actual):
    size = len(actual)
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)] / sum(actual)
    except TypeError:
        return [random.uniform(a, b)
                for a, b in zip([low] * size, [up] * size)] / sum(actual)
