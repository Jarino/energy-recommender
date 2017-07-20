import random

import matplotlib.pyplot as plt
import numpy as np


def uniform(low, up, actual):
    size = len(actual)
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)] / sum(actual)
    except TypeError:
        return [random.uniform(a, b)
                for a, b in zip([low] * size, [up] * size)] / sum(actual)

def get_results(pop, toolbox):
    """
    Filter out results, that deviates more than 0.5kWh in total consumption
    """
    results = np.array([toolbox.evaluate(x) for x in pop])

    x = np.hstack([results, pop])
    filtered = x
    filtered = x[np.where(x[:,2] <= 0.5)[0]]
    
    return filtered


def set_labels(ax):
    ax.set_xlabel('Hour of a day $[h]$')
    ax.set_ylabel('Consumption $[kWh]$')
    ax.set_ylim(0,5)
    ax.set_xlim(0,24)