from deap import base
from deap import creator
from deap import tools

import nsgaii.utils as utl
from nsgaii.objectives import cost_function_with_euclidean
from nsgaii.objectives import cost_function_with_mape

SUPPORTED_DISTANCE_METRICS = ['mape', 'euclidean']

def setup(actual, pv, ndim, bound_low, bound_up, battery=None, distance='mape'):
    """
    creates toolbox with neccessary genetic operators
    """
    if distance not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError('%s is not supported distance metric' % distance)


    toolbox = base.Toolbox()
    toolbox.register('attr_float', utl.uniform, 0, 10, actual)
    toolbox.register('individual', tools.initIterate,
                     creator.Individual, toolbox.attr_float)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    if distance == 'mape':
        toolbox.register('evaluate', cost_function_with_mape, actual, pv, battery)
    if distance == 'euclidean':
        toolbox.register('evaluate', cost_function_with_euclidean, actual, pv, battery)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=bound_low, up=bound_up, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=bound_low, up=bound_up, eta=20.0, indpb=1.0 / ndim)
    toolbox.register("select", tools.selNSGA2)

    return toolbox
