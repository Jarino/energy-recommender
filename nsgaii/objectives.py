import nsgaii.price as pr


def mape(predicted, actual):
    return sum(abs((actual - predicted) / predicted)) * 100 / len(actual)


def distance_from_total(individual, actual):
    return (sum(individual) - sum(actual))**2


def cost_function(actual, pv, individual):
    cons_with_pv = (individual - pv).clip(min=0)
    return (pr.calculate(cons_with_pv),
            mape(individual, actual), distance_from_total(individual, actual))
