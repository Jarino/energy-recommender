import nsgaii.price as pr


def mape(predicted, actual):
    return sum(abs((actual - predicted) / predicted)) * 100 / len(actual)


def distance_from_total(individual, actual):
    return (sum(individual) - sum(actual))**2


def consumption_with_battery(consumption, excess_energy, battery):
    from_grid = []

    for c, e in zip(consumption, excess_energy):
        battery.charge(e)

        from_grid.append(battery.discharge(c))

    return from_grid


def cost_function(actual, pv, battery, individual):
    excess_energy = abs((individual - pv).clip(max=0))

    from_grid = (individual - pv).clip(min=0)

    if battery is not None:
        from_grid = consumption_with_battery(from_grid, excess_energy, battery)

    return (pr.calculate(from_grid),
            mape(individual, actual), distance_from_total(individual, actual))
