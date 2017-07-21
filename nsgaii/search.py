import random
import copy

from deap import tools
from deap.benchmarks.tools import hypervolume
import numpy as np


def output(src, should):
    if should:
        print(src)



def search(toolbox, seed=None, gens=500, mu=200, verbose=False):
    random.seed(seed)

    CXPB = 0.9  # pravdepodobnost krizenia?

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    # stats.register("min", np.min, axis=0)
    # stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "hypervolume"

    # vytvorime inicialnu populaciu
    pop = toolbox.population(n=mu)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Choose worst solution as reference point for hypervolume calculation
    ref = np.max([x.fitness.values for x in pop], axis=0) + 1

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    best_hypervolume = hypervolume(pop, ref)
    record['hypervolume'] = best_hypervolume
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    output(logbook.stream, verbose)

    best_hypervolume_population = pop

    # Begin the generational process
    for gen in range(1, gens):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, mu, nd='log')
        record = stats.compile(pop)

        current_hypervolume = hypervolume(pop, ref)

        record['hypervolume'] = current_hypervolume

        if current_hypervolume > best_hypervolume:
            best_hypervolume = current_hypervolume
            best_hypervolume_population = pop

        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        output(logbook.stream, verbose)


    return pop, logbook, best_hypervolume_population, best_hypervolume
