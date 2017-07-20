import matplotlib.pyplot as plt
import numpy as np

from nsgaii.utils import set_labels

class Evaluation():

    def __init__(self, pop, toolbox, logbook, actual, distance_treshold):
        self.actual = actual
        self.logbook = logbook
        self.toolbox = toolbox
        self.distance_treshold = distance_treshold

        self.objective_values = np.array([toolbox.evaluate(x) for x in pop])

        filter_mask = np.where(
            (self.objective_values[:,2] <= 0.5)
            & (self.objective_values[:,3] <= 1)
            & (self.objective_values[:,1] <= distance_treshold)
            )[0]

        self.filtered_pop = np.array(pop)[filter_mask]
            
        self.filtered_objective_values = self.objective_values[
            filter_mask
        ]

        self.n_objectives = len(self.objective_values)
        
        self.lowest_price = self.filtered_pop[self.filtered_objective_values[:,0].argmin(), : ]
        self.lowest_price_objectives = self.filtered_objective_values[self.filtered_objective_values[:,0].argmin(), : ]
        self.highest_comfort = self.filtered_pop[self.filtered_objective_values[:1].argmin(), : ]
        self.highest_comfort_objectives = self.filtered_objective_values[self.filtered_objective_values[:1].argmin(), : ]

    def __str__(self):
        return '\n'.join([
            'Number of solutions: %d' % len(self.filtered_pop),
            'Lowest price solution: cost: %f, distance %f, diff %f' % (
                self.lowest_price_objectives[0], self.lowest_price_objectives[1], self.lowest_price_objectives[2]
            ),
            'Highest comfort solution: cost: %f, distance %f, diff %f' % (
                self.highest_comfort_objectives[0], self.highest_comfort_objectives[1], self.highest_comfort_objectives[2]
            )
        ])


    def plot(self, figsize=None):
        if figsize is None:
            figsize = (10,8)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=figsize)

        self.plot_all_consumptions(ax1)
        
        self.plot_marginal_consumptions(ax2)

        self.plot_objective_values(ax3)

        self.plot_hypervolume(ax4)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        return fig, ((ax1, ax2), (ax3, ax4))


    def plot_all_consumptions(self, ax):
        for solution in self.filtered_pop:
            ax.plot(solution)
        set_labels(ax)
        ax.set_title('All filtered solutions')
    

    def plot_marginal_consumptions(self, ax):
        ax.set_title('Marginal consumptions')
        ax.plot(self.lowest_price, label='Lowest price')
        ax.plot(self.highest_comfort, label='Highest comfort')
        ax.plot(self.actual, label='Actual consmuption')
        set_labels(ax)
        ax.legend(loc=2, prop={'size': 10})


    def plot_objective_values(self, ax):
        ax.set_title('Distribution of solutions')
        ax.scatter(self.filtered_objective_values[:,0],self.filtered_objective_values[:,1])
        ax.set_xlabel('Cost of electricity $[\$]$')
        ax.set_ylabel('Distance')
        ax.set_ylim(0, self.distance_treshold)

    def plot_hypervolume(self, ax):
        ax.set_title('Hypervolume')
        ax.plot(self.logbook.select('hypervolume'))
        ax.set_xlabel('Generations')
        ax.set_ylabel('Hypervolume')
