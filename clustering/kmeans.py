"""
Slightly modified implementation of k-means clustering algorithm taken from:
http://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb

Reason for not using the scikit-learn k means is missing DTW distance metric.
"""
import random

import numpy as np

from clustering.metrics import LB_Keogh
from clustering.metrics import DTWDistance

def k_means_clust(data, num_clust, num_iter, w=5):
    """
    K means clustering using DTW distance
    """
    centroids = np.array(random.sample(data.tolist(), num_clust))

    for counter in range(num_iter):
        print(counter, end=', ')
        assignments = {}

        #assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None

            for c_ind, j in enumerate(centroids):
                if LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = DTWDistance(i, j, w)

                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind

            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
                assignments[closest_clust].append(ind)
        
        #recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0
        
            # if cluster is empty, assign random point and carry on
            if len(assignments[key]) == 0:
                assignments[key].append(random.choice(range(0, len(data))))
                print(' empty cluster ', end='')

            for k in assignments[key]:
                clust_sum = clust_sum + data[k]

            centroids[key] = [m/len(assignments[key]) for m in clust_sum]
            
    print('\n')
    return centroids, assignments

