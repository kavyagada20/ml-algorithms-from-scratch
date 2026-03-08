"""
Local Outlier Factor implemented from scratch.

Author: Kavya Gada
Purpose: Detect anomalies based on local density deviation.
"""

import numpy as np


class LocalOutlierFactorScratch:

    def __init__(self, k=10):

        self.k = k

    def _pairwise_distances(self, X):

        sum_X = np.sum(X**2, axis=1)

        distances = (
            -2 * np.dot(X, X.T)
            + sum_X[:, None]
            + sum_X[None, :]
        )

        return np.sqrt(distances)

    def fit(self, X):

        self.X = X

        distances = self._pairwise_distances(X)

        self.distances = distances

        self.neighbors = np.argsort(distances, axis=1)[:, 1:self.k+1]

    def _local_reachability_density(self, i):

        reach_dists = []

        for j in self.neighbors[i]:

            dist = self.distances[i][j]

            k_dist = self.distances[j][self.neighbors[j][-1]]

            reach_dists.append(max(dist, k_dist))

        return len(reach_dists) / np.sum(reach_dists)

    def predict(self, X):

        lrd = np.array([self._local_reachability_density(i) for i in range(len(self.X))])

        lof_scores = []

        for i in range(len(self.X)):

            ratios = []

            for j in self.neighbors[i]:
                ratios.append(lrd[j] / lrd[i])

            lof_scores.append(np.mean(ratios))

        lof_scores = np.array(lof_scores)

        threshold = np.percentile(lof_scores, 90)

        return (lof_scores > threshold).astype(int)