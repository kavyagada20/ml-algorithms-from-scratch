"""
Hierarchical Clustering (Agglomerative) implemented from scratch.

Author: Kavya Gada
Purpose: Understand bottom-up clustering using distance merging.
"""

import numpy as np


class HierarchicalClusteringScratch:
    """
    Agglomerative hierarchical clustering using single linkage.
    """

    def __init__(self, n_clusters=3):

        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):

        n_samples = X.shape[0]

        clusters = [[i] for i in range(n_samples)]

        distance_matrix = self._compute_distance_matrix(X)

        while len(clusters) > self.n_clusters:

            min_dist = float("inf")
            pair = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):

                    dist = self._cluster_distance(
                        clusters[i],
                        clusters[j],
                        distance_matrix
                    )

                    if dist < min_dist:
                        min_dist = dist
                        pair = (i, j)

            clusters[pair[0]] = clusters[pair[0]] + clusters[pair[1]]

            clusters.pop(pair[1])

        labels = np.zeros(n_samples)

        for cluster_id, cluster in enumerate(clusters):

            for sample in cluster:
                labels[sample] = cluster_id

        self.labels_ = labels

        return self

    def predict(self, X):

        return self.labels_

    def _compute_distance_matrix(self, X):

        n = X.shape[0]

        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):

                dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])

        return dist_matrix

    def _cluster_distance(self, cluster1, cluster2, dist_matrix):

        min_dist = float("inf")

        for i in cluster1:
            for j in cluster2:

                dist = dist_matrix[i, j]

                if dist < min_dist:
                    min_dist = dist

        return min_dist