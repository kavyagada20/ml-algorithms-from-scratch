"""
DBSCAN implemented from scratch.

Author: Kavya Gada
Purpose: Understand density-based clustering using DBSCAN.
"""

import numpy as np


class DBSCANScratch:
    """
    Density-Based Spatial Clustering of Applications with Noise.
    """

    def __init__(self, eps=0.5, min_samples=5):

        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):

        n = X.shape[0]

        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)

        cluster_id = 0

        for i in range(n):

            if visited[i]:
                continue

            visited[i] = True

            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:

                labels[i] = -1

            else:

                self._expand_cluster(
                    X,
                    labels,
                    i,
                    neighbors,
                    cluster_id,
                    visited
                )

                cluster_id += 1

        self.labels_ = labels

        return self

    def predict(self, X):

        return self.labels_

    def _region_query(self, X, point_idx):

        neighbors = []

        for i in range(len(X)):

            dist = np.linalg.norm(X[point_idx] - X[i])

            if dist < self.eps:
                neighbors.append(i)

        return neighbors

    def _expand_cluster(
        self,
        X,
        labels,
        point_idx,
        neighbors,
        cluster_id,
        visited
    ):

        labels[point_idx] = cluster_id

        i = 0

        while i < len(neighbors):

            neighbor = neighbors[i]

            if not visited[neighbor]:

                visited[neighbor] = True

                neighbor_neighbors = self._region_query(X, neighbor)

                if len(neighbor_neighbors) >= self.min_samples:

                    neighbors += neighbor_neighbors

            if labels[neighbor] == -1:

                labels[neighbor] = cluster_id

            i += 1