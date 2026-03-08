"""
Mean Shift Clustering implemented from scratch.

Author: Kavya Gada
Purpose: Understand density-based clustering using mean shift algorithm.
"""

import numpy as np


class MeanShiftScratch:
    """
    Mean Shift clustering using iterative centroid shifting.
    """

    def __init__(self, bandwidth=2.0, max_iter=100):

        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None

    def fit(self, X):

        centroids = X.copy()

        for _ in range(self.max_iter):

            new_centroids = []

            for centroid in centroids:

                distances = np.linalg.norm(X - centroid, axis=1)

                points = X[distances < self.bandwidth]

                if len(points) == 0:
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(np.mean(points, axis=0))

            centroids = np.unique(np.round(new_centroids, decimals=2), axis=0)

        self.centroids = centroids

        labels = []

        for x in X:

            distances = np.linalg.norm(self.centroids - x, axis=1)

            labels.append(np.argmin(distances))

        self.labels_ = np.array(labels)

        return self

    def predict(self, X):

        labels = []

        for x in X:

            distances = np.linalg.norm(self.centroids - x, axis=1)

            labels.append(np.argmin(distances))

        return np.array(labels)