"""
KMeans Clustering implemented from scratch.

Author: Kavya Gada
Purpose: Understand clustering algorithms and centroid optimization.
"""

import numpy as np


class KMeans:

    def __init__(self, k=3, max_iters=100):
        """
        Parameters
        ----------
        k : number of clusters
        max_iters : number of iterations
        """

        self.k = k
        self.max_iters = max_iters

        self.centroids = None
        self.loss_history = []

    def fit(self, X):

        n_samples, n_features = X.shape

        # randomly initialize centroids
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):

            clusters = self._assign_clusters(X)

            new_centroids = self._compute_centroids(X, clusters)

            loss = self._compute_loss(X, clusters)
            self.loss_history.append(loss)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):

        return self._assign_clusters(X)

    def _assign_clusters(self, X):

        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, clusters):

        centroids = []

        for i in range(self.k):

            cluster_points = X[clusters == i]

            centroid = np.mean(cluster_points, axis=0)

            centroids.append(centroid)

        return np.array(centroids)

    def _compute_loss(self, X, clusters):

        loss = 0

        for i in range(self.k):

            cluster_points = X[clusters == i]

            centroid = self.centroids[i]

            loss += np.sum((cluster_points - centroid) ** 2)

        return loss