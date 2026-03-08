"""
t-SNE implemented from scratch.

Author: Kavya Gada
Purpose: Nonlinear dimensionality reduction preserving local similarities.
"""

import numpy as np


class TSNEScratch:

    def __init__(self, n_components=2, perplexity=30, lr=200, epochs=500):

        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.epochs = epochs

        self.loss_history = []

    def _pairwise_distances(self, X):

        sum_X = np.sum(X**2, axis=1)

        dists = (
            -2 * np.dot(X, X.T)
            + sum_X[:, None]
            + sum_X[None, :]
        )

        return dists

    def _compute_p(self, X):

        distances = self._pairwise_distances(X)

        P = np.exp(-distances / np.std(distances))

        np.fill_diagonal(P, 0)

        P /= np.sum(P)

        return P

    def fit(self, X):

        n = X.shape[0]

        P = self._compute_p(X)

        Y = np.random.randn(n, self.n_components)

        for epoch in range(self.epochs):

            distances = self._pairwise_distances(Y)

            Q = 1 / (1 + distances)

            np.fill_diagonal(Q, 0)

            Q /= np.sum(Q)

            gradient = np.zeros_like(Y)

            for i in range(n):

                diff = Y[i] - Y

                gradient[i] = np.sum(
                    (P[i] - Q[i])[:, None] * diff,
                    axis=0
                )

            Y += self.lr * gradient

            loss = np.sum(P * np.log((P + 1e-8) / (Q + 1e-8)))

            self.loss_history.append(loss)

        self.embedding = Y

    def predict(self, X):

        return self.embedding