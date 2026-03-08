"""
Label Propagation implemented from scratch.

Author: Kavya Gada
Purpose: Semi-supervised learning using graph-based label spreading.
"""

import numpy as np


class LabelPropagationScratch:

    def __init__(self, gamma=20, max_iter=100):

        self.gamma = gamma
        self.max_iter = max_iter
        self.loss_history = []

    def _rbf_kernel(self, X):

        sq_dists = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(X**2, axis=1)[None, :]
            - 2 * np.dot(X, X.T)
        )

        return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):

        n = X.shape[0]

        W = self._rbf_kernel(X)

        D = np.diag(np.sum(W, axis=1))

        S = np.linalg.inv(D).dot(W)

        Y = np.zeros((n, len(np.unique(y[y != -1]))))

        for i, label in enumerate(y):
            if label != -1:
                Y[i, label] = 1

        F = Y.copy()

        for _ in range(self.max_iter):

            F = S.dot(F)

            F[y != -1] = Y[y != -1]

            loss = np.linalg.norm(F - Y)

            self.loss_history.append(loss)

        self.labels_ = np.argmax(F, axis=1)

    def predict(self, X):

        return self.labels_