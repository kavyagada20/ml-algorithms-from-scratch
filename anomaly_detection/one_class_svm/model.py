"""
One-Class SVM implemented from scratch.

Author: Kavya Gada
Purpose: Detect anomalies using a hypersphere boundary.
"""

import numpy as np


class OneClassSVMScratch:

    def __init__(self, nu=0.1, lr=0.001, epochs=500):

        self.nu = nu
        self.lr = lr
        self.epochs = epochs

        self.w = None
        self.rho = 0

        self.loss_history = []

    def fit(self, X):

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)

        for _ in range(self.epochs):

            scores = X.dot(self.w) - self.rho

            loss = np.mean(np.maximum(0, scores))

            self.loss_history.append(loss)

            grad_w = self.w - self.nu * np.mean(
                X[scores > 0], axis=0
            )

            grad_rho = -self.nu * np.mean(scores > 0)

            self.w -= self.lr * grad_w
            self.rho -= self.lr * grad_rho

    def decision_function(self, X):

        return X.dot(self.w) - self.rho

    def predict(self, X):

        scores = self.decision_function(X)

        return (scores < 0).astype(int)