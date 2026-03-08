"""
Elastic Net Regression implemented from scratch.

Author: Kavya Gada
Purpose: Combine L1 and L2 regularization for regression.
"""

import numpy as np


class ElasticNetRegression:
    """
    Elastic Net regression using gradient descent.
    Combines Ridge (L2) and Lasso (L1) regularization.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, epochs=1000):

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = np.dot(X, self.weights) + self.bias

            errors = y_pred - y

            # gradients
            dw = (1/n_samples) * np.dot(X.T, errors)
            db = (1/n_samples) * np.sum(errors)

            # L1 regularization
            l1_grad = self.l1_ratio * np.sign(self.weights)

            # L2 regularization
            l2_grad = (1 - self.l1_ratio) * self.weights

            dw += self.alpha * (l1_grad + l2_grad)

            # update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # loss
            mse = np.mean(errors**2)

            l1_penalty = np.sum(np.abs(self.weights))
            l2_penalty = np.sum(self.weights**2)

            loss = mse + self.alpha * (
                self.l1_ratio * l1_penalty +
                (1-self.l1_ratio) * l2_penalty
            )

            self.loss_history.append(loss)

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias