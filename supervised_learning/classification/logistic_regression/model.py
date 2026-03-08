"""
Logistic Regression implemented from scratch.

Author: Kavya Gada
Purpose: Understand probabilistic classification using sigmoid and gradient descent.
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Logistic Regression classifier using gradient descent.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            linear_model = np.dot(X, self.weights) + self.bias

            y_pred = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = -np.mean(
                y * np.log(y_pred + 1e-10) +
                (1 - y) * np.log(1 - y_pred + 1e-10)
            )

            self.loss_history.append(loss)

    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        y_pred = self._sigmoid(linear_model)

        return np.where(y_pred >= 0.5, 1, 0)