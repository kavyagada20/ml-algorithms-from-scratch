"""
Linear Regression implemented from scratch using Gradient Descent.

Author: Kavya Gada
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression Model using Batch Gradient Descent.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

    def predict(self, X: np.ndarray):
        return np.dot(X, self.weights) + self.bias