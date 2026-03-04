"""
Ridge Regression implemented from scratch using Gradient Descent
"""

import numpy as np


class RidgeRegression:

    def __init__(self, alpha=1.0, learning_rate=0.01, epochs=1000):

        self.alpha = alpha
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

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * self.weights
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean((y - y_pred) ** 2)
            ridge_penalty = self.alpha * np.sum(self.weights ** 2)

            loss = mse + ridge_penalty
            self.loss_history.append(loss)

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias