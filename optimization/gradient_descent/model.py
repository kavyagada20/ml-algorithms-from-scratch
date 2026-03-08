"""
Gradient Descent optimization implemented from scratch.

Author: Kavya Gada
Purpose: Demonstrate optimization of a linear regression model using gradient descent.
"""

import numpy as np


class GradientDescentScratch:

    def __init__(self, lr=0.01, epochs=1000):

        self.lr = lr
        self.epochs = epochs

        self.weights = None
        self.bias = None

        self.loss_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = X.dot(self.weights) + self.bias

            error = y_pred - y

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):

        return X.dot(self.weights) + self.bias