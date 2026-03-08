"""
Support Vector Regression implemented from scratch.

Author: Kavya Gada
Purpose: Understand margin-based regression using epsilon-insensitive loss.
"""

import numpy as np


class SVR:
    """
    Support Vector Regression using gradient descent.
    """

    def __init__(self, learning_rate=0.001, epochs=1000, C=1.0, epsilon=0.1):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.epsilon = epsilon

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

            dw = np.zeros(n_features)
            db = 0

            for i in range(n_samples):

                if abs(errors[i]) <= self.epsilon:
                    continue

                gradient = np.sign(errors[i])

                dw += gradient * X[i]
                db += gradient

            dw = (self.weights + self.C * dw) / n_samples
            db = self.C * db / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(np.maximum(0, np.abs(errors) - self.epsilon))
            self.loss_history.append(loss)

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias