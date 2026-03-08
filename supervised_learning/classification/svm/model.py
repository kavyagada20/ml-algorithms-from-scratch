"""
Support Vector Machine implemented from scratch.

Author: Kavya Gada
Purpose: Understand maximum margin classification using hinge loss.
"""

import numpy as np


class SVMScratch:
    """
    Linear Support Vector Machine using gradient descent.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs

        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):

        y = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            loss_epoch = 0

            for idx, x_i in enumerate(X):

                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1

                if condition:

                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)

                else:

                    self.weights -= self.lr * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y[idx])
                    )

                    self.bias -= self.lr * y[idx]

                    loss_epoch += 1

            self.loss_history.append(loss_epoch)

    def predict(self, X):

        approx = np.dot(X, self.weights) - self.bias

        return np.sign(approx)