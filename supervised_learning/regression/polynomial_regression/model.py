"""
Polynomial Regression implemented from scratch using Gradient Descent.

Author: Kavya Gada
Purpose: Learn polynomial feature transformation and regression optimization.
"""

import numpy as np


class PolynomialRegression:
    """
    Polynomial Regression using Gradient Descent.

    Converts input feature into polynomial features and applies linear regression.
    """

    def __init__(self, degree=2, learning_rate=0.01, epochs=1000):
        """
        Initialize model parameters.

        Parameters
        ----------
        degree : int
            Degree of polynomial features.
        learning_rate : float
            Learning rate for gradient descent.
        epochs : int
            Number of training iterations.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None
        self.loss_history = []

    def _polynomial_features(self, X):
        """
        Generate polynomial features.

        Example:
        X -> [x]
        degree=3 -> [x, x^2, x^3]
        """
        X_poly = X
        for i in range(2, self.degree + 1):
            X_poly = np.c_[X_poly, X ** i]

        return X_poly

    def fit(self, X, y):
        """
        Train polynomial regression model using gradient descent.
        """

        X = self._polynomial_features(X)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        """
        Predict values using trained model.
        """

        X = self._polynomial_features(X)

        return np.dot(X, self.weights) + self.bias