"""
Linear Discriminant Analysis implemented from scratch.

Author: Kavya Gada
Purpose: Supervised dimensionality reduction maximizing class separation.
"""

import numpy as np


class LDAScratch:

    def __init__(self, n_components=2):

        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):

        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within-class scatter matrix
        SW = np.zeros((n_features, n_features))

        # Between-class scatter matrix
        SB = np.zeros((n_features, n_features))

        mean_overall = np.mean(X, axis=0)

        for c in class_labels:

            X_c = X[y == c]

            mean_c = np.mean(X_c, axis=0)

            SW += (X_c - mean_c).T @ (X_c - mean_c)

            n_c = X_c.shape[0]

            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)

            SB += n_c * (mean_diff @ mean_diff.T)

        # Solve eigenvalue problem
        A = np.linalg.inv(SW).dot(SB)

        eigenvalues, eigenvectors = np.linalg.eig(A)

        idx = np.argsort(abs(eigenvalues))[::-1]

        eigenvectors = eigenvectors[:, idx]

        self.linear_discriminants = eigenvectors[:, :self.n_components]

    def transform(self, X):

        return np.dot(X, self.linear_discriminants)

    def predict(self, X):

        return self.transform(X)