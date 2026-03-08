"""
Principal Component Analysis implemented from scratch.

Author: Kavya Gada
Purpose: Understand dimensionality reduction using eigen decomposition.
"""

import numpy as np


class PCAScratch:
    """
    PCA implementation using covariance matrix eigen decomposition.
    """

    def __init__(self, n_components=2):

        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):

        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):

        X_centered = X - self.mean

        return np.dot(X_centered, self.components)

    def predict(self, X):

        return self.transform(X)