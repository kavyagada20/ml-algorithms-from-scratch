"""
Kernel PCA implemented from scratch.

Author: Kavya Gada
Purpose: Non-linear dimensionality reduction using kernel trick.
"""

import numpy as np


class KernelPCAScratch:
    """
    Kernel PCA using RBF kernel.
    """

    def __init__(self, n_components=2, gamma=15):

        self.n_components = n_components
        self.gamma = gamma

        self.alphas = None
        self.lambdas = None
        self.X_fit = None

    def _rbf_kernel(self, X, Y=None):

        if Y is None:
            Y = X

        sq_dists = np.sum(X**2, axis=1).reshape(-1,1) + \
                   np.sum(Y**2, axis=1) - \
                   2*np.dot(X, Y.T)

        return np.exp(-self.gamma * sq_dists)

    def fit(self, X):

        self.X_fit = X

        K = self._rbf_kernel(X)

        n = K.shape[0]

        one_n = np.ones((n,n)) / n

        K_centered = K - one_n@K - K@one_n + one_n@K@one_n

        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        self.lambdas = eigenvalues[:self.n_components]
        self.alphas = eigenvectors[:,:self.n_components]

    def transform(self, X):

        K = self._rbf_kernel(X, self.X_fit)

        return np.dot(K, self.alphas)

    def predict(self, X):

        return self.transform(X)