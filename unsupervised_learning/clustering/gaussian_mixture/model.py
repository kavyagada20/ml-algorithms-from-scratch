"""
Gaussian Mixture Model implemented from scratch.

Author: Kavya Gada
Purpose: Understand probabilistic clustering using EM algorithm.
"""

import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixtureScratch:
    """
    Gaussian Mixture Model using Expectation-Maximization.
    """

    def __init__(self, n_components=3, max_iter=100):

        self.n_components = n_components
        self.max_iter = max_iter

        self.means = None
        self.covariances = None
        self.weights = None
        self.loss_history = []

    def fit(self, X):

        n_samples, n_features = X.shape

        rng = np.random.default_rng()

        # initialize parameters
        self.means = X[rng.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array(
            [np.eye(n_features) for _ in range(self.n_components)]
        )
        self.weights = np.ones(self.n_components) / self.n_components

        for _ in range(self.max_iter):

            # E step
            responsibilities = np.zeros((n_samples, self.n_components))

            for k in range(self.n_components):

                pdf = multivariate_normal(
                    mean=self.means[k],
                    cov=self.covariances[k]
                ).pdf(X)

                responsibilities[:, k] = self.weights[k] * pdf

            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M step
            Nk = responsibilities.sum(axis=0)

            self.weights = Nk / n_samples

            self.means = (responsibilities.T @ X) / Nk[:, None]

            for k in range(self.n_components):

                diff = X - self.means[k]

                self.covariances[k] = (
                    responsibilities[:, k][:, None] * diff
                ).T @ diff / Nk[k]

            log_likelihood = np.sum(np.log(responsibilities.sum(axis=1)))
            self.loss_history.append(log_likelihood)

    def predict(self, X):

        responsibilities = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):

            pdf = multivariate_normal(
                mean=self.means[k],
                cov=self.covariances[k]
            ).pdf(X)

            responsibilities[:, k] = self.weights[k] * pdf

        return np.argmax(responsibilities, axis=1)