"""
AdaBoost Classifier implemented from scratch.

Author: Kavya Gada
Purpose: Understand boosting ensembles using weak learners.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoostScratch:
    """
    AdaBoost classifier using decision stumps.
    """

    def __init__(self, n_estimators=50):

        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.loss_history = []

    def fit(self, X, y):

        n_samples = X.shape[0]

        # convert labels to {-1,1}
        y = np.where(y == 0, -1, 1)

        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):

            stump = DecisionTreeClassifier(max_depth=1)

            stump.fit(X, y, sample_weight=weights)

            pred = stump.predict(X)

            error = np.sum(weights * (pred != y))

            error = max(error, 1e-10)

            alpha = 0.5 * np.log((1 - error) / error)

            weights *= np.exp(-alpha * y * pred)

            weights /= np.sum(weights)

            self.models.append(stump)
            self.alphas.append(alpha)

            self.loss_history.append(error)

    def predict(self, X):

        final_pred = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):

            final_pred += alpha * model.predict(X)

        return np.sign(final_pred)