"""
Random Forest Regressor implemented from scratch.

Author: Kavya Gada
Purpose: Understand ensemble learning using decision trees.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RandomForestRegressorScratch:
    """
    Random Forest Regressor using bootstrap aggregation.
    """

    def __init__(self, n_trees=10, max_depth=5, random_state=42):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state

        self.trees = []
        self.loss_history = []

    def _bootstrap_sample(self, X, y):

        n_samples = X.shape[0]

        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs], y[idxs]

    def fit(self, X, y):

        self.trees = []

        for _ in range(self.n_trees):

            tree = DecisionTreeRegressor(max_depth=self.max_depth)

            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def predict(self, X):

        predictions = np.array([tree.predict(X) for tree in self.trees])

        return np.mean(predictions, axis=0)