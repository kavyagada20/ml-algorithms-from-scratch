"""
Random Forest Classifier implemented from scratch.

Author: Kavya Gada
Purpose: Understand ensemble learning using decision trees and bagging.
"""

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class RandomForestScratch:
    """
    Random Forest classifier using bagging of decision trees.
    """

    def __init__(self, n_estimators=20, max_depth=5):

        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.trees = []
        self.loss_history = []

    def _bootstrap_sample(self, X, y):

        n_samples = X.shape[0]

        idx = np.random.choice(n_samples, n_samples, replace=True)

        return X[idx], y[idx]

    def fit(self, X, y):

        self.trees = []

        for _ in range(self.n_estimators):

            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree = DecisionTreeClassifier(max_depth=self.max_depth)

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def predict(self, X):

        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        tree_preds = np.swapaxes(tree_preds, 0, 1)

        predictions = []

        for preds in tree_preds:

            predictions.append(Counter(preds).most_common(1)[0][0])

        return np.array(predictions)