"""
Gradient Boosting Regressor implemented from scratch.

Author: Kavya Gada
Purpose: Understand boosting ensembles using decision trees.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressorScratch:
    """
    Gradient Boosting Regressor using decision trees as weak learners.
    """

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.trees = []
        self.loss_history = []

    def fit(self, X, y):

        # Initial prediction
        self.initial_prediction = np.mean(y)

        y_pred = np.full_like(y, self.initial_prediction)

        for _ in range(self.n_estimators):

            residuals = y - y_pred

            tree = DecisionTreeRegressor(max_depth=self.max_depth)

            tree.fit(X, residuals)

            update = tree.predict(X)

            y_pred += self.learning_rate * update

            self.trees.append(tree)

            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):

        y_pred = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:

            y_pred += self.learning_rate * tree.predict(X)

        return y_pred