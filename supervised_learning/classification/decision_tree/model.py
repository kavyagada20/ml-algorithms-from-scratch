"""
Decision Tree Classifier implemented from scratch.

Author: Kavya Gada
Purpose: Understand recursive tree splitting using Gini impurity.
"""

import numpy as np


class Node:
    """
    Node class used to build the decision tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeScratch:
    """
    Decision Tree Classifier using Gini impurity.
    """

    def __init__(self, max_depth=5):

        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):

        self.root = self._grow_tree(X, y)

    def predict(self, X):

        return np.array([
            self._traverse_tree(x, self.root) for x in X
        ])

    def _gini(self, y):

        classes = np.unique(y)

        gini = 1

        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p**2

        return gini

    def _best_split(self, X, y):

        best_feature = None
        best_threshold = None
        best_gain = -1

        parent_gini = self._gini(y)

        n_features = X.shape[1]

        for feature in range(n_features):

            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:

                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                left_gini = self._gini(y[left_idx])
                right_gini = self._gini(y[right_idx])

                n = len(y)
                n_left = len(y[left_idx])
                n_right = len(y[right_idx])

                weighted_gini = (
                    n_left/n * left_gini +
                    n_right/n * right_gini
                )

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):

        num_samples = len(y)
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or num_classes == 1:

            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left, right)

    def _traverse_tree(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)