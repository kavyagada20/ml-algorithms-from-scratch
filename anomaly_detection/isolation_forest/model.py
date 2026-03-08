"""
Isolation Forest implemented from scratch.

Author: Kavya Gada
Purpose: Detect anomalies using isolation trees.
"""

import numpy as np
import random


class IsolationTree:

    def __init__(self, height_limit):

        self.height_limit = height_limit
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.size = 0

    def fit(self, X, current_height=0):

        self.size = len(X)

        if current_height >= self.height_limit or self.size <= 1:
            return

        n_features = X.shape[1]

        self.split_feature = random.randint(0, n_features - 1)

        min_val = np.min(X[:, self.split_feature])
        max_val = np.max(X[:, self.split_feature])

        if min_val == max_val:
            return

        self.split_value = random.uniform(min_val, max_val)

        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask

        self.left = IsolationTree(self.height_limit)
        self.right = IsolationTree(self.height_limit)

        self.left.fit(X[left_mask], current_height + 1)
        self.right.fit(X[right_mask], current_height + 1)

    def path_length(self, x, current_height=0):

        if self.left is None or self.right is None:
            return current_height

        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)


class IsolationForestScratch:

    def __init__(self, n_trees=100, sample_size=64):

        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X):

        height_limit = int(np.ceil(np.log2(self.sample_size)))

        for _ in range(self.n_trees):

            sample = X[np.random.choice(len(X), self.sample_size)]

            tree = IsolationTree(height_limit)

            tree.fit(sample)

            self.trees.append(tree)

    def anomaly_score(self, x):

        path_lengths = [tree.path_length(x) for tree in self.trees]

        return np.mean(path_lengths)

    def predict(self, X):

        scores = np.array([self.anomaly_score(x) for x in X])

        threshold = np.percentile(scores, 30)

        return (scores < threshold).astype(int)