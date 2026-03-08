"""
K-Nearest Neighbors Classifier implemented from scratch.

Author: Kavya Gada
Purpose: Understand instance-based learning and distance metrics.
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors classifier using Euclidean distance.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """
        Store training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for input samples.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """
        Predict a single sample.
        """

        distances = np.linalg.norm(self.X_train - x, axis=1)

        k_indices = np.argsort(distances)[:self.k]

        k_labels = self.y_train[k_indices]

        most_common = Counter(k_labels).most_common(1)

        return most_common[0][0]