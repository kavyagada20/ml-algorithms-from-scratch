"""
Self-Training Semi-Supervised Learning implemented from scratch.

Author: Kavya Gada
Purpose: Iteratively label unlabeled data using a base classifier.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class SelfTrainingScratch:

    def __init__(self, threshold=0.8, max_iter=10):

        self.threshold = threshold
        self.max_iter = max_iter

        self.model = LogisticRegression()

        self.loss_history = []

    def fit(self, X, y):

        X_labeled = X[y != -1]
        y_labeled = y[y != -1]

        X_unlabeled = X[y == -1]

        for _ in range(self.max_iter):

            if len(X_unlabeled) == 0:
                break

            self.model.fit(X_labeled, y_labeled)

            probs = self.model.predict_proba(X_unlabeled)

            confidence = np.max(probs, axis=1)

            labels = np.argmax(probs, axis=1)

            mask = confidence > self.threshold

            if np.sum(mask) == 0:
                break

            X_labeled = np.vstack([X_labeled, X_unlabeled[mask]])
            y_labeled = np.concatenate([y_labeled, labels[mask]])

            X_unlabeled = X_unlabeled[~mask]

            loss = 1 - np.mean(confidence[mask])
            self.loss_history.append(loss)

        self.model.fit(X_labeled, y_labeled)

    def predict(self, X):

        return self.model.predict(X)