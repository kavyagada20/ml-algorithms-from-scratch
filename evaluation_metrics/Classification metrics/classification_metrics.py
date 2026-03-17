"""
Classification Metrics implemented from scratch.

Author: Kavya Gada
Repository: ml-algorithms-from-scratch
"""

import numpy as np


class ClassificationMetrics:

    def __init__(self):
        pass

    def confusion_matrix(self, y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }

    def accuracy(self, y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.sum(y_true == y_pred) / len(y_true)

    def precision(self, y_true, y_pred):

        cm = self.confusion_matrix(y_true, y_pred)

        return cm["TP"] / (cm["TP"] + cm["FP"] + 1e-10)

    def recall(self, y_true, y_pred):

        cm = self.confusion_matrix(y_true, y_pred)

        return cm["TP"] / (cm["TP"] + cm["FN"] + 1e-10)

    def f1_score(self, y_true, y_pred):

        precision = self.precision(y_true, y_pred)

        recall = self.recall(y_true, y_pred)

        return 2 * precision * recall / (precision + recall + 1e-10)