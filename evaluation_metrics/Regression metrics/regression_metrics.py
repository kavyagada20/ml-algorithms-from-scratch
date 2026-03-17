"""
Regression Metrics implemented from scratch.

Author: Kavya Gada
Repository: ml-algorithms-from-scratch
"""

import numpy as np


class RegressionMetrics:

    def __init__(self):
        pass

    def mae(self, y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.mean(np.abs(y_true - y_pred))

    def mse(self, y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.mean((y_true - y_pred) ** 2)

    def rmse(self, y_true, y_pred):

        return np.sqrt(self.mse(y_true, y_pred))

    def r2_score(self, y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

        ss_residual = np.sum((y_true - y_pred) ** 2)

        return 1 - (ss_residual / ss_total)