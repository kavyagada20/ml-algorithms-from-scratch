"""
Training script for Ridge Regression from scratch.

Includes:
- Dataset generation
- Model training
- Evaluation metrics
- Sklearn comparison
- Visualization
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from model import RidgeRegression


# -----------------------------
# 1 Generate dataset
# -----------------------------

X, y = make_regression(
    n_samples=120,
    n_features=1,
    noise=25,
    random_state=42
)

X = X.reshape(-1, 1)


# -----------------------------
# 2 Train custom ridge model
# -----------------------------

model = RidgeRegression(alpha=1.0, learning_rate=0.01, epochs=1000)
model.fit(X, y)

predictions = model.predict(X)


# -----------------------------
# 3 Evaluation metrics
# -----------------------------

mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Custom Ridge Regression")
print("MSE:", mse)
print("R2:", r2)

with open("metrics.txt", "w") as f:
    f.write(f"MSE: {mse}\n")
    f.write(f"R2 Score: {r2}\n")


# -----------------------------
# 4 Sklearn comparison
# -----------------------------

sk_model = Ridge(alpha=1.0)
sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

sk_mse = mean_squared_error(y, sk_pred)

print("\nSklearn Ridge Regression")
print("MSE:", sk_mse)


# -----------------------------
# 5 Sort for plotting
# -----------------------------

sorted_idx = X[:, 0].argsort()

X_sorted = X[sorted_idx]
pred_sorted = predictions[sorted_idx]


# -----------------------------
# 6 Regression plot
# -----------------------------

plt.figure(figsize=(8,5))

plt.scatter(X, y, label="Data")
plt.plot(X_sorted, pred_sorted, color="red", label="Ridge Fit")

plt.title("Ridge Regression (From Scratch)")
plt.xlabel("Feature")
plt.ylabel("Target")

plt.legend()

plt.savefig("ridge_regression_plot.png")
plt.show()


# -----------------------------
# 7 Loss curve
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("loss_curve.png")
plt.show()