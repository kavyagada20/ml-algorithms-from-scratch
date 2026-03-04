"""
Training script for Polynomial Regression from scratch.
Includes visualization, evaluation metrics and sklearn comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from model import PolynomialRegression


# -----------------------------
# 1. Generate dataset
# -----------------------------

X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=15,
    random_state=42
)

# make dataset non-linear
y = y + 0.5 * (X[:, 0] ** 2)

X = X.reshape(-1, 1)


# -----------------------------
# 2. Train custom model
# -----------------------------

model = PolynomialRegression(degree=2, learning_rate=0.01, epochs=1000)
model.fit(X, y)

predictions = model.predict(X)


# -----------------------------
# 3. Evaluation metrics
# -----------------------------

mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Custom Polynomial Regression")
print("MSE:", mse)
print("R2 Score:", r2)

with open("metrics.txt", "w") as f:
    f.write(f"MSE: {mse}\n")
    f.write(f"R2 Score: {r2}\n")


# -----------------------------
# 4. Sklearn comparison
# -----------------------------

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

sk_model = LinearRegression()
sk_model.fit(X_poly, y)

sk_predictions = sk_model.predict(X_poly)

sk_mse = mean_squared_error(y, sk_predictions)

print("\nSklearn Polynomial Regression")
print("MSE:", sk_mse)


# -----------------------------
# 5. Sort for smooth curve
# -----------------------------

sorted_index = X[:, 0].argsort()
X_sorted = X[sorted_index]
pred_sorted = predictions[sorted_index]


# -----------------------------
# 6. Plot regression curve
# -----------------------------

plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Data")
plt.plot(X_sorted, pred_sorted, color="red", label="Polynomial Fit")

plt.title("Polynomial Regression (From Scratch)")
plt.xlabel("Feature")
plt.ylabel("Target")

plt.legend()
plt.savefig("regression_plot.png")
plt.show()


# -----------------------------
# 7. Loss curve
# -----------------------------

plt.figure(figsize=(8,5))
plt.plot(model.loss_history)

plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.savefig("loss_curve.png")
plt.show()