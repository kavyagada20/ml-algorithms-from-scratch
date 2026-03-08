"""
Training script for Random Forest Regressor from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from model import RandomForestRegressorScratch


# Generate dataset
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

X = X.reshape(-1, 1)


# Train custom model
model = RandomForestRegressorScratch(
    n_trees=20,
    max_depth=5
)

model.fit(X, y)

predictions = model.predict(X)


# Evaluation metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Custom Random Forest")
print("MSE:", mse)
print("R2:", r2)


with open("metrics.txt", "w") as f:

    f.write(f"MSE: {mse}\n")
    f.write(f"R2: {r2}\n")


# Sklearn comparison
sk_model = RandomForestRegressor(
    n_estimators=20,
    max_depth=5,
    random_state=42
)

sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

print("\nSklearn Random Forest MSE:",
      mean_squared_error(y, sk_pred))


# Sort for visualization
sorted_idx = X[:, 0].argsort()

X_sorted = X[sorted_idx]
pred_sorted = predictions[sorted_idx]


# Plot
plt.figure(figsize=(8,5))

plt.scatter(X, y, label="Data")

plt.plot(X_sorted, pred_sorted,
         color="red",
         label="Random Forest Fit")

plt.title("Random Forest Regressor")

plt.xlabel("Feature")

plt.ylabel("Target")

plt.legend()

plt.savefig("random_forest_plot.png")

plt.show()