"""
Training script for Gradient Boosting Regressor from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from model import GradientBoostingRegressorScratch


# Generate dataset
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

X = X.reshape(-1,1)


# Train custom model
model = GradientBoostingRegressorScratch(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3
)

model.fit(X, y)

predictions = model.predict(X)


# Metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Custom Gradient Boosting")
print("MSE:", mse)
print("R2:", r2)

with open("metrics.txt","w") as f:
    f.write(f"MSE: {mse}\n")
    f.write(f"R2: {r2}\n")


# Sklearn comparison
sk_model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3
)

sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

print("\nSklearn MSE:",
      mean_squared_error(y, sk_pred))


# Visualization
sorted_idx = X[:,0].argsort()

X_sorted = X[sorted_idx]
pred_sorted = predictions[sorted_idx]

plt.figure(figsize=(8,5))

plt.scatter(X,y,label="Data")

plt.plot(X_sorted,pred_sorted,
         color="red",
         label="Gradient Boosting Fit")

plt.title("Gradient Boosting Regressor")

plt.xlabel("Feature")

plt.ylabel("Target")

plt.legend()

plt.savefig("gradient_boosting_plot.png")

plt.show()


# Loss curve
plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("Training Loss")

plt.xlabel("Iteration")

plt.ylabel("MSE")

plt.savefig("loss_curve.png")

plt.show()