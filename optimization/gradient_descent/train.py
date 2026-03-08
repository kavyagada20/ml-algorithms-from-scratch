"""
Training script for Gradient Descent optimization
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from model import GradientDescentScratch


# Generate dataset
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=10,
    random_state=42
)


# Train custom model
model = GradientDescentScratch(lr=0.01, epochs=500)

model.fit(X, y)

pred = model.predict(X)

print("Custom Gradient Descent completed")


# sklearn comparison
sk_model = LinearRegression()

sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

print("Sklearn Linear Regression completed")


# Evaluation metrics
mse_custom = mean_squared_error(y, pred)
mse_sklearn = mean_squared_error(y, sk_pred)

print("Custom MSE:", mse_custom)
print("Sklearn MSE:", mse_sklearn)


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("Gradient Descent Optimization\n")
    f.write(f"Custom MSE: {mse_custom}\n")
    f.write(f"Sklearn MSE: {mse_sklearn}\n")


# Visualization (Regression Fit)
plt.figure(figsize=(6,5))

plt.scatter(X, y)

plt.plot(X, pred, color="red")

plt.title("Gradient Descent Regression")

plt.savefig("regression_plot.png")

plt.show()


# Loss Curve
plt.figure(figsize=(6,5))

plt.plot(model.loss_history)

plt.title("Gradient Descent Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.savefig("loss_curve.png")

plt.show()