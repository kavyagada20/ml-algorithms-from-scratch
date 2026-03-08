"""
Training script for SVR from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.svm import SVR as SKSVR
from sklearn.metrics import mean_squared_error, r2_score

from model import SVR


# Generate dataset
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=15,
    random_state=42
)

X = X.reshape(-1, 1)


# Train custom SVR
model = SVR(
    learning_rate=0.001,
    epochs=1000,
    C=1.0,
    epsilon=0.1
)

model.fit(X, y)

predictions = model.predict(X)


# Metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Custom SVR")
print("MSE:", mse)
print("R2:", r2)

with open("metrics.txt", "w") as f:

    f.write(f"MSE: {mse}\n")
    f.write(f"R2: {r2}\n")


# sklearn comparison
sk_model = SKSVR(kernel="linear")

sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

print("\nSklearn SVR MSE:",
      mean_squared_error(y, sk_pred))


# Visualization
sorted_idx = X[:, 0].argsort()

X_sorted = X[sorted_idx]
pred_sorted = predictions[sorted_idx]


plt.figure(figsize=(8,5))

plt.scatter(X, y, label="Data")

plt.plot(X_sorted, pred_sorted,
         color="red",
         label="SVR Fit")

plt.title("Support Vector Regression")

plt.xlabel("Feature")

plt.ylabel("Target")

plt.legend()

plt.savefig("svr_plot.png")

plt.show()


# Loss curve
plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("SVR Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("loss_curve.png")

plt.show()