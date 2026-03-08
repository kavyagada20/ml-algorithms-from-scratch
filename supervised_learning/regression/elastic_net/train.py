"""
Training script for Elastic Net Regression from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

from model import ElasticNetRegression


# Generate dataset
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

X = X.reshape(-1,1)


# Train custom model
model = ElasticNetRegression(
    alpha=0.5,
    l1_ratio=0.5,
    learning_rate=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)


# Evaluation
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Custom Elastic Net")
print("MSE:", mse)
print("R2:", r2)


with open("metrics.txt","w") as f:
    f.write(f"MSE: {mse}\n")
    f.write(f"R2: {r2}\n")


# sklearn comparison
sk_model = ElasticNet(alpha=0.5, l1_ratio=0.5)

sk_model.fit(X,y)

sk_pred = sk_model.predict(X)

print("\nSklearn ElasticNet MSE:",
      mean_squared_error(y, sk_pred))


# Visualization
sorted_idx = X[:,0].argsort()

X_sorted = X[sorted_idx]
pred_sorted = predictions[sorted_idx]

plt.figure(figsize=(8,5))

plt.scatter(X,y,label="Data")

plt.plot(X_sorted,pred_sorted,
         color="red",
         label="ElasticNet Fit")

plt.title("Elastic Net Regression")

plt.xlabel("Feature")

plt.ylabel("Target")

plt.legend()

plt.savefig("elastic_net_plot.png")

plt.show()


# Loss curve
plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("loss_curve.png")

plt.show()