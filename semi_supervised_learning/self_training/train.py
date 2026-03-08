"""
Training script for Self-Training Semi-Supervised Learning
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model import SelfTrainingScratch


# Generate dataset
X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    random_state=42
)

# Hide most labels
y = y_true.copy()

mask = np.random.rand(len(y)) < 0.8

y[mask] = -1


# Train custom model
model = SelfTrainingScratch()

model.fit(X, y)

pred = model.predict(X)

print("Custom Self Training completed")


# sklearn comparison
sk_model = SelfTrainingClassifier(LogisticRegression())

sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

print("Sklearn Self Training completed")


# Evaluation metrics
acc_custom = accuracy_score(y_true, pred)
acc_sklearn = accuracy_score(y_true, sk_pred)

print("Custom Accuracy:", acc_custom)
print("Sklearn Accuracy:", acc_sklearn)


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("Self Training Semi-Supervised Learning\n")
    f.write(f"Custom Accuracy: {acc_custom}\n")
    f.write(f"Sklearn Accuracy: {acc_sklearn}\n")


# Visualization (Custom)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=pred,
    cmap="viridis"
)

plt.title("Self Training From Scratch")

plt.savefig("self_training_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=sk_pred,
    cmap="viridis"
)

plt.title("Sklearn Self Training")

plt.savefig("sklearn_self_training_plot.png")

plt.show()


# Loss Curve
plt.figure(figsize=(6,5))

plt.plot(model.loss_history)

plt.title("Self Training Loss")

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.savefig("loss_curve.png")

plt.show()
