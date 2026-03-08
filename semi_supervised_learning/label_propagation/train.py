"""
Training script for Label Propagation from scratch
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.semi_supervised import LabelPropagation

from model import LabelPropagationScratch


# Generate dataset
X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    random_state=42
)

# Remove most labels (simulate semi-supervised learning)
y = y_true.copy()

mask = np.random.rand(len(y)) < 0.8

y[mask] = -1


# Train custom model
model = LabelPropagationScratch()

model.fit(X, y)

pred = model.predict(X)

print("Custom Label Propagation completed")


# sklearn comparison
sk_model = LabelPropagation()

sk_model.fit(X, y)

sk_labels = sk_model.predict(X)

print("Sklearn Label Propagation completed")


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("Label Propagation completed\n")
    f.write(f"Total labeled points predicted: {len(pred)}\n")


# Visualization (Custom)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=pred,
    cmap="viridis"
)

plt.title("Label Propagation From Scratch")

plt.savefig("label_propagation_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=sk_labels,
    cmap="viridis"
)

plt.title("Sklearn Label Propagation")

plt.savefig("sklearn_label_propagation_plot.png")

plt.show()


# Loss Curve
plt.figure(figsize=(6,5))

plt.plot(model.loss_history)

plt.title("Propagation Loss")

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.savefig("loss_curve.png")

plt.show()