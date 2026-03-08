"""
Training script for Mean Shift clustering from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

from model import MeanShiftScratch


# Generate dataset
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)


# Train custom Mean Shift
model = MeanShiftScratch(bandwidth=2)

model.fit(X)

labels = model.predict(X)


score = silhouette_score(X, labels)

print("Custom Mean Shift Silhouette Score:", score)

with open("metrics.txt","w") as f:
    f.write(f"Silhouette Score: {score}")


# sklearn comparison
sk_model = MeanShift(bandwidth=2)

sk_labels = sk_model.fit_predict(X)

sk_score = silhouette_score(X, sk_labels)

print("Sklearn Mean Shift Silhouette Score:", sk_score)


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=labels,
    cmap="viridis"
)

plt.title("Mean Shift Clustering")

plt.savefig("cluster_plot.png")

plt.show()