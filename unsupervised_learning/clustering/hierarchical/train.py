"""
Training script for Hierarchical Clustering from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from model import HierarchicalClustering


# Generate dataset
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    random_state=42
)


# Train custom hierarchical clustering
model = HierarchicalClustering(n_clusters=3)

model.fit(X)

labels = model.predict(X)


score = silhouette_score(X, labels)

print("Custom Hierarchical Silhouette Score:", score)

with open("metrics.txt", "w") as f:
    f.write(f"Silhouette Score: {score}")


# sklearn comparison
sk_model = AgglomerativeClustering(n_clusters=3)

sk_labels = sk_model.fit_predict(X)

sk_score = silhouette_score(X, sk_labels)

print("Sklearn Silhouette Score:", sk_score)


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=labels,
    cmap="viridis"
)

plt.title("Hierarchical Clustering")

plt.savefig("cluster_plot.png")

plt.show()