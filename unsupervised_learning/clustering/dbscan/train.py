"""
Training script for DBSCAN from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from model import DBSCANScratch


# Generate dataset
X, _ = make_moons(
    n_samples=300,
    noise=0.05,
    random_state=42
)


# Train custom DBSCAN
model = DBSCANScratch(
    eps=0.3,
    min_samples=5
)

model.fit(X)

labels = model.predict(X)


score = silhouette_score(X, labels)

print("Custom DBSCAN Silhouette Score:", score)

with open("metrics.txt","w") as f:
    f.write(f"Silhouette Score: {score}")


# sklearn comparison
sk_model = DBSCAN(
    eps=0.3,
    min_samples=5
)

sk_labels = sk_model.fit_predict(X)

sk_score = silhouette_score(X, sk_labels)

print("Sklearn DBSCAN Silhouette Score:", sk_score)


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=labels,
    cmap="viridis"
)

plt.title("DBSCAN Clustering")

plt.savefig("cluster_plot.png")

plt.show()