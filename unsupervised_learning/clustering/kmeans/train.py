"""
Training script for KMeans from scratch.

Includes:
- dataset generation
- clustering visualization
- sklearn comparison
- loss tracking
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import silhouette_score

from model import KMeans


# dataset generation
X, y = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# train custom model
model = KMeans(k=3, max_iters=100)

model.fit(X)

predictions = model.predict(X)

# evaluation metric
score = silhouette_score(X, predictions)

print("Custom KMeans Silhouette Score:", score)

with open("metrics.txt", "w") as f:
    f.write(f"Silhouette Score: {score}")


# sklearn comparison
sk_model = SKKMeans(n_clusters=3)

sk_labels = sk_model.fit_predict(X)

print("Sklearn Silhouette Score:", silhouette_score(X, sk_labels))


# clustering plot
plt.figure(figsize=(7,5))

plt.scatter(X[:,0], X[:,1], c=predictions)

plt.title("KMeans Clustering (From Scratch)")

plt.savefig("cluster_plot.png")

plt.show()


# loss curve
plt.figure(figsize=(7,5))

plt.plot(model.loss_history)

plt.title("KMeans Loss Curve")

plt.savefig("loss_curve.png")

plt.show()