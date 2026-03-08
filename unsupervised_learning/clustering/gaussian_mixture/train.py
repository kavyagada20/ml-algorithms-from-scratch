"""
Training script for Gaussian Mixture Model from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from model import GaussianMixtureScratch


# Generate dataset
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    random_state=42
)


# Train custom GMM
model = GaussianMixtureScratch(
    n_components=3,
    max_iter=100
)

model.fit(X)

labels = model.predict(X)


score = silhouette_score(X, labels)

print("Custom GMM Silhouette Score:", score)

with open("metrics.txt", "w") as f:
    f.write(f"Silhouette Score: {score}")


# sklearn comparison
sk_model = GaussianMixture(n_components=3)

sk_labels = sk_model.fit_predict(X)

sk_score = silhouette_score(X, sk_labels)

print("Sklearn GMM Silhouette Score:", sk_score)


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=labels,
    cmap="viridis"
)

plt.title("Gaussian Mixture Clustering")

plt.savefig("cluster_plot.png")

plt.show()


# Loss curve
plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("Log Likelihood")

plt.xlabel("Iteration")

plt.ylabel("Likelihood")

plt.savefig("loss_curve.png")

plt.show()