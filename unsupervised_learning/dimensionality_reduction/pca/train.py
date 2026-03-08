"""
Training script for PCA from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from model import PCAScratch


# Load dataset
data = load_iris()
X = data.data
y = data.target


# Train custom PCA
model = PCAScratch(n_components=2)

model.fit(X)

X_reduced = model.transform(X)


print("Explained variance:", model.explained_variance)

with open("metrics.txt", "w") as f:
    f.write(f"Explained variance: {model.explained_variance}")


# sklearn comparison
sk_model = PCA(n_components=2)

X_sk = sk_model.fit_transform(X)

print("Sklearn explained variance:", sk_model.explained_variance_)


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    c=y,
    cmap="viridis"
)

plt.title("PCA From Scratch")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("pca_plot.png")

plt.show()


# sklearn visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X_sk[:,0],
    X_sk[:,1],
    c=y,
    cmap="viridis"
)

plt.title("Sklearn PCA")

plt.savefig("sklearn_pca_plot.png")

plt.show()