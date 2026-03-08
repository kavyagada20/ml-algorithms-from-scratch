"""
Training script for Kernel PCA from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

from model import KernelPCAScratch


# Generate dataset
X, y = make_circles(
    n_samples=400,
    factor=0.3,
    noise=0.05,
    random_state=42
)


# Train custom Kernel PCA
model = KernelPCAScratch(
    n_components=2,
    gamma=15
)

model.fit(X)

X_reduced = model.transform(X)


# Save metrics
with open("metrics.txt","w") as f:
    f.write("Kernel PCA dimensionality reduction completed\n")
    f.write(f"Components: 2\n")


print("Kernel PCA completed")


# sklearn comparison
sk_model = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=15
)

X_sk = sk_model.fit_transform(X)


# Visualization (Original)
plt.figure(figsize=(6,5))

plt.scatter(X[:,0], X[:,1], c=y)

plt.title("Original Data")

plt.savefig("original_plot.png")

plt.show()


# Visualization (Custom Kernel PCA)
plt.figure(figsize=(6,5))

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y)

plt.title("Kernel PCA From Scratch")

plt.savefig("kernel_pca_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(6,5))

plt.scatter(X_sk[:,0], X_sk[:,1], c=y)

plt.title("Sklearn Kernel PCA")

plt.savefig("sklearn_kernel_pca_plot.png")

plt.show()