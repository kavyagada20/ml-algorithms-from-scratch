"""
Training script for Local Outlier Factor from scratch
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor

from model import LocalOutlierFactorScratch


# Generate dataset
X, _ = make_blobs(
    n_samples=300,
    centers=1,
    cluster_std=0.6,
    random_state=42
)

# Add anomalies
anomalies = np.random.uniform(-6, 6, (20, 2))

X = np.vstack([X, anomalies])


# Train custom LOF
model = LocalOutlierFactorScratch(k=10)

model.fit(X)

pred = model.predict(X)

print("Custom LOF completed")


# sklearn comparison
sk_model = LocalOutlierFactor(n_neighbors=10)

sk_labels = sk_model.fit_predict(X)

print("Sklearn LOF completed")


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("Local Outlier Factor anomaly detection completed\n")
    f.write(f"Detected anomalies: {sum(pred)}\n")


# Visualization (Custom)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=pred,
    cmap="coolwarm"
)

plt.title("Local Outlier Factor From Scratch")

plt.savefig("lof_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=sk_labels,
    cmap="coolwarm"
)

plt.title("Sklearn LOF")

plt.savefig("sklearn_lof_plot.png")

plt.show()