"""
Training script for Isolation Forest from scratch
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

from model import IsolationForestScratch


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


# Train custom model
model = IsolationForestScratch(n_trees=100)

model.fit(X)

pred = model.predict(X)


print("Custom Isolation Forest completed")


# sklearn comparison
sk_model = IsolationForest(contamination=0.1)

sk_labels = sk_model.fit_predict(X)

print("Sklearn Isolation Forest completed")


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("Isolation Forest anomaly detection completed\n")
    f.write(f"Detected anomalies: {sum(pred)}\n")


# Visualization (Custom)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=pred,
    cmap="coolwarm"
)

plt.title("Isolation Forest From Scratch")

plt.savefig("isolation_forest_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=sk_labels,
    cmap="coolwarm"
)

plt.title("Sklearn Isolation Forest")

plt.savefig("sklearn_isolation_forest_plot.png")

plt.show()