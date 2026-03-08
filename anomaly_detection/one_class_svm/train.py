"""
Training script for One-Class SVM from scratch
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

from model import OneClassSVMScratch


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
model = OneClassSVMScratch()

model.fit(X)

pred = model.predict(X)

print("Custom One-Class SVM completed")


# sklearn comparison
sk_model = OneClassSVM(nu=0.1, kernel="rbf")

sk_labels = sk_model.fit_predict(X)

print("Sklearn One-Class SVM completed")


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("One-Class SVM anomaly detection completed\n")
    f.write(f"Detected anomalies: {sum(pred)}\n")


# Visualization (Custom)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=pred,
    cmap="coolwarm"
)

plt.title("One-Class SVM From Scratch")

plt.savefig("one_class_svm_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=sk_labels,
    cmap="coolwarm"
)

plt.title("Sklearn One-Class SVM")

plt.savefig("sklearn_one_class_svm_plot.png")

plt.show()


# Loss Curve
plt.figure(figsize=(6,5))

plt.plot(model.loss_history)

plt.title("Training Loss")

plt.savefig("loss_curve.png")

plt.show()