"""
Training script for SVM from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from model import SVMScratch


# Generate dataset
X, y = make_blobs(
    n_samples=300,
    centers=2,
    random_state=42
)

y = np.where(y == 0, -1, 1)


# Train custom SVM
model = SVMScratch(
    learning_rate=0.001,
    lambda_param=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)


# Evaluation metrics
accuracy = accuracy_score(y, predictions)

print("Custom SVM Accuracy:", accuracy)

with open("metrics.txt","w") as f:
    f.write(f"Accuracy: {accuracy}")


# sklearn comparison
sk_model = SVC(kernel="linear")

sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

print("Sklearn SVM Accuracy:",
      accuracy_score(y, sk_pred))


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=predictions,
    cmap="coolwarm"
)

plt.title("SVM Classification")

plt.savefig("classification_plot.png")

plt.show()


# Loss curve
plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Hinge Loss Count")

plt.savefig("loss_curve.png")

plt.show()