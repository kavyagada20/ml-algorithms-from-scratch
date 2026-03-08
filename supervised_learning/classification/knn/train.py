"""
Training script for KNN classifier from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from model import KNNClassifier


# Generate dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)


# Train custom model
model = KNNClassifier(k=5)

model.fit(X, y)

predictions = model.predict(X)


# Evaluation metrics
accuracy = accuracy_score(y, predictions)

print("Custom KNN Accuracy:", accuracy)

with open("metrics.txt","w") as f:
    f.write(f"Accuracy: {accuracy}")


# sklearn comparison
sk_model = KNeighborsClassifier(n_neighbors=5)

sk_model.fit(X, y)

sk_pred = sk_model.predict(X)

print("Sklearn Accuracy:",
      accuracy_score(y, sk_pred))


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=predictions,
    cmap="coolwarm"
)

plt.title("KNN Classification")

plt.savefig("classification_plot.png")

plt.show()