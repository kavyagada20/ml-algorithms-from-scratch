"""
Training script for Naive Bayes from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from model import NaiveBayesScratch


# Generate dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)


# Train custom model
model = NaiveBayesScratch()

model.fit(X, y)

predictions = model.predict(X)


# Evaluation
accuracy = accuracy_score(y, predictions)

print("Custom Naive Bayes Accuracy:", accuracy)

with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")


# sklearn comparison
sk_model = GaussianNB()

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

plt.title("Naive Bayes Classification")

plt.savefig("classification_plot.png")

plt.show()