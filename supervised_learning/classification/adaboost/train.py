"""
Training script for AdaBoost from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

from model import AdaBoostScratch


# Generate dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)


# Train custom AdaBoost
model = AdaBoostScratch(n_estimators=50)

model.fit(X, y)

pred = model.predict(X)

accuracy = accuracy_score(
    np.where(y == 0, -1, 1),
    pred
)

print("Custom AdaBoost Accuracy:", accuracy)

with open("metrics.txt","w") as f:
    f.write(f"Accuracy: {accuracy}")


# sklearn comparison
sk_model = AdaBoostClassifier(n_estimators=50)

sk_model.fit(X,y)

sk_pred = sk_model.predict(X)

print("Sklearn Accuracy:",
      accuracy_score(y, sk_pred))


# Visualization
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=pred,
    cmap="coolwarm"
)

plt.title("AdaBoost Classification")

plt.savefig("classification_plot.png")

plt.show()


# Loss curve
plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("Training Error")

plt.xlabel("Iteration")

plt.ylabel("Error")

plt.savefig("loss_curve.png")

plt.show()