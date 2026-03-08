"""
Training script for Logistic Regression from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model import LogisticRegressionScratch


# Generate dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)


# Train custom model
model = LogisticRegressionScratch(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)


# Evaluation metrics
accuracy = accuracy_score(y, predictions)

print("Custom Logistic Regression Accuracy:", accuracy)

with open("metrics.txt","w") as f:
    f.write(f"Accuracy: {accuracy}")


# sklearn comparison
sk_model = LogisticRegression()

sk_model.fit(X,y)

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

plt.title("Logistic Regression Classification")

plt.savefig("classification_plot.png")

plt.show()


# Loss curve
plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Log Loss")

plt.savefig("loss_curve.png")

plt.show()