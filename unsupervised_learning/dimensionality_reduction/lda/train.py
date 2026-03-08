"""
Training script for LDA from scratch
"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from model import LDAScratch


# Load dataset
data = load_iris()

X = data.data
y = data.target


# Train custom LDA
model = LDAScratch(n_components=2)

model.fit(X, y)

X_reduced = model.transform(X)


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("LDA dimensionality reduction completed\n")
    f.write("Components: 2\n")


print("Custom LDA completed")


# sklearn comparison
sk_model = LinearDiscriminantAnalysis(n_components=2)

X_sk = sk_model.fit_transform(X, y)


# Visualization (Custom)
plt.figure(figsize=(7,5))

plt.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    c=y,
    cmap="viridis"
)

plt.title("LDA From Scratch")

plt.savefig("lda_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(7,5))

plt.scatter(
    X_sk[:,0],
    X_sk[:,1],
    c=y,
    cmap="viridis"
)

plt.title("Sklearn LDA")

plt.savefig("sklearn_lda_plot.png")

plt.show()