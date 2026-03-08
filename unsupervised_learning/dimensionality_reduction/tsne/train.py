"""
Training script for t-SNE from scratch
"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

from model import TSNEScratch


# Load dataset
data = load_digits()

X = data.data
y = data.target


# Train custom t-SNE
model = TSNEScratch(
    n_components=2,
    perplexity=30,
    lr=200,
    epochs=300
)

model.fit(X)

X_reduced = model.predict(X)


# Save metrics
with open("metrics.txt","w") as f:

    f.write("t-SNE completed\n")
    f.write(f"Final Loss: {model.loss_history[-1]}\n")


print("Custom t-SNE completed")


# sklearn comparison
sk_model = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_sk = sk_model.fit_transform(X)


# Visualization (Custom)
plt.figure(figsize=(7,5))

plt.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    c=y,
    cmap="tab10"
)

plt.title("t-SNE From Scratch")

plt.savefig("tsne_plot.png")

plt.show()


# Visualization (Sklearn)
plt.figure(figsize=(7,5))

plt.scatter(
    X_sk[:,0],
    X_sk[:,1],
    c=y,
    cmap="tab10"
)

plt.title("Sklearn t-SNE")

plt.savefig("sklearn_tsne_plot.png")

plt.show()


# Loss Curve
plt.figure(figsize=(7,5))

plt.plot(model.loss_history)

plt.title("t-SNE Loss")

plt.xlabel("Epoch")

plt.ylabel("KL Divergence")

plt.savefig("loss_curve.png")

plt.show()