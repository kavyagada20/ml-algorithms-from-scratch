"""
Training script for Linear Regression from scratch.

Includes:
- Synthetic dataset generation
- Model training
- sklearn comparison
- Metrics calculation (MSE, R2)
- Plot saving
- Metrics file saving
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error, r2_score
from model import LinearRegression


def main():

    # ------------------------------------------
    # Create outputs folder
    # ------------------------------------------
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------
    # Generate Dataset
    # ------------------------------------------
    X, y = make_regression(
        n_samples=100,
        n_features=1,
        noise=15,
        random_state=42
    )

    # ------------------------------------------
    # Train Our Model
    # ------------------------------------------
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    predictions = model.predict(X)

    # ------------------------------------------
    # Train sklearn Model (Comparison)
    # ------------------------------------------
    sklearn_model = SklearnLR()
    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X)

    # ------------------------------------------
    # Metrics
    # ------------------------------------------
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    sklearn_mse = mean_squared_error(y, sklearn_predictions)
    sklearn_r2 = r2_score(y, sklearn_predictions)

    print("---- Our Model ----")
    print("MSE:", mse)
    print("R2 Score:", r2)

    print("\n---- Sklearn Model ----")
    print("MSE:", sklearn_mse)
    print("R2 Score:", sklearn_r2)

    # ------------------------------------------
    # Save Metrics to File
    # ------------------------------------------
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write("Our Model\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 Score: {r2}\n\n")
        f.write("Sklearn Model\n")
        f.write(f"MSE: {sklearn_mse}\n")
        f.write(f"R2 Score: {sklearn_r2}\n")

    # ------------------------------------------
    # Plot Regression Comparison
    # ------------------------------------------
    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, predictions)
    plt.plot(X, sklearn_predictions, linestyle="--")
    plt.title("Linear Regression Comparison")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.savefig(os.path.join(output_dir, "regression_plot.png"))
    plt.show()

    # ------------------------------------------
    # Plot Loss Curve
    # ------------------------------------------
    plt.figure()
    plt.plot(model.loss_history)
    plt.title("Loss Curve (MSE vs Epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.show()


if __name__ == "__main__":
    main()