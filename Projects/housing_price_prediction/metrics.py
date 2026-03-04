import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    max_error
)

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Check if model and pipeline exist
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    print("⚠️ Model or pipeline not found! Please train the model first using main_joblibupdate.py.")
    exit()

# Load model and pipeline
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# Load test data
if not os.path.exists("input.csv"):
    print("⚠️ input.csv not found! Please ensure it's generated after training.")
    exit()

input_data = pd.read_csv("input.csv")

# Check if labels exist
if 'median_house_value' not in input_data.columns:
    print("⚠️ 'median_house_value' column not found in input.csv. Can't calculate accuracy metrics.")
    exit()

# Separate features and labels
X_test = input_data.drop("median_house_value", axis=1)
y_true = input_data["median_house_value"]

# Transform features and predict
X_test_prepared = pipeline.transform(X_test)
predictions = model.predict(X_test_prepared)

# Compute metrics
r2 = r2_score(y_true, predictions)
mae = mean_absolute_error(y_true, predictions)
mse = mean_squared_error(y_true, predictions)
rmse = np.sqrt(mse)
medae = median_absolute_error(y_true, predictions)
evs = explained_variance_score(y_true, predictions)
maxerr = max_error(y_true, predictions)

# Display results
print("\n📊 Extended Model Evaluation Metrics on Test Data:")
print("===================================================")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Median Absolute Error: {medae:.2f}")
print(f"Explained Variance Score: {evs:.4f}")
print(f"Max Error: {maxerr:.2f}")

# Save results to output file
output = X_test.copy()
output["actual_house_value"] = y_true
output["predicted_house_value"] = predictions
output.to_csv("output.csv", index=False)

print("\n✅ Inference Complete! Results with predictions saved to output.csv")
