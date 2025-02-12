import pandas as pd

# Import necessary metrics from sklearn
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

# Load the dataset
file_path = "manual_analysis_updated.xlsx"  # Replace with the correct file path
df = pd.read_excel(file_path)

# Specify the ground truth and prediction columns
ground_truth_col = "Evaluation2"
prediction_columns = [
    "all-miniLM-L6-v2 (Comment)",
]

# Initialize a dictionary to store metrics for each prompt
results = {}

# Loop through each prediction column and calculate metrics
for pred_col in prediction_columns:
    # Extract ground truth and predictions
    y_true = df[ground_truth_col].astype(int).tolist()
    y_pred = df[pred_col].astype(int).tolist()

    # Calculate metrics
    metrics = {}
    metrics["F1-Score"] = f1_score(y_true, y_pred)
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    # Check for probability scores for AUROC and AUPRC (if available)
    if pred_col + " Probabilities" in df.columns:
        y_scores = df[pred_col + " Probabilities"].astype(float).tolist()
        metrics["AUROC"] = roc_auc_score(y_true, y_scores)
        metrics["AUPRC"] = average_precision_score(y_true, y_scores)
    else:
        metrics["AUROC"] = "NA (No probabilities)"
        metrics["AUPRC"] = "NA (No probabilities)"

    # Store the results
    results[pred_col] = metrics

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results).T

# Display the results
print("Evaluation Metrics for Each Prompt:")
print(results_df)

# # Optionally save the results to a CSV or Excel file
# results_df.to_csv("evaluation_metrics.csv", index=True)
# print("Metrics saved to evaluation_metrics.csv")
