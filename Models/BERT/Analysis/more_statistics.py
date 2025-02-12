import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

# Load the dataset
file_path = "manual_analysis_huggingface.xlsx"  # Replace with the correct file path
df = pd.read_excel(file_path)

# Specify the ground truth and relevant columns
ground_truth_col = "Evaluation2"  # Ground truth (binary: 0s and 1s)
binary_column = "all-miniLM-L6-v2"  # Binary predictions (0s and 1s)
probability_column = "all-miniLM-L6-v2 (Similarity Score)"  # Probability scores

# Extract ground truth and predictions
y_true = df[ground_truth_col].astype(int).tolist()
y_pred_binary = df[binary_column].astype(int).tolist()  # Binary predictions
y_scores = df[probability_column].astype(float).tolist()  # Probability scores

# Calculate metrics
metrics = {
    "F1-Score": f1_score(y_true, y_pred_binary),
    "Accuracy": accuracy_score(y_true, y_pred_binary),
    "MCC": matthews_corrcoef(y_true, y_pred_binary),
    "AUROC": roc_auc_score(y_true, y_scores),
    "AUPRC": average_precision_score(y_true, y_scores),
}

# Display the results
print("Evaluation Metrics for all-miniLM-L6-v2:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Optionally save metrics to a file
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("all_mini_metrics.csv", index=False)
print("Metrics saved to all_mini_metrics.csv")
