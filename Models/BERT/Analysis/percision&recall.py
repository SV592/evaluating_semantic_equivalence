import pandas as pd

# Load the dataset
file_path = "manual_analysis_huggingface.xlsx"
df = pd.read_excel(file_path)


# Define a function to calculate precision and recall
def calculate_precision_recall(true_col, pred_col):
    TP = ((df[true_col] == 1) & (df[pred_col] == 1)).sum()
    FP = ((df[true_col] == 0) & (df[pred_col] == 1)).sum()
    FN = ((df[true_col] == 1) & (df[pred_col] == 0)).sum()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall


# Define the comparisons
comparisons = {
    "Evaluation2 vs all-miniLM-L6-v2": (
        "Evaluation2",
        "all-miniLM-L6-v2",
    ),
}

# Calculate precision and recall for each comparison
results = {}
for name, (true_col, pred_col) in comparisons.items():
    precision, recall = calculate_precision_recall(true_col, pred_col)
    results[name] = {"Precision": precision, "Recall": recall}

# Convert results to a DataFrame for display
results_df = pd.DataFrame(results).T
print(results_df)
