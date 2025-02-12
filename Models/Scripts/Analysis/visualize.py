import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "manual_analysis_updated.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Specify the ground truth column and the prediction columns
ground_truth_col = "Evaluation2"
prediction_columns = [
    "Prompt 1: GPT-4 (Code & Comment)",
    "Prompt 2: GPT-4 (Comment)",
    "Prompt 3: GPT-4 (Code & Comment)",
    "Prompt 4: GPT-4 (Comment)",
    "Prompt 5: GPT-4 (Code & Comment)",
    "Prompt 6: GPT-4 (Comment)",
    "Prompt 7: GPT-4 (Code & Comment)",
    "Prompt 8: GPT-4 (Comment)",
]

# Identify false negatives and false positives for each prompt
false_negative_dict = {}
false_positive_dict = {}
for pred_col in prediction_columns:
    false_negatives = df[(df[ground_truth_col] == 1) & (df[pred_col] == 0)]
    false_positives = df[(df[ground_truth_col] == 0) & (df[pred_col] == 1)]
    false_negative_dict[pred_col] = set(false_negatives.index)
    false_positive_dict[pred_col] = set(false_positives.index)

# Create overlap matrices for FNs and FPs
fn_overlap_matrix = np.zeros((len(prediction_columns), len(prediction_columns)))
fp_overlap_matrix = np.zeros((len(prediction_columns), len(prediction_columns)))

for i, col_i in enumerate(prediction_columns):
    for j, col_j in enumerate(prediction_columns):
        fn_overlap_matrix[i, j] = len(
            false_negative_dict[col_i] & false_negative_dict[col_j]
        )
        fp_overlap_matrix[i, j] = len(
            false_positive_dict[col_i] & false_positive_dict[col_j]
        )

# Plot the heatmaps
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# False negatives heatmap
sns.heatmap(
    fn_overlap_matrix,
    annot=True,
    fmt=".0f",
    ax=axes[0],
    xticklabels=prediction_columns,
    yticklabels=prediction_columns,
    cmap="Blues",
)
axes[0].set_title("Overlap of False Negatives Across Prompts")
axes[0].set_xlabel("Prompts")
axes[0].set_ylabel("Prompts")
axes[0].tick_params(axis="x", rotation=45)

# False positives heatmap
sns.heatmap(
    fp_overlap_matrix,
    annot=True,
    fmt=".0f",
    ax=axes[1],
    xticklabels=prediction_columns,
    yticklabels=prediction_columns,
    cmap="Reds",
)
axes[1].set_title("Overlap of False Positives Across Prompts")
axes[1].set_xlabel("Prompts")
axes[1].set_ylabel("Prompts")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
