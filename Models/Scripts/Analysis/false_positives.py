import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def safe_log_transform(values, offset=1):
    """Return log10(v + offset) to avoid log(0)."""
    return [np.log10(v + offset) for v in values]


# -------------------------------------------------------------------
# 1) Load data & compute the needed counts
# -------------------------------------------------------------------
data_file_path = "manual_analysis_updated.xlsx"
data = pd.read_excel(data_file_path)

false_positives_gpt4 = set(
    data[(data["Evaluation2"] == 0) & (data["Prompt 10: GPT-4 (Comment)"] == 1)].index
)
false_negatives_gpt4 = set(
    data[(data["Evaluation2"] == 1) & (data["Prompt 10: GPT-4 (Comment)"] == 0)].index
)

false_positives_all_mini = set(
    data[(data["Evaluation2"] == 0) & (data["all-miniLM-L6-v2 (Comment)"] == 1)].index
)
false_negatives_all_mini = set(
    data[(data["Evaluation2"] == 1) & (data["all-miniLM-L6-v2 (Comment)"] == 0)].index
)

overlap_fp = len(false_positives_gpt4 & false_positives_all_mini)
overlap_fn = len(false_negatives_gpt4 & false_negatives_all_mini)

# -------------------------------------------------------------------
# 2) Define radar categories + raw counts
# -------------------------------------------------------------------
categories = ["FP", "FN", "Overlap FP", "Overlap FN"]
num_vars = len(categories)

gpt4_raw = [
    len(false_positives_gpt4),
    len(false_negatives_gpt4),
    overlap_fp,
    overlap_fn,
]
allmini_raw = [
    len(false_positives_all_mini),
    len(false_negatives_all_mini),
    overlap_fp,
    overlap_fn,
]

# -------------------------------------------------------------------
# 3) Log transform & close the circle
# -------------------------------------------------------------------
gpt4_log = safe_log_transform(gpt4_raw)
allmini_log = safe_log_transform(allmini_raw)
gpt4_log += gpt4_log[:1]
allmini_log += allmini_log[:1]

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# -------------------------------------------------------------------
# 4) Create the radar chart
# -------------------------------------------------------------------

# CHANGES: Increase figsize a bit
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Plot GPT-4
ax.plot(angles, gpt4_log, color="red", linewidth=2, label="GPT-4")
ax.fill(angles, gpt4_log, color="red", alpha=0.25)

# Plot all-miniLM
ax.plot(angles, allmini_log, color="blue", linewidth=2, label="all-miniLM")
ax.fill(angles, allmini_log, color="blue", alpha=0.25)

# CHANGES: Set radial limit with some padding
max_log_val = max(gpt4_log + allmini_log)
ax.set_rlim(0, max_log_val + 0.5)  # or something bigger if needed

# -------------------------------------------------------------------
# 5) Label each spoke with the original (untransformed) counts
# -------------------------------------------------------------------
# CHANGES: Increase the offset & the font size, and use a bbox so it’s easier to read
offset = 0.2
for i, angle in enumerate(angles[:-1]):
    # GPT-4 label
    ax.text(
        angle,
        gpt4_log[i] + offset,
        str(gpt4_raw[i]),
        color="red",
        ha="center",
        va="center",
        fontsize=12,  # bigger font
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),  # background box
    )
    # all-miniLM label
    ax.text(
        angle,
        allmini_log[i] - offset,
        str(allmini_raw[i]),
        color="blue",
        ha="center",
        va="center",
        fontsize=12,  # bigger font
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
    )

# -------------------------------------------------------------------
# 6) Final touches (labels, legend, etc.)
# -------------------------------------------------------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, color="black")
ax.set_yticks([])

plt.title("False Positives/Negatives", size=16, y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))

plt.tight_layout()
plt.show()

# save diagram
fig.savefig("radar_chart.png", dpi=300)
