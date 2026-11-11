
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_FILE = os.path.join(
    RESULTS_DIR, "comparison_metrics.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR, "f1_comparison.png"
)

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("No evaluation results found.")

# Generate F1-score comparison
plt.figure(figsize=(10, 6))

bars = plt.bar(
    df["Approach"],
    df["F1-score (%)"]
)

plt.xlabel("Evaluation Approach")
plt.ylabel("F1-score (%)")

plt.title(
    "F1-score Comparison of PR-Sentinel Approaches"
)

plt.ylim(0, 100)

plt.xticks(rotation=25, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.4)

# Display values above bars
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Graph saved to:", OUTPUT_FILE)
