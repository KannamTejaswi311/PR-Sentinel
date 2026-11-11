
import pandas as pd
from pathlib import Path

ground_truth = pd.read_csv("evaluation/ground_truth.csv")
path_validation = pd.read_csv(
    "evaluation/ground_truth_validation.csv"
)
code_validation = pd.read_csv(
    "evaluation/problematic_code_validation.csv"
)

summary = {
    "Total ground-truth issues": len(ground_truth),
    "File found in diff": (
        path_validation["status"] == "FILE_FOUND_IN_DIFF"
    ).sum(),
    "Missing ground-truth file path": (
        path_validation["status"]
        == "MISSING_GROUND_TRUTH_FILE_PATH"
    ).sum(),
    "File not found in diff": (
        path_validation["status"] == "FILE_NOT_FOUND_IN_DIFF"
    ).sum(),
    "Missing-path issues with code matches": (
        code_validation["status"] == "CODE_MATCH_FOUND"
    ).sum(),
    "Missing-path issues without code matches": (
        code_validation["status"] == "NO_CODE_MATCH"
    ).sum(),
}

output = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Count"]
)

output.to_csv(
    "evaluation/validation_summary.csv",
    index=False
)

print("========== FINAL DATASET VALIDATION SUMMARY ==========")
print(output.to_string(index=False))
print("\nSaved: evaluation/validation_summary.csv")
