
import pandas as pd
from pathlib import Path

ground_truth = pd.read_csv("evaluation/ground_truth.csv")
manifest = pd.read_csv("evaluation/diff_manifest.csv")

results = []

for _, issue in ground_truth.iterrows():

    pr_url = issue["pr_url"]
    file_path = issue["file_path"]

    if pd.isna(file_path) or not str(file_path).strip():
        results.append({
            "pr_url": pr_url,
            "issue_title": issue["issue_title"],
            "file_path": "",
            "status": "MISSING_GROUND_TRUTH_FILE_PATH"
        })
        continue

    match = manifest[manifest["pr_url"] == pr_url]

    if match.empty:
        status = "PR_NOT_FOUND"
    else:
        diff_file = match.iloc[0]["diff_file"]

        if not isinstance(diff_file, str) or not Path(diff_file).exists():
            status = "DIFF_MISSING"
        else:
            diff_text = Path(diff_file).read_text(
                encoding="utf-8",
                errors="replace"
            )

            normalized_path = str(file_path).replace("\\", "/")

            file_found = any(
                line.startswith(("+++ ", "--- "))
                and normalized_path in line
                for line in diff_text.splitlines()
            )

            status = (
                "FILE_FOUND_IN_DIFF"
                if file_found
                else "FILE_NOT_FOUND_IN_DIFF"
            )

    results.append({
        "pr_url": pr_url,
        "issue_title": issue["issue_title"],
        "file_path": str(file_path),
        "status": status
    })

validation = pd.DataFrame(results)

validation.to_csv(
    "evaluation/ground_truth_validation.csv",
    index=False
)

print("========== CORRECTED VALIDATION ==========")
print("Total issues:", len(validation))
print("\nResults:")
print(validation["status"].value_counts().to_string())

print("\nSaved: evaluation/ground_truth_validation.csv")
