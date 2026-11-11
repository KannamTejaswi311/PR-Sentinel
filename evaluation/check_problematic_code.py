
import pandas as pd
from pathlib import Path
import re

gt = pd.read_csv("evaluation/ground_truth.csv")
manifest = pd.read_csv("evaluation/diff_manifest.csv")

results = []

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()

for _, issue in gt.iterrows():

    if pd.notna(issue["file_path"]):
        continue

    pr_url = issue["pr_url"]

    match = manifest[manifest["pr_url"] == pr_url]

    if match.empty:
        continue

    diff_file = Path(match.iloc[0]["diff_file"])

    if not diff_file.exists():
        continue

    diff_text = diff_file.read_text(
        encoding="utf-8",
        errors="replace"
    )

    code = issue["problematic_code"]

    if pd.isna(code) or not str(code).strip():
        results.append({
            "pr_url": pr_url,
            "issue_title": issue["issue_title"],
            "matched_lines": 0,
            "total_lines": 0,
            "match_percentage": 0,
            "status": "NO_CODE_SNIPPET"
        })
        continue

    code_lines = [
        normalize(line)
        for line in str(code).splitlines()
        if len(normalize(line)) >= 20
    ]

    diff_lines = {
        normalize(line)
        for line in diff_text.splitlines()
    }

    matched = sum(
        1 for line in code_lines
        if line in diff_lines
    )

    total = len(code_lines)

    percentage = (
        matched / total * 100
        if total else 0
    )

    results.append({
        "pr_url": pr_url,
        "issue_title": issue["issue_title"],
        "matched_lines": matched,
        "total_lines": total,
        "match_percentage": round(percentage, 2),
        "status": (
            "CODE_MATCH_FOUND"
            if matched > 0
            else "NO_CODE_MATCH"
        )
    })

output = pd.DataFrame(results)

output.to_csv(
    "evaluation/problematic_code_validation.csv",
    index=False
)

print("========== PROBLEMATIC CODE VALIDATION ==========")
print("Issues checked:", len(output))

print("\nResults:")
print(output["status"].value_counts().to_string())

print("\nDetailed results:")
print(
    output[
        ["issue_title", "matched_lines",
         "total_lines", "match_percentage", "status"]
    ].to_string(index=False)
)

print("\nSaved: evaluation/problematic_code_validation.csv")

