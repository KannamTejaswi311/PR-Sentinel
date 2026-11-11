import csv
from pathlib import Path

manifest = Path("evaluation/diff_manifest.csv")

manual_recovered = {
    "https://github.com/agentic-review-benchmarks/cal.com/pull/11":
        "evaluation/diffs/cal.com_pr11.diff",

    "https://github.com/agentic-review-benchmarks/cal.com/pull/12":
        "evaluation/diffs/cal.com_pr12.diff",
}

with manifest.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
    fieldnames = ["repo", "pr_url", "diff_file", "status"]

for row in rows:
    if row["pr_url"] in manual_recovered:
        row["diff_file"] = manual_recovered[row["pr_url"]]
        row["status"] = "SUCCESS"

with manifest.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Manifest updated successfully.")

successful = sum(row["status"] == "SUCCESS" for row in rows)
print("Successful:", successful)
print("Failed:", len(rows) - successful)