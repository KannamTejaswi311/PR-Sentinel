
import json
import pandas as pd
from pathlib import Path

dataset_path = Path(
    "evaluation/datasets/git_code_review_bench_100_w_open_prs.jsonl"
)

output_path = Path("evaluation/ground_truth.csv")

records = []

with dataset_path.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        pr = json.loads(line)

        for issue in pr.get("issues", []):
            records.append({
                "repo": pr.get("repo"),
                "pr_url": pr.get("pr_url_to_review"),
                "issue_title": issue.get("title"),
                "description": issue.get("description"),
                "file_path": issue.get("file_path"),
                "start_line": issue.get("start_line"),
                "end_line": issue.get("end_line"),
                "problematic_code": issue.get("problematic_code_snippet"),
                "rule_name": issue.get("rule_name")
            })

df = pd.DataFrame(records)
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Ground-truth extraction completed!")
print("Total findings:", len(df))
print("Saved to:", output_path)
print("\nFirst five records:")
print(df.head().to_string())
