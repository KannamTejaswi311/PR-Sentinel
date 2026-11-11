import json
import re
import time
import urllib.request
from pathlib import Path

DATASET = Path("evaluation/datasets/git_code_review_bench_100_w_open_prs.jsonl")
OUTPUT_DIR = Path("evaluation/diffs")
MANIFEST = Path("evaluation/diff_manifest.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_filename(repo, pr_url):
    repo_name = repo.split("/")[-1]
    pr_number = pr_url.rstrip("/").split("/")[-1]
    return f"{repo_name}_pr{pr_number}.diff"

records = []

with DATASET.open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))

results = []

for index, record in enumerate(records, start=1):
    repo = record["repo"]
    pr_url = record["pr_url_to_review"]
    diff_url = pr_url.rstrip("/") + ".diff"

    output_file = OUTPUT_DIR / safe_filename(repo, pr_url)

    print(f"[{index}/{len(records)}] Downloading {repo} - {pr_url}")

    request = urllib.request.Request(
        diff_url,
        headers={"User-Agent": "PR-Sentinel-Evaluation"}
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")

        if not content.strip() or "diff --git" not in content:
            status = "INVALID_DIFF"
            print("  Invalid or empty diff")
        else:
            output_file.write_text(content, encoding="utf-8")
            status = "SUCCESS"
            print(f"  Saved: {output_file}")

    except Exception as error:
        status = f"FAILED: {error}"
        print(f"  Download failed: {error}")

    results.append({
        "repo": repo,
        "pr_url": pr_url,
        "diff_file": str(output_file) if status == "SUCCESS" else "",
        "status": status
    })

    time.sleep(1)

import csv

with MANIFEST.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["repo", "pr_url", "diff_file", "status"]
    )
    writer.writeheader()
    writer.writerows(results)

successful = sum(r["status"] == "SUCCESS" for r in results)
failed = len(results) - successful

print("\n========== DOWNLOAD SUMMARY ==========")
print("Total PRs:", len(results))
print("Successful:", successful)
print("Failed/Invalid:", failed)
print("Manifest:", MANIFEST)