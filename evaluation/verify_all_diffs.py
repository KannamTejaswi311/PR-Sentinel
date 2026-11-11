import csv
from pathlib import Path

diff_dir = Path("evaluation/diffs")
manifest = Path("evaluation/diff_manifest.csv")

with manifest.open("r", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

# Add the two manually recovered files
manual_files = [
    diff_dir / "cal.com_pr11.diff",
    diff_dir / "cal.com_pr12.diff",
]

valid = []
invalid = []

for record in records:
    if record["status"] == "SUCCESS":
        path = Path(record["diff_file"])
        if path.exists() and "diff --git" in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            valid.append(path)
        else:
            invalid.append(str(path))

for path in manual_files:
    if path.exists() and "diff --git" in path.read_text(
        encoding="utf-8", errors="replace"
    ):
        valid.append(path)
    else:
        invalid.append(str(path))

print("========== DIFF VERIFICATION ==========")
print("Valid diffs:", len(valid))
print("Invalid/missing diffs:", len(invalid))

if invalid:
    print("\nInvalid files:")
    for item in invalid:
        print(item)