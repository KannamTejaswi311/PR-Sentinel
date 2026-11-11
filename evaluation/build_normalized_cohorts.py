"""
PR-Sentinel: Dataset Normalization Script
Generates evaluation/datasets/normalized/ground_truth_cohorts.csv
without modifying any original assets.
"""

import re
import pandas as pd
from pathlib import Path

def categorize_issue(rule_name: str, issue_title: str, description: str) -> str:
    combined = f"{rule_name} {issue_title} {description}".lower()
    
    # Security patterns
    if any(k in combined for k in [
        "security", "cwe", "sql injection", "command injection", "xss", "cross-site",
        "secret", "credential", "password", "token", "auth", "permission", "traversal",
        "vulnerability", "sanitize", "escape"
    ]):
        return "SECURITY"
    
    # Bug / Correctness patterns
    if any(k in combined for k in [
        "bug", "error", "exception", "null", "undefined", "crash", "division by zero",
        "off-by-one", "race condition", "deadlock", "leak", "infinite loop", "fail",
        "unhandled", "incorrect", "broken"
    ]):
        return "BUG"
    
    # Performance patterns
    if any(k in combined for k in [
        "performance", "memory", "n+1", "slow", "complexity", "o(n", "cache",
        "unbounded", "latency", "efficient", "overhead"
    ]):
        return "PERFORMANCE"
    
    # Code quality / Style / Convention
    if any(k in combined for k in [
        "format", "lint", "naming", "convention", "pep8", "unused import", "docstring",
        "annotation", "type", "semicolon", "style", "access control", "final"
    ]):
        return "CODE_QUALITY"
        
    return "UNKNOWN"

def build_cohorts():
    gt_path = Path("evaluation/ground_truth.csv")
    gt_val_path = Path("evaluation/ground_truth_validation.csv")
    prob_val_path = Path("evaluation/problematic_code_validation.csv")
    output_path = Path("evaluation/datasets/normalized/ground_truth_cohorts.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    gt = pd.read_csv(gt_path)
    gt_val = pd.read_csv(gt_val_path)
    prob_val = pd.read_csv(prob_val_path)
    
    # Create lookup for problematic code status by (pr_url, issue_title)
    code_match_lookup = {}
    for _, row in prob_val.iterrows():
        key = (str(row["pr_url"]).strip(), str(row["issue_title"]).strip())
        code_match_lookup[key] = str(row["status"]).strip()
        
    cohort_records = []
    
    for idx, row in gt.iterrows():
        pr_url = str(row["pr_url"]).strip() if pd.notna(row["pr_url"]) else ""
        issue_title = str(row["issue_title"]).strip() if pd.notna(row["issue_title"]) else ""
        file_path = str(row["file_path"]).strip() if pd.notna(row["file_path"]) else ""
        repo = str(row["repo"]).strip() if pd.notna(row["repo"]) else ""
        rule_name = str(row["rule_name"]).strip() if pd.notna(row["rule_name"]) else ""
        desc = str(row["description"]).strip() if pd.notna(row["description"]) else ""
        code = str(row["problematic_code"]).strip() if pd.notna(row["problematic_code"]) else ""
        
        # PR number extraction
        pr_num_match = re.search(r"/pull/(\d+)", pr_url)
        pr_number = int(pr_num_match.group(1)) if pr_num_match else 0
        
        # Determine validation status from gt_val
        val_match = gt_val.iloc[idx]
        val_status = str(val_match["status"]).strip()
        
        # Assign Cohort based on ratified Protocol rules
        if val_status == "FILE_FOUND_IN_DIFF":
            cohort = "COHORT_1_CONFIRMED"
            verifiable = True
        elif val_status == "MISSING_GROUND_TRUTH_FILE_PATH":
            prob_status = code_match_lookup.get((pr_url, issue_title), "NO_CODE_MATCH")
            if prob_status == "CODE_MATCH_FOUND":
                cohort = "COHORT_2_CANDIDATE"
                verifiable = False  # Candidate, held pending manual review
            else:
                cohort = "COHORT_3_UNRESOLVED"
                verifiable = False
        else:
            cohort = "COHORT_3_UNRESOLVED"
            verifiable = False
            
        std_cat = categorize_issue(rule_name, issue_title, desc)
        
        start_line = int(row["start_line"]) if pd.notna(row["start_line"]) else None
        end_line = int(row["end_line"]) if pd.notna(row["end_line"]) else None
        
        cohort_records.append({
            "issue_id": f"gt_{idx+1:04d}",
            "repo": repo,
            "pr_number": pr_number,
            "pr_url": pr_url,
            "file_path": file_path if file_path else None,
            "cohort": cohort,
            "verifiable": verifiable,
            "start_line": start_line,
            "end_line": end_line,
            "problematic_code": code if code else None,
            "rule_name": rule_name if rule_name else None,
            "standardized_category": std_cat,
            "validation_status": val_status
        })
        
    df_cohorts = pd.DataFrame(cohort_records)
    df_cohorts.to_csv(output_path, index=False)
    print(f"Normalized cohorts saved to {output_path}")
    print("Cohort breakdown:")
    print(df_cohorts["cohort"].value_counts())
    print("\nVerifiable breakdown:")
    print(df_cohorts["verifiable"].value_counts())
    print("\nCategory breakdown:")
    print(df_cohorts["standardized_category"].value_counts())

if __name__ == "__main__":
    build_cohorts()
