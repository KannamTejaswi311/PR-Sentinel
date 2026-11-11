"""
PR-Sentinel: Module 10 - Dataset Documentation & Benchmark Evaluation
Documents Google-discovered public datasets for CodeBERT evaluation and code-review benchmarks,
and implements a controlled benchmark evaluation suite for functional verification.
"""

from typing import Any, Dict, List


DATASETS_REGISTRY = [
    {
        "name": "CodeXGLUE Defect Detection (microsoft/CodeXGLUE)",
        "source": "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection",
        "huggingface_url": "https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection",
        "discovery_method": "Google Search / Hugging Face Catalog Index",
        "license": "MIT License",
        "format": "JSON / JSONL with source code functions and binary defect labels (0=secure, 1=vulnerable)",
        "relevance": (
            "Standard Microsoft / CodeBERT benchmark for evaluating code representation and defect detection. "
            "Derived from Devign dataset (NeurIPS 2019) across diverse C/C++ and Python functions."
        ),
        "intended_use": "Evaluating CodeBERT contextual representations on vulnerability identification.",
        "verified_access": True,
    },
    {
        "name": "GitHub Code Review Dataset (ronantakizawa/github-codereview)",
        "source": "https://huggingface.co/datasets/ronantakizawa/github-codereview",
        "huggingface_url": "https://huggingface.co/datasets/ronantakizawa/github-codereview",
        "discovery_method": "Google Search / Hugging Face Catalog Index",
        "license": "Apache 2.0",
        "format": "Parquet with fields: before_code, after_code, reviewer_comment, language, quality_score",
        "relevance": (
            "Hundreds of thousands of human-written code reviews from top GitHub repositories "
            "(TensorFlow, React, VS Code) paired with exact Git diffs."
        ),
        "intended_use": "Benchmarking LLM and multi-agent review recommendation accuracy against human reviews.",
        "verified_access": True,
    },
    {
        "name": "PR-Review-Bench (Qodo / CodiumAI)",
        "source": "https://huggingface.co/datasets/Qodo/PR-Review-Bench",
        "huggingface_url": "https://huggingface.co/datasets/Qodo/PR-Review-Bench",
        "discovery_method": "Google Search / Hugging Face Datasets",
        "license": "Open Source Benchmark",
        "format": "JSON with PR metadata, diffs, changed files, ground truth issue titles and descriptions",
        "relevance": (
            "Production pull request review evaluation benchmark specifically measuring accuracy, "
            "hallucination rates, and false positives in AI PR reviewers."
        ),
        "intended_use": "Evaluation of Review Synthesizer, false-positive filtering, and severity scoring.",
        "verified_access": True,
    },
]


CONTROLLED_BENCHMARK_CASES = [
    {
        "id": "BENCH-01",
        "title": "SQL Injection in User Lookup",
        "type": "Security (CWE-89)",
        "expected_severity": "HIGH",
        "expected_detection": "SQL Injection",
        "diff": """diff --git a/auth.py b/auth.py
new file mode 100644
--- /dev/null
+++ b/auth.py
@@ -0,0 +1,5 @@
+import sqlite3
+def find_user(username):
+    conn = sqlite3.connect("app.db")
+    return conn.execute("SELECT * FROM users WHERE username = '" + username + "'").fetchone()
""",
    },
    {
        "id": "BENCH-02",
        "title": "Hardcoded AWS Secret Key",
        "type": "Security (CWE-798)",
        "expected_severity": "HIGH",
        "expected_detection": "Hardcoded Secret",
        "diff": """diff --git a/config.py b/config.py
new file mode 100644
--- /dev/null
+++ b/config.py
@@ -0,0 +1,4 @@
+def get_aws_client():
+    api_key = "AKIAIOSFODNN7EXAMPLE"
+    secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
+    return api_key, secret
""",
    },
    {
        "id": "BENCH-03",
        "title": "Potential ZeroDivisionError and Mutable Default Argument",
        "type": "Logic & Runtime Bug",
        "expected_severity": "HIGH",
        "expected_detection": "ZeroDivisionError",
        "diff": """diff --git a/math_utils.py b/math_utils.py
new file mode 100644
--- /dev/null
+++ b/math_utils.py
@@ -0,0 +1,4 @@
+def compute_ratio(val, divisor, history=[]):
+    history.append(val)
+    return val / divisor
""",
    },
    {
        "id": "BENCH-04",
        "title": "Arbitrary Command Execution via os.system",
        "type": "Security (CWE-78)",
        "expected_severity": "CRITICAL",
        "expected_detection": "Command Injection",
        "diff": """diff --git a/backup.py b/backup.py
new file mode 100644
--- /dev/null
+++ b/backup.py
@@ -0,0 +1,4 @@
+import os
+def run_backup(target_folder):
+    os.system("tar -czf backup.tar.gz " + target_folder)
""",
    },
    {
        "id": "BENCH-05",
        "title": "Clean, Parameterized, Robust Function",
        "type": "Clean Code",
        "expected_severity": "LOW",
        "expected_detection": "None",
        "diff": """diff --git a/safe_user.py b/safe_user.py
new file mode 100644
--- /dev/null
+++ b/safe_user.py
@@ -0,0 +1,7 @@
+import sqlite3
+
+def get_user_safely(db_path: str, username: str):
+    with sqlite3.connect(db_path) as conn:
+        cursor = conn.cursor()
+        cursor.execute("SELECT id, username FROM users WHERE username = ?", (username,))
+        return cursor.fetchone()
""",
    },
]


def get_dataset_registry() -> List[Dict[str, Any]]:
    """Return documented datasets discovered via Google."""
    return DATASETS_REGISTRY


def get_controlled_benchmarks() -> List[Dict[str, Any]]:
    """Return controlled benchmark suite cases."""
    return CONTROLLED_BENCHMARK_CASES
