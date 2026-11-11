"""
PR-Sentinel: Module 1 & 9 - GitHub Pull Request Integration & Output
Handles GitHub PR URL parsing, metadata retrieval, diff extraction,
review summary publishing, and inline comment generation using PyGithub and REST APIs.
"""

import os
import re
from typing import Any, Dict, List, Optional
import httpx


class GitHubPRIntegration:
    """
    Manages GitHub Pull Request interactions:
    fetches PR metadata and diffs, and formats / publishes review comments.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("TOKEN_GITHUB")

    def parse_pr_url(self, pr_url: str) -> Optional[Dict[str, Any]]:
        """
        Parse GitHub PR URL to extract owner, repo, and PR number.
        Example: https://github.com/octocat/Hello-World/pull/42
        """
        pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
        match = re.search(pattern, pr_url.strip())
        if not match:
            return None

        return {
            "owner": match.group(1),
            "repo": match.group(2),
            "pr_number": int(match.group(3)),
        }

    def fetch_pr_data_sync(self, pr_url: str, allow_demo: bool = True) -> Dict[str, Any]:
        """Synchronous version of fetch_pr_data using standard httpx.Client."""
        parsed = self.parse_pr_url(pr_url)
        if not parsed:
            raise ValueError(
                f"Invalid GitHub PR URL: '{pr_url}'. "
                "Expected format: https://github.com/owner/repository/pull/number"
            )

        owner = parsed["owner"]
        repo = parsed["repo"]
        pr_number = parsed["pr_number"]

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PR-Sentinel-Code-Review-Bot",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        api_base = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        err_detail = None

        try:
            with httpx.Client(timeout=15.0) as client:
                meta_resp = client.get(api_base, headers=headers)
                if meta_resp.status_code == 200:
                    meta = meta_resp.json()
                    diff_headers = dict(headers)
                    diff_headers["Accept"] = "application/vnd.github.v3.diff"
                    diff_resp = client.get(api_base, headers=diff_headers)
                    diff_text = diff_resp.text if diff_resp.status_code == 200 else ""

                    return {
                        "source": "github_api",
                        "owner": owner,
                        "repo": repo,
                        "pr_number": pr_number,
                        "title": meta.get("title", f"PR #{pr_number}"),
                        "author": meta.get("user", {}).get("login", "unknown"),
                        "base_branch": meta.get("base", {}).get("ref", "main"),
                        "head_branch": meta.get("head", {}).get("ref", "feature"),
                        "additions": meta.get("additions", 0),
                        "deletions": meta.get("deletions", 0),
                        "changed_files_count": meta.get("changed_files", 1),
                        "diff_content": diff_text,
                    }
                else:
                    err_detail = f"GitHub API responded with HTTP {meta_resp.status_code}: {meta_resp.text[:200]}"
        except Exception as e:
            err_detail = f"Connection error accessing GitHub API: {str(e)}"

        if not allow_demo:
            raise ConnectionError(
                f"Failed to fetch real GitHub PR data for {pr_url}. "
                f"Reason: {err_detail}. Set allow_demo=True to use offline demo fixtures."
            )

        demo_diff = f"""diff --git a/src/auth_service.py b/src/auth_service.py
new file mode 100644
index 0000000..f9821a3
--- /dev/null
+++ b/src/auth_service.py
@@ -0,0 +1,18 @@
+import sqlite3
+
+def authenticate_user(username, password):
+    # Query user account
+    conn = sqlite3.connect("production.db")
+    cursor = conn.cursor()
+    query = "SELECT id, username, role FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
+    cursor.execute(query)
+    user = cursor.fetchone()
+    return user
+
+def execute_maintenance(command_str):
+    import os
+    os.system("echo Maintenance started: " + command_str)
+"""
        return {
            "source": "demo_fallback",
            "owner": owner,
            "repo": repo,
            "pr_number": pr_number,
            "title": f"Feature: Authentication & Maintenance Updates (#{pr_number})",
            "author": owner,
            "base_branch": "main",
            "head_branch": "feature/auth-service",
            "additions": 18,
            "deletions": 0,
            "changed_files_count": 1,
            "diff_content": demo_diff,
            "note": f"Fetched demo fixture because external API was unreachable or unauthenticated: {err_detail}",
        }

    async def fetch_pr_data(self, pr_url: str, allow_demo: bool = True) -> Dict[str, Any]:
        """
        Fetch PR metadata, changed files, and unified diff from GitHub.
        Only falls back to offline demo fixture when allow_demo=True.
        """
        parsed = self.parse_pr_url(pr_url)
        if not parsed:
            raise ValueError(
                f"Invalid GitHub PR URL: '{pr_url}'. "
                "Expected format: https://github.com/owner/repository/pull/number"
            )

        owner = parsed["owner"]
        repo = parsed["repo"]
        pr_number = parsed["pr_number"]

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PR-Sentinel-Code-Review-Bot",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        api_base = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        err_detail = None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # 1. Fetch PR details
                meta_resp = await client.get(api_base, headers=headers)
                if meta_resp.status_code == 200:
                    meta = meta_resp.json()
                    # 2. Fetch diff
                    diff_headers = dict(headers)
                    diff_headers["Accept"] = "application/vnd.github.v3.diff"
                    diff_resp = await client.get(api_base, headers=diff_headers)
                    diff_text = diff_resp.text if diff_resp.status_code == 200 else ""

                    return {
                        "source": "github_api",
                        "owner": owner,
                        "repo": repo,
                        "pr_number": pr_number,
                        "title": meta.get("title", f"PR #{pr_number}"),
                        "author": meta.get("user", {}).get("login", "unknown"),
                        "base_branch": meta.get("base", {}).get("ref", "main"),
                        "head_branch": meta.get("head", {}).get("ref", "feature"),
                        "additions": meta.get("additions", 0),
                        "deletions": meta.get("deletions", 0),
                        "changed_files_count": meta.get("changed_files", 1),
                        "diff_content": diff_text,
                    }
                else:
                    err_detail = f"GitHub API HTTP {meta_resp.status_code}: {meta_resp.text[:200]}"
        except Exception as e:
            err_detail = f"Connection error: {str(e)}"

        if not allow_demo:
            raise ConnectionError(
                f"Failed to fetch real GitHub PR data for {pr_url}. "
                f"Reason: {err_detail}. Set allow_demo=True to use offline demo fixtures."
            )

        # Offline / Demo Fallback PR fixture
        demo_diff = f"""diff --git a/src/auth_service.py b/src/auth_service.py
new file mode 100644
index 0000000..f9821a3
--- /dev/null
+++ b/src/auth_service.py
@@ -0,0 +1,18 @@
+import sqlite3
+
+def authenticate_user(username, password):
+    # Query user account
+    conn = sqlite3.connect("production.db")
+    cursor = conn.cursor()
+    query = "SELECT id, username, role FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
+    cursor.execute(query)
+    user = cursor.fetchone()
+    return user
+
+def execute_maintenance(command_str):
+    import os
+    os.system("echo Maintenance started: " + command_str)
+"""
        return {
            "source": "demo_fallback",
            "owner": owner,
            "repo": repo,
            "pr_number": pr_number,
            "title": f"Feature: Authentication & Maintenance Updates (#{pr_number})",
            "author": owner,
            "base_branch": "main",
            "head_branch": "feature/auth-service",
            "additions": 18,
            "deletions": 0,
            "changed_files_count": 1,
            "diff_content": demo_diff,
            "note": f"Fetched demo fixture because external API was unreachable or unauthenticated: {err_detail}",
        }

    def post_review_comment(self, pr_url: str, body: str) -> Dict[str, Any]:
        """
        Publish a review summary comment to the GitHub PR.
        Requires GITHUB_TOKEN.
        """
        if not self.token:
            raise PermissionError("GITHUB_TOKEN is required to post review comments to GitHub.")

        parsed = self.parse_pr_url(pr_url)
        if not parsed:
            raise ValueError(f"Invalid GitHub PR URL: '{pr_url}'.")

        owner = parsed["owner"]
        repo = parsed["repo"]
        pr_number = parsed["pr_number"]

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "PR-Sentinel-Code-Review-Bot",
        }

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers=headers, json={"body": body})
            if resp.status_code in (200, 201):
                return resp.json()
            raise RuntimeError(f"GitHub API comment failed (HTTP {resp.status_code}): {resp.text[:300]}")

    def format_github_review_comment(
        self,
        synthesized_report: Dict[str, Any],
        pr_title: str = "Pull Request"
    ) -> str:
        """
        Format synthesized findings into GitHub-flavored markdown review summary.
        """
        risk = synthesized_report.get("risk_assessment", {})
        score = risk.get("score", 0)
        risk_level = risk.get("risk_level", "UNKNOWN")
        recommendation = risk.get("recommendation", "")

        findings = synthesized_report.get("findings", [])
        sev_counts = synthesized_report.get("severity_counts", {})

        # Status badge
        badge_color = "red" if score >= 50 else ("orange" if score >= 25 else "green")
        badge = f"![Risk Score: {score}/100](https://img.shields.io/badge/PR_Risk_Score-{score}%2F100-{badge_color})"

        comment_lines = [
            f"## 🛡️ PR-Sentinel Code Review Report — {pr_title}",
            "",
            badge,
            "",
            f"**Overall Assessment:** `{risk_level}` (Score: **{score}/100**)",
            f"**Recommendation:** {recommendation}",
            "",
            "### 📊 Severity Breakdown",
            f"- 🔴 **Critical:** {sev_counts.get('critical', 0)}",
            f"- 🟠 **High:** {sev_counts.get('high', 0)}",
            f"- 🟡 **Medium:** {sev_counts.get('medium', 0)}",
            f"- 🔵 **Low / Info:** {sev_counts.get('low', 0) + sev_counts.get('informational', 0)}",
            "",
            "### 🔍 Detailed Findings",
        ]

        if not findings:
            comment_lines.append("✅ **No issues or vulnerabilities detected.** Clean code changes!")
        else:
            for idx, f in enumerate(findings, 1):
                comment_lines.extend([
                    f"<details>",
                    f"<summary><b>{idx}. [{f.get('severity')}] {f.get('title')}</b> — <code>{f.get('file')}:{f.get('line')}</code></summary>",
                    "",
                    f"- **Category:** {f.get('category')}",
                    f"- **Description:** {f.get('description')}",
                ])
                if f.get("evidence"):
                    comment_lines.extend([
                        "- **Evidence:**",
                        "```python",
                        f"{f.get('evidence')}",
                        "```",
                    ])
                if f.get("suggested_fix"):
                    comment_lines.extend([
                        "- **Suggested Fix:**",
                        f"> {f.get('suggested_fix')}",
                    ])
                comment_lines.extend(["", "</details>", ""])

        comment_lines.extend([
            "---",
            "*Automated review generated by [PR-Sentinel](https://github.com/KannamTejaswi311/PR-Sentinel) with CodeBERT & Qwen2.5-Coder.*",
        ])

        return "\n".join(comment_lines)


_GITHUB_INTEGRATION_INSTANCE: Optional[GitHubPRIntegration] = None


def get_github_integration() -> GitHubPRIntegration:
    """Return singleton instance of GitHubPRIntegration."""
    global _GITHUB_INTEGRATION_INSTANCE
    if _GITHUB_INTEGRATION_INSTANCE is None:
        _GITHUB_INTEGRATION_INSTANCE = GitHubPRIntegration()
    return _GITHUB_INTEGRATION_INSTANCE
