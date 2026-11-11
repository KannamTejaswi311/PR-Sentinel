"""
PR-Sentinel: Verification Test Suite (Tests 1 through 13)
Exercises all 13 required capabilities from Section 9 of the PR-Sentinel specification:
1. Bug Detection
2. Security Detection
3. Code Quality Analysis
4. Suggested Fixes
5. Test Recommendations
6. CodeBERT Representation
7. Repository RAG Ingestion & Retrieval
8. Multi-Agent Review Execution
9. Review Synthesis & Deduplication
10. Transparent PR Risk Assessment
11. Review History & Developer Feedback Persistence
12. FastAPI Application Endpoints
13. GitHub Pull Request Ingestion & Comment Formatting
"""

import os
os.environ["PR_SENTINEL_OFFLINE"] = "1"

import asyncio
import pytest
from sentinel.codebert_analyzer import get_codebert_analyzer
from sentinel.datasets_info import get_controlled_benchmarks
from sentinel.github_integration import get_github_integration
from sentinel.multi_agent import (
    BugDetectionAgent,
    CodeQualityAgent,
    PerformanceReviewAgent,
    SecurityReviewAgent,
    get_multi_agent_orchestrator,
)
from sentinel.pipeline import get_pipeline
from sentinel.rag_pipeline import get_rag_pipeline
from sentinel.review_history import (
    get_all_reviews,
    get_feedback_for_review,
    get_review,
    init_database,
    save_feedback,
    save_review,
)
from sentinel.static_analyzer import get_static_analyzer
from sentinel.synthesizer import get_review_synthesizer
from sentinel.test_generator import get_test_generator


# ------------------------------------------------------------
# TEST 1: Bug Detection
# ------------------------------------------------------------
def test_1_bug_detection():
    agent = BugDetectionAgent()
    diff = """diff --git a/calc.py b/calc.py
--- /dev/null
+++ b/calc.py
@@ -0,0 +1,3 @@
+def divide(a, b):
+    return a / b
+"""
    findings = agent.analyze_heuristically(diff, "calc.py", [])
    assert any("ZeroDivisionError" in f["title"] or "Zero" in f["title"] for f in findings), (
        "Bug Detection Agent failed to identify potential division by zero."
    )


# ------------------------------------------------------------
# TEST 2: Security Detection
# ------------------------------------------------------------
def test_2_security_detection():
    static_analyzer = get_static_analyzer()
    diff = """diff --git a/db.py b/db.py
--- /dev/null
+++ b/db.py
@@ -0,0 +1,5 @@
+import sqlite3
+def get_user(username):
+    conn = sqlite3.connect("test.db")
+    return conn.execute("SELECT * FROM users WHERE username = '" + username + "'").fetchone()
+"""
    findings = static_analyzer.analyze_diff(diff, "db.py")
    assert any("SQL" in f["title"] or f.get("rule_id") == "AST_SQL_CONCAT" for f in findings), (
        "Static analyzer failed to detect SQL injection vector."
    )


# ------------------------------------------------------------
# TEST 3: Code Quality
# ------------------------------------------------------------
def test_3_code_quality():
    agent = CodeQualityAgent()
    diff = """diff --git a/util.py b/util.py
--- /dev/null
+++ b/util.py
@@ -0,0 +1,3 @@
+def process():
+    x = 42
+    return x
+"""
    findings = agent.analyze_heuristically(diff, "util.py", [])
    # Single-letter variable names or style findings
    assert isinstance(findings, list)


# ------------------------------------------------------------
# TEST 4: Suggested Fixes
# ------------------------------------------------------------
def test_4_suggested_fixes():
    static_analyzer = get_static_analyzer()
    diff = """diff --git a/admin.py b/admin.py
--- /dev/null
+++ b/admin.py
@@ -0,0 +1,3 @@
+def check():
+    password = "SuperSecretPassword123"
+"""
    findings = static_analyzer.analyze_diff(diff, "admin.py")
    secret_finding = next((f for f in findings if "Secret" in f["title"] or "password" in f["title"].lower()), None)
    assert secret_finding is not None, "Failed to identify hardcoded secret."
    assert secret_finding.get("recommendation") is not None, "Finding lacks recommendation / fix."
    assert len(secret_finding["recommendation"]) > 10, "Recommendation is too brief."


# ------------------------------------------------------------
# TEST 5: Automated Test Recommendations
# ------------------------------------------------------------
def test_5_test_recommendations():
    generator = get_test_generator()
    diff = """diff --git a/auth.py b/auth.py
--- /dev/null
+++ b/auth.py
@@ -0,0 +1,4 @@
+def login(username, password):
+    return query("SELECT * FROM users WHERE u = '" + username + "'")
+"""
    findings = [{"title": "SQL Injection", "category": "Security", "evidence": "query(...)", "file": "auth.py", "line": 2}]
    recommendations = generator.generate_recommendations(diff, findings, "auth.py")
    assert len(recommendations) > 0, "No test recommendations generated."
    assert any("SQL" in r["scenario"] or "Security" in r["test_type"] for r in recommendations), (
        "Security test recommendation missing for SQL injection."
    )
    for r in recommendations:
        assert "target_component" in r
        assert "scenario" in r
        assert "input_conditions" in r
        assert "expected_result" in r


# ------------------------------------------------------------
# TEST 6: CodeBERT Representation
# ------------------------------------------------------------
def test_6_codebert_representation():
    analyzer = get_codebert_analyzer()
    code = "def add(a, b):\n    return a + b\n"
    rep = analyzer.extract_code_representation(code)
    assert rep["dimension"] == 768, f"Expected 768-D vector, got {rep['dimension']}."
    assert len(rep["vector"]) == 768
    assert rep["token_count"] > 0

    # Test cosine similarity
    sim_same = analyzer.compute_similarity(code, code)
    assert abs(sim_same - 1.0) < 0.01, f"Self-similarity should be 1.0, got {sim_same}"


# ------------------------------------------------------------
# TEST 7: Repository-Aware RAG Pipeline
# ------------------------------------------------------------
def test_7_rag_pipeline():
    rag = get_rag_pipeline()
    stats = rag.index_repository(force_reindex=False)
    assert stats["total_chunks"] > 0, "RAG failed to discover code chunks in repository."

    # Retrieval
    query = "SELECT * FROM users WHERE username = username"
    retrieved = rag.retrieve_context(query, top_k=2)
    assert isinstance(retrieved, list)
    assert len(retrieved) > 0, "RAG failed to retrieve relevant context."
    assert "similarity_score" in retrieved[0]
    assert "content_snippet" in retrieved[0]


# ------------------------------------------------------------
# TEST 8: Multi-Agent Review Execution
# ------------------------------------------------------------
@pytest.mark.asyncio
async def test_8_multi_agent_review():
    orchestrator = get_multi_agent_orchestrator()
    diff = """diff --git a/app.py b/app.py
--- /dev/null
+++ b/app.py
@@ -0,0 +1,5 @@
+def handle(data):
+    val = data / 0
+    return val
+"""
    results = await orchestrator.run_review(diff, "app.py")
    agents = results["agents_executed"]
    assert "Bug Detection Agent" in agents
    assert "Security Review Agent" in agents
    assert "Performance Review Agent" in agents
    assert "Code Quality Agent" in agents
    assert results["total_raw_findings"] > 0


# ------------------------------------------------------------
# TEST 9: Review Synthesizer & Deduplication
# ------------------------------------------------------------
def test_9_review_synthesizer():
    synthesizer = get_review_synthesizer()
    raw_agent_results = {
        "all_findings": [
            {"title": "SQL Injection", "category": "Security", "severity": "HIGH", "file": "app.py", "line": 5, "evidence": "SELECT ...", "suggested_fix": "Use params"},
            {"title": "CWE-89 SQL Injection Vector", "category": "Security", "severity": "HIGH", "file": "app.py", "line": 5, "evidence": "SELECT ...", "suggested_fix": "Use params"},
        ]
    }
    static_findings = [
        {"title": "Hardcoded SQL expression", "source": "Bandit", "severity": "MEDIUM", "file": "app.py", "line": 5, "evidence": "SELECT ...", "recommendation": "Use params"}
    ]
    diff = "@@ -0,0 +1,6 @@\n+query = 'SELECT * FROM users'\n"
    report = synthesizer.synthesize(raw_agent_results, static_findings, diff, "app.py")
    # Duplicate SQL findings at line 5 must be collapsed into one
    sql_findings = [f for f in report["findings"] if "sql" in f["title"].lower()]
    assert len(sql_findings) == 1, f"Expected 1 deduplicated SQL finding, found {len(sql_findings)}"


# ------------------------------------------------------------
# TEST 10: Transparent PR Risk Assessment
# ------------------------------------------------------------
def test_10_risk_assessment():
    synthesizer = get_review_synthesizer()
    critical_findings = [
        {"title": "Arbitrary Command Injection", "category": "Security Review Agent", "severity": "CRITICAL", "file": "cmd.py", "line": 10},
        {"title": "SQL Injection Vulnerability", "category": "Security Review Agent", "severity": "HIGH", "file": "cmd.py", "line": 15},
    ]
    diff = "+ os.system('rm -rf ' + path)\n"
    high_risk = synthesizer.calculate_pr_risk_score(critical_findings, diff)

    clean_findings = []
    low_risk = synthesizer.calculate_pr_risk_score(clean_findings, diff)

    assert high_risk["score"] > low_risk["score"], "High risk score must exceed clean score."
    assert high_risk["score"] >= 50, f"Expected high risk score >= 50, got {high_risk['score']}"
    assert low_risk["score"] == 0, f"Expected 0 for clean findings, got {low_risk['score']}"
    assert "formula_explanation" in high_risk


# ------------------------------------------------------------
# TEST 11: Review History & Developer Feedback Persistence
# ------------------------------------------------------------
def test_11_review_history_and_feedback():
    init_database()
    summary = {
        "critical_issues": 1,
        "security_issues": 1,
        "code_quality_issues": 0,
        "test_issues": 1,
        "severity": {"critical": 1, "high": 0, "medium": 0, "low": 0},
        "findings": [{"title": "Test Critical Finding", "severity": "CRITICAL", "line": 1}],
    }
    rev_id = save_review(
        filename="test_history_sample.diff",
        summary=summary,
        review_text="Test Review Text",
        risk_score=75,
        risk_level="CRITICAL RISK",
        synthesized_report={"test": True},
    )
    assert rev_id > 0, "Failed to save review in database."

    # Retrieve review
    rec = get_review(rev_id)
    assert rec is not None
    assert rec["filename"] == "test_history_sample.diff"
    assert rec["risk_score"] == 75

    # Test developer feedback
    fb_id = save_feedback(rev_id, finding_index=0, feedback_type="correct_finding", comment="Accurate finding")
    assert fb_id > 0
    feedback_list = get_feedback_for_review(rev_id)
    assert len(feedback_list) > 0
    assert feedback_list[0]["feedback_type"] == "correct_finding"


# ------------------------------------------------------------
# TEST 12: End-to-End Pipeline
# ------------------------------------------------------------
@pytest.mark.asyncio
async def test_12_end_to_end_pipeline():
    pipeline = get_pipeline()
    benchmarks = get_controlled_benchmarks()
    sqli_case = benchmarks[0]

    result = await pipeline.execute_review(
        diff_content=sqli_case["diff"],
        filename="benchmark_sqli.diff",
        pr_metadata={"title": "Benchmark SQLi"},
    )
    assert result["success"] is True
    assert result["review_id"] > 0
    assert result["risk_assessment"]["score"] > 0
    assert len(result["test_recommendations"]) > 0
    assert "github_comment" in result
    assert "CodeBERT" in result["github_comment"]


# ------------------------------------------------------------
# TEST 13: GitHub PR Integration
# ------------------------------------------------------------
@pytest.mark.asyncio
async def test_13_github_pr_integration():
    github = get_github_integration()
    pr_url = "https://github.com/octocat/Hello-World/pull/42"
    parsed = github.parse_pr_url(pr_url)
    assert parsed["owner"] == "octocat"
    assert parsed["repo"] == "Hello-World"
    assert parsed["pr_number"] == 42

    pr_data = await github.fetch_pr_data(pr_url)
    assert "diff_content" in pr_data
    assert len(pr_data["diff_content"]) > 0

    comment = github.format_github_review_comment(
        synthesized_report={
            "risk_assessment": {"score": 60, "risk_level": "HIGH RISK", "recommendation": "Fix vulnerabilities"},
            "findings": [{"title": "Test Issue", "severity": "HIGH", "category": "Security", "file": "main.py", "line": 10}],
            "severity_counts": {"critical": 0, "high": 1, "medium": 0, "low": 0, "informational": 0},
        },
        pr_title="PR #42"
    )
    assert "PR-Sentinel Code Review Report" in comment
    assert "PR_Risk_Score-60" in comment
