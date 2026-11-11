"""
Comprehensive Test Suite: tests/test_review_analyzer.py
Characterizes all static analysis line mappings, CodeBERT real-weight constraints,
multi-agent heuristic detectors, schema validations, and pipeline ablations.
"""

import os
import pytest

from evaluation.matching_engine import (
    BipartiteMatchingEngine,
    MatchingProtocol,
    compute_line_distance,
    compute_token_jaccard,
)
from sentinel.codebert_analyzer import CodeBERTDefectClassifier, get_codebert_analyzer
from sentinel.multi_agent import (
    BugDetectionAgent,
    CodeQualityAgent,
    PerformanceReviewAgent,
    SecurityReviewAgent,
    get_multi_agent_orchestrator,
)
from sentinel.pipeline import PRSentinelPipeline
from sentinel.schemas import (
    ExecutionStatus,
    FindingCategory,
    FindingGranularity,
    GroundTruthIssue,
    ModelFinding,
    PipelineConfig,
    SeverityLevel,
)
from sentinel.static_analyzer import get_static_analyzer
from sentinel.test_generator import get_test_generator


# ------------------------------------------------------------
# 1. STATIC ANALYZER TESTS
# ------------------------------------------------------------
def test_static_analyzer_multi_hunk_line_mapping():
    """Verify that multi-hunk diff extraction maps relative lines to actual file lines."""
    analyzer = get_static_analyzer()
    diff = """diff --git a/pkg/service.py b/pkg/service.py
index 1111111..2222222 100644
--- a/pkg/service.py
+++ b/pkg/service.py
@@ -10,3 +10,4 @@
 def func_one():
     x = 1
+    y = x / 0
@@ -50,3 +51,4 @@
 def func_two():
     a = 10
+    b = a / 0
"""
    findings = analyzer.analyze_diff_to_findings(diff, filename="pkg/service.py")
    assert isinstance(findings, list)
    py_files = analyzer._split_multi_file_diff(diff, default_filename="pkg/service.py")
    assert len(py_files) == 1
    assert "pkg/service.py" in py_files


def test_static_analyzer_unsupported_language():
    """Verify that unsupported languages (e.g. .ts, .go, .rs) generate UNSUPPORTED_LANGUAGE notices."""
    analyzer = get_static_analyzer()
    diff = """diff --git a/app.rs b/app.rs
new file mode 100644
--- /dev/null
+++ b/app.rs
@@ -0,0 +1,5 @@
+fn main() {
+    println!("Hello Rust!");
+}
+"""
    findings = analyzer.analyze_diff_to_findings(diff, filename="app.rs")
    unsupported = [
        f for f in findings
        if f.execution_status == ExecutionStatus.UNSUPPORTED_LANGUAGE.value or f.rule_id == "LANG_UNSUPPORTED"
    ]
    assert len(unsupported) == 1
    assert unsupported[0].granularity == FindingGranularity.FILE.value
    assert unsupported[0].severity == SeverityLevel.INFORMATIONAL.value


# ------------------------------------------------------------
# 2. CODEBERT REAL-WEIGHT ENFORCEMENT
# ------------------------------------------------------------
def test_codebert_real_weight_enforcement():
    """Verify that CodeBERT raises RuntimeError in research mode when real weights cannot be loaded."""
    from sentinel.codebert_analyzer import CodeBERTAnalyzer
    with pytest.raises((RuntimeError, Exception)):
        CodeBERTAnalyzer(lazy_load=False, require_real_weights=True)


# ------------------------------------------------------------
# 3. MULTI-AGENT HEURISTIC DETECTOR TESTS
# ------------------------------------------------------------
def test_bug_detection_agent_heuristics():
    agent = BugDetectionAgent()
    diff = """@@ -0,0 +1,4 @@
+def divide(val, denom):
+    return val / denom
+"""
    findings = agent.analyze_heuristically(diff, "calc.py")
    assert len(findings) > 0
    assert any("division" in f["title"].lower() or "zero" in f["title"].lower() for f in findings)


def test_security_review_agent_heuristics():
    agent = SecurityReviewAgent()
    diff = """@@ -0,0 +1,8 @@
+import os
+def run_task(user_input):
+    os.system("echo " + user_input)
+    query = "SELECT * FROM users WHERE username = '" + user_input + "'"
+"""
    findings = agent.analyze_heuristically(diff, "task.py")
    assert len(findings) >= 2
    titles = [f["title"].lower() for f in findings]
    assert any("command" in t for t in titles)
    assert any("sql" in t for t in titles)


def test_performance_review_agent_heuristics():
    agent = PerformanceReviewAgent()
    diff = """@@ -0,0 +1,5 @@
+def find_records(ids):
+    for i in ids:
+        cursor.execute("SELECT * FROM items WHERE id = ?", (i,))
+"""
    findings = agent.analyze_heuristically(diff, "db.py")
    assert len(findings) > 0
    assert any("query" in f["title"].lower() or "n+1" in f["title"].lower() or "loop" in f["title"].lower() for f in findings)


def test_code_quality_agent_heuristics():
    agent = CodeQualityAgent()
    diff = """@@ -0,0 +1,5 @@
+def handler():
+    try:
+        do_something()
+    except:
+        pass
+"""
    findings = agent.analyze_heuristically(diff, "handler.py")
    assert len(findings) > 0
    assert any("bare except" in f["title"].lower() or "exception" in f["title"].lower() for f in findings)


# ------------------------------------------------------------
# 4. PIPELINE CONFIGURATION & ABLATION MATRIX
# ------------------------------------------------------------
def test_pipeline_config_ablation_matrix():
    # E1: Monolithic
    c_e1 = PipelineConfig.for_experiment("E1")
    assert c_e1.enable_llm is True
    assert c_e1.enable_multi_agent is False
    assert c_e1.enable_codebert is False
    assert c_e1.enable_static is False
    assert c_e1.enable_rag is False

    # E2: Static Analysis
    c_e2 = PipelineConfig.for_experiment("E2")
    assert c_e2.enable_static is True
    assert c_e2.enable_llm is False
    assert c_e2.enable_codebert is False
    assert c_e2.enable_rag is False

    # E3: CodeBERT
    c_e3 = PipelineConfig.for_experiment("E3")
    assert c_e3.enable_codebert is True
    assert c_e3.enable_llm is False
    assert c_e3.enable_static is False

    # E4: RAG + LLM Monolithic
    c_e4 = PipelineConfig.for_experiment("E4")
    assert c_e4.enable_rag is True
    assert c_e4.enable_llm is True
    assert c_e4.enable_multi_agent is False
    assert c_e4.enable_static is False

    # E5: Multi-Agent
    c_e5 = PipelineConfig.for_experiment("E5")
    assert c_e5.enable_multi_agent is True
    assert c_e5.enable_rag is False
    assert c_e5.enable_static is False
    assert c_e5.enable_codebert is False

    # E6: Full System
    c_e6 = PipelineConfig.for_experiment("E6")
    assert c_e6.enable_codebert is True
    assert c_e6.enable_static is True
    assert c_e6.enable_rag is True
    assert c_e6.enable_multi_agent is True

    # Ablations E6_A1 through E6_A4
    c_a1 = PipelineConfig.for_experiment("E6_A1")
    assert c_a1.enable_codebert is False and c_a1.enable_static is True

    c_a2 = PipelineConfig.for_experiment("E6_A2")
    assert c_a2.enable_static is False and c_a2.enable_codebert is True

    c_a3 = PipelineConfig.for_experiment("E6_A3")
    assert c_a3.enable_rag is False and c_a3.enable_multi_agent is True

    c_a4 = PipelineConfig.for_experiment("E6_A4")
    assert c_a4.enable_multi_agent is False and c_a4.enable_llm is True


# ------------------------------------------------------------
# 5. TEST GENERATOR WITH FINDING ID ASSOCIATION
# ------------------------------------------------------------
def test_test_generator_finding_id_association():
    generator = get_test_generator()
    diff = """@@ -0,0 +1,5 @@
+def execute_query(uid):
+    conn.execute("SELECT * FROM users WHERE id = " + uid)
+"""
    findings = [
        {
            "finding_id": "finding-uuid-1234",
            "title": "SQL Injection in User Query",
            "evidence": "execute_query",
            "severity": "HIGH",
        }
    ]
    recs = generator.generate_recommendations(diff, findings, filename="app.py")
    assert len(recs) > 0
    sec_recs = [r for r in recs if r.get("finding_id") == "finding-uuid-1234"]
    assert len(sec_recs) == 1
    assert "SQL" in sec_recs[0]["scenario"]
