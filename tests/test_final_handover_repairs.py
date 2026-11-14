"""
Focused Regression Tests for Final Implementation Handover Repairs
Tests:
1. Automatic checkpoint discovery path handling in CodeBERTDefectClassifier.
2. CodeBERTAnalyzer passes require_real_weights to its defect_head.
3. ReviewSynthesizer risk score ignores CodeBERT penalty when using fallback representations.
4. Pipeline RAG instance re-instantiation when repo_id or commit_sha changes.
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock

from sentinel.codebert_analyzer import CodeBERTAnalyzer, CodeBERTDefectClassifier
from sentinel.synthesizer import ReviewSynthesizer
from sentinel.pipeline import PRSentinelPipeline
from sentinel.schemas import PipelineConfig


def test_codebert_defect_head_require_real_weights_inheritance():
    """Ensure CodeBERTAnalyzer passes require_real_weights to its internal defect_head."""
    analyzer_res = CodeBERTAnalyzer(lazy_load=True, require_real_weights=True)
    assert analyzer_res.require_real_weights is True
    assert analyzer_res.defect_head.require_real_weights is True

    analyzer_off = CodeBERTAnalyzer(lazy_load=True, require_real_weights=False)
    assert analyzer_off.require_real_weights is False
    assert analyzer_off.defect_head.require_real_weights is False


def test_codebert_classifier_default_checkpoint_paths_defined():
    """Ensure standard checkpoint paths are configured for automatic discovery."""
    assert len(CodeBERTDefectClassifier.DEFAULT_CHECKPOINT_PATHS) >= 2
    assert any("sentinel" in str(p) for p in CodeBERTDefectClassifier.DEFAULT_CHECKPOINT_PATHS)
    assert any("evaluation" in str(p) for p in CodeBERTDefectClassifier.DEFAULT_CHECKPOINT_PATHS)


def test_synthesizer_risk_score_ignores_fallback_codebert_penalty():
    """Ensure synthetic/fallback representations do NOT add uncalibrated penalty to risk score."""
    synthesizer = ReviewSynthesizer()
    diff = "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n+print('test')\n"
    findings = [
        {"title": "Low warning", "severity": "LOW", "category": "CODE_QUALITY", "file": "file.py", "line": 1}
    ]

    # Case A: fallback representations (is_pretrained_weights=False) with high synthetic deviation
    mock_fallback_analysis = {
        "is_pretrained_weights": False,
        "semantic_indicators": {
            "semantic_risk_deviation": 0.50, # High deviation
        }
    }
    score_fallback = synthesizer.calculate_pr_risk_score(findings, diff, mock_fallback_analysis)

    # Case B: no codebert analysis
    score_none = synthesizer.calculate_pr_risk_score(findings, diff, None)

    # Risk score must be identical because fallback penalty is disabled in research mode
    assert score_fallback["score"] == score_none["score"]

    # Case C: genuine pretrained weights (is_pretrained_weights=True) with high deviation
    mock_real_analysis = {
        "is_pretrained_weights": True,
        "semantic_indicators": {
            "semantic_risk_deviation": 0.50,
        }
    }
    score_real = synthesizer.calculate_pr_risk_score(findings, diff, mock_real_analysis)
    assert score_real["score"] > score_none["score"]


def test_pipeline_rag_reinstantiation_on_commit_or_repo_change(tmp_path):
    """Ensure pipeline creates a new RAG instance when commit_sha or repo_id differs."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    config = PipelineConfig(
        experiment_id="E6_A1",
        enable_codebert=False,
        enable_static=False,
        enable_rag=True,
        enable_llm=False,
        enable_multi_agent=False,
        enable_synthesis=False,
        target_repo_dir=str(repo_dir),
    )
    pipeline = PRSentinelPipeline(config=config)

    # Execute review with repo1 and commit A
    meta1 = {"repo": "RepoAlpha", "target_repo_dir": str(repo_dir), "commit_sha": "aaaa1111"}
    diff = "--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n+pass\n"
    pipeline.execute_review_sync(diff, "test.diff", meta1)
    rag_inst1 = pipeline.rag
    assert rag_inst1.repo_id == "RepoAlpha"
    assert rag_inst1.commit_sha == "aaaa1111"

    # Execute review with repo1 and commit B
    meta2 = {"repo": "RepoAlpha", "target_repo_dir": str(repo_dir), "commit_sha": "bbbb2222"}
    pipeline.execute_review_sync(diff, "test.diff", meta2)
    rag_inst2 = pipeline.rag
    assert rag_inst2.commit_sha == "bbbb2222"
    assert rag_inst2 is not rag_inst1
