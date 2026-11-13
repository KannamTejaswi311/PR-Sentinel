"""
PR-Sentinel Prompt 2 Verification & Integration Tests
Tests all critical repairs:
- CodeBERT classification architecture, dataset loading, Youden's J calibration, hunk findings.
- Pipeline ablation configurations (E1, E2, E3, E4, E5, E6, E6_A1..A4).
- Target repository RAG isolation and rejection of unconfigured workspace.
- Manual audit rubric (CAT-A FP, CAT-B TP, CAT-C Nitpick), proportional sampling, sample adequacy.
- RAG evaluation: DFOR@K coverage proxy and PENDING_HUMAN_ANNOTATION status.
"""

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import pandas as pd

from sentinel.codebert_analyzer import (
    CodeBERTClassificationHead,
    CodeBERTDefectClassifier,
    get_codebert_analyzer,
    HAS_TORCH,
)
from sentinel.schemas import PipelineConfig, ExecutionStatus, ModelFinding
from sentinel.pipeline import PRSentinelPipeline
from sentinel.rag_pipeline import RepositoryRAGPipeline, get_rag_pipeline
from evaluation.sample_audit import (
    AUDIT_RUBRIC,
    calculate_adjusted_precision,
    sample_candidates,
)
from evaluation.evaluate_rag import (
    evaluate_rag_retrieval,
    extract_files_from_diff,
    compute_dcg_at_k,
    compute_ndcg_at_k,
)


SAMPLE_DIFF = """diff --git a/sentinel/auth.py b/sentinel/auth.py
index 1234567..89abcdef 100644
--- a/sentinel/auth.py
+++ b/sentinel/auth.py
@@ -10,6 +10,8 @@ def authenticate(user, password):
-    query = f"SELECT * FROM users WHERE user = '{user}'"
-    return db.execute(query)
+    query = "SELECT * FROM users WHERE user = %s"
+    return db.execute(query, (user,))
"""


# =====================================================================
# 1. CODEBERT TESTS
# =====================================================================

def test_codebert_architecture():
    """Verify CodeBERT classification head has 1536 -> 256 -> 2 dimensions."""
    head = CodeBERTClassificationHead(input_dim=1536, hidden_dim=256, num_classes=2, dropout_rate=0.1)
    if HAS_TORCH:
        import torch.nn as nn
        assert isinstance(head.dense, nn.Linear)
        assert head.dense.in_features == 1536
        assert head.dense.out_features == 256
        assert isinstance(head.out_proj, nn.Linear)
        assert head.out_proj.in_features == 256
        assert head.out_proj.out_features == 2
    else:
        # Fallback stub preserves dimensions
        assert head.input_dim == 1536
        assert head.hidden_dim == 256
        assert head.num_classes == 2


def test_codebert_representation_dimensions():
    """Verify representation returns 768-dim pooled and 1536-dim concatenated representation."""
    analyzer = CodeBERTDefectClassifier(require_real_weights=False)
    rep = analyzer.extract_code_representation("def hello_world(): return 42")
    assert "vector" in rep
    assert len(rep["vector"]) == 768
    assert "concat_cls_mean" in rep
    assert len(rep["concat_cls_mean"]) == 1536


def test_codebert_untrained_rejection():
    """Verify untrained classifier rejects inference with RuntimeError in strict mode."""
    analyzer = CodeBERTDefectClassifier(require_real_weights=False)
    analyzer.is_trained = False
    analyzer.has_calibrated_threshold = False
    with pytest.raises(RuntimeError) as exc_info:
        analyzer.predict("def test(): pass")
    assert "Untrained CodeBERT classifier" in str(exc_info.value)


def test_codebert_research_mode_rejection():
    """Verify research mode rejects synthetic vectors when real weights are missing."""
    try:
        analyzer = CodeBERTDefectClassifier(require_real_weights=True)
        if not analyzer.has_real_weights:
            findings = analyzer.analyze_diff_hunks(SAMPLE_DIFF, pr_id="test_pr")
            assert len(findings) == 1
            assert "TOOL_ERROR" in findings[0].execution_status or "weights failed to load" in findings[0].description
    finally:
        get_codebert_analyzer(require_real_weights=False)


def test_codebert_hunk_findings_schema():
    """Verify analyze_diff_hunks generates ModelFinding with hunk line attribution."""
    analyzer = CodeBERTDefectClassifier(require_real_weights=False)
    analyzer.is_trained = True
    analyzer.has_calibrated_threshold = True
    analyzer.calibrated_threshold = 0.5
    analyzer.mock_defective = True

    findings = analyzer.analyze_diff_hunks(SAMPLE_DIFF, pr_id="test_pr_123", default_filename="auth.py")
    assert len(findings) == 1
    f = findings[0]
    assert f.pr_id == "test_pr_123"
    assert f.granularity == "HUNK_SPAN"
    assert f.line_start == 10
    assert "auth.py" in f.file_path
    assert f.confidence == 0.85


def test_codebert_dataset_loading():
    """Verify CodeXGLUE defect dataset loader handles valid JSONL and filters properly."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"idx": 0, "func": "def clean(): pass", "target": 0}) + "\n")
        f.write(json.dumps({"idx": 1, "func": "def bug(): crash()", "target": 1}) + "\n")
        f.write("invalid json line\n")
        tmp_path = f.name

    try:
        samples = CodeBERTDefectClassifier.load_codexglue_dataset(tmp_path, max_samples=10)
        assert len(samples) == 2
        assert samples[0]["target"] == 0
        assert samples[1]["target"] == 1
    finally:
        os.remove(tmp_path)


def test_codebert_youden_j_calibration():
    """Verify threshold calibration implements Youden's J optimization: J = TPR - FPR."""
    analyzer = CodeBERTDefectClassifier(require_real_weights=False)
    # Val data: 3 non-defective (probs: 0.1, 0.2, 0.4), 3 defective (probs: 0.6, 0.8, 0.9)
    val_data = [
        {"func": "clean1", "target": 0},
        {"func": "clean2", "target": 0},
        {"func": "clean3", "target": 0},
        {"func": "bug1", "target": 1},
        {"func": "bug2", "target": 1},
        {"func": "bug3", "target": 1},
    ]

    prob_map = {
        "clean1": 0.1,
        "clean2": 0.2,
        "clean3": 0.4,
        "bug1": 0.6,
        "bug2": 0.8,
        "bug3": 0.9,
    }

    def mock_predict(code):
        return {"defect_probability": prob_map.get(code, 0.5), "is_defective": False}

    with patch.object(analyzer, "predict", side_effect=mock_predict):
        calib = analyzer.calibrate_threshold(val_data)
        assert "optimal_threshold" in calib
        assert "youden_j" in calib
        # Optimal threshold should be between 0.4 and 0.6, yielding TPR=1.0, FPR=0.0 -> J=1.0
        assert 0.4 <= calib["optimal_threshold"] <= 0.6
        assert calib["youden_j"] == 1.0
        assert analyzer.has_calibrated_threshold is True


# =====================================================================
# 2. PIPELINE ABLATION CONFIGURATIONS
# =====================================================================

@pytest.mark.parametrize("config_id", [
    "E1", "E2", "E3", "E4", "E5", "E6",
    "E6_A1", "E6_A2", "E6_A3", "E6_A4",
])
def test_all_ablation_configurations_instantiate_and_run(config_id):
    """Verify all 8+ ablation configurations instantiate and execute without crashing."""
    config = PipelineConfig.for_experiment(config_id)
    pipeline = PRSentinelPipeline(config=config)
    result = pipeline.execute_review_sync(
        diff_content=SAMPLE_DIFF,
        filename="sentinel/auth.py",
        pr_id=f"test_{config_id}",
    )
    assert result["success"] is True
    assert result["experiment_id"] == config_id
    manifest = result["execution_manifest"]
    assert isinstance(manifest, dict)


def test_e1_monolithic_review_routing():
    """Verify E1 executes monolithic review, NOT multi-agent."""
    config = PipelineConfig.for_experiment("E1")
    pipeline = PRSentinelPipeline(config=config)
    result = pipeline.execute_review_sync(
        diff_content=SAMPLE_DIFF,
        filename="sentinel/auth.py",
        pr_id="test_e1",
    )
    manifest = result["execution_manifest"]
    assert "MONOLITHIC_MODE" in manifest["multi_agent"]
    assert manifest["codebert"] == ExecutionStatus.SKIPPED.value
    assert manifest["rag_pipeline"] == ExecutionStatus.SKIPPED.value
    assert manifest["static_analysis"] == ExecutionStatus.SKIPPED.value


def test_e2_static_only():
    """Verify E2 executes ONLY static analysis."""
    config = PipelineConfig.for_experiment("E2")
    pipeline = PRSentinelPipeline(config=config)
    result = pipeline.execute_review_sync(
        diff_content=SAMPLE_DIFF,
        filename="sentinel/auth.py",
        pr_id="test_e2",
    )
    manifest = result["execution_manifest"]
    assert manifest["static_analysis"] == ExecutionStatus.SUCCESS.value
    assert manifest["codebert"] == ExecutionStatus.SKIPPED.value
    assert manifest["rag_pipeline"] == ExecutionStatus.SKIPPED.value
    assert manifest["multi_agent"] == ExecutionStatus.SKIPPED.value


def test_e3_codebert_only():
    """Verify E3 executes ONLY CodeBERT."""
    config = PipelineConfig.for_experiment("E3")
    config.require_real_weights = False
    pipeline = PRSentinelPipeline(config=config)
    result = pipeline.execute_review_sync(
        diff_content=SAMPLE_DIFF,
        filename="sentinel/auth.py",
        pr_id="test_e3",
    )
    manifest = result["execution_manifest"]
    assert manifest["codebert"] == ExecutionStatus.SUCCESS.value
    assert manifest["static_analysis"] == ExecutionStatus.SKIPPED.value
    assert manifest["rag_pipeline"] == ExecutionStatus.SKIPPED.value
    assert manifest["multi_agent"] == ExecutionStatus.SKIPPED.value


def test_e4_monolithic_plus_rag():
    """Verify E4 executes monolithic review + RAG, but NO multi-agent and NO codebert/static."""
    config = PipelineConfig.for_experiment("E4")
    pipeline = PRSentinelPipeline(config=config)
    result = pipeline.execute_review_sync(
        diff_content=SAMPLE_DIFF,
        filename="sentinel/auth.py",
        pr_id="test_e4",
    )
    manifest = result["execution_manifest"]
    assert "MONOLITHIC_MODE" in manifest["multi_agent"]
    assert manifest["codebert"] == ExecutionStatus.SKIPPED.value
    assert manifest["static_analysis"] == ExecutionStatus.SKIPPED.value


def test_e5_multi_agent_without_rag_codebert():
    """Verify E5 executes multi-agent, but NO RAG and NO CodeBERT."""
    config = PipelineConfig.for_experiment("E5")
    pipeline = PRSentinelPipeline(config=config)
    result = pipeline.execute_review_sync(
        diff_content=SAMPLE_DIFF,
        filename="sentinel/auth.py",
        pr_id="test_e5",
    )
    manifest = result["execution_manifest"]
    assert manifest["multi_agent"] == ExecutionStatus.SUCCESS.value
    assert manifest["codebert"] == ExecutionStatus.SKIPPED.value
    assert manifest["rag_pipeline"] == ExecutionStatus.SKIPPED.value


def test_e6_a4_monolithic_ablation():
    """Verify E6_A4 executes monolithic review with full pipeline (RAG + static + CodeBERT)."""
    config = PipelineConfig.for_experiment("E6_A4")
    config.require_real_weights = False
    pipeline = PRSentinelPipeline(config=config)
    result = pipeline.execute_review_sync(
        diff_content=SAMPLE_DIFF,
        filename="sentinel/auth.py",
        pr_id="test_e6_a4",
    )
    manifest = result["execution_manifest"]
    assert "MONOLITHIC_MODE" in manifest["multi_agent"]
    assert manifest["codebert"] == ExecutionStatus.SUCCESS.value
    assert manifest["static_analysis"] == ExecutionStatus.SUCCESS.value


# =====================================================================
# 3. RAG TARGET REPOSITORY ISOLATION
# =====================================================================

def test_rag_unconfigured_workspace():
    """Verify RAG is marked unavailable / skipped when no target workspace is provided."""
    rag = RepositoryRAGPipeline(workspace_dir=None)
    assert rag.is_available is False
    stats = rag.get_stats()
    assert stats["is_available"] is False
    assert stats["is_indexed"] is False

    # Check indexing skips cleanly without indexing PR-Sentinel host
    idx_res = rag.index_repository()
    assert idx_res["is_indexed"] is False
    assert idx_res["status"] == "SKIPPED_NO_TARGET_WORKSPACE"


def test_rag_target_isolation():
    """Verify two different target repositories produce two distinct index file paths."""
    with tempfile.TemporaryDirectory() as repo_a, tempfile.TemporaryDirectory() as repo_b:
        rag_a = RepositoryRAGPipeline(workspace_dir=repo_a, repo_id="repo_alpha", commit_sha="aaa111")
        rag_b = RepositoryRAGPipeline(workspace_dir=repo_b, repo_id="repo_beta", commit_sha="bbb222")

        assert rag_a.index_path != rag_b.index_path
        assert "repo_alpha" in str(rag_a.index_path)
        assert "repo_beta" in str(rag_b.index_path)


def test_rag_provenance_recording():
    """Verify indexed chunks carry repo_id and commit_sha provenance."""
    with tempfile.TemporaryDirectory() as repo_dir:
        test_file = Path(repo_dir) / "service.py"
        test_file.write_text("def run_query(): return True\n", encoding="utf-8")

        rag = RepositoryRAGPipeline(workspace_dir=repo_dir, repo_id="custom_repo", commit_sha="c0ffee1234")
        rag.index_repository(force_reindex=True)

        assert len(rag.chunks) > 0
        chunk = rag.chunks[0]
        assert chunk["repo_id"] == "custom_repo"
        assert chunk["commit_sha"] == "c0ffee1234"
        assert chunk["file_path"] == "service.py"


def test_rag_research_mode_rejects_synthetic_embeddings():
    """Verify RAG indexing fails with RuntimeError in research mode if real CodeBERT weights missing."""
    with tempfile.TemporaryDirectory() as repo_dir:
        test_file = Path(repo_dir) / "app.py"
        test_file.write_text("x = 1\n", encoding="utf-8")

        rag = RepositoryRAGPipeline(workspace_dir=repo_dir, repo_id="sec_repo")
        try:
            rag.codebert.require_real_weights = True
            with patch.object(type(rag.codebert), "has_real_weights", False), patch.object(rag.codebert, "model", None):
                with pytest.raises(RuntimeError) as exc_info:
                    rag.index_repository(force_reindex=True)
                assert "strictly forbidden" in str(exc_info.value)
        finally:
            rag.codebert.require_real_weights = False


# =====================================================================
# 4. MANUAL AUDIT RUBRIC & PROPORTIONAL STRATIFIED SAMPLING
# =====================================================================

def test_manual_audit_rubric_definitions():
    """Verify exact rubric categories: CAT-A (FP), CAT-B (TP), CAT-C (Nitpick)."""
    assert "Confirmed False Positive" in AUDIT_RUBRIC["CAT-A"] or "False alarm" in AUDIT_RUBRIC["CAT-A"]
    assert "Reclassified True Positive" in AUDIT_RUBRIC["CAT-B"] or "Genuine defect" in AUDIT_RUBRIC["CAT-B"]
    assert "Nitpick" in AUDIT_RUBRIC["CAT-C"] or "Stylistic suggestion" in AUDIT_RUBRIC["CAT-C"]


def test_adjusted_precision_formula():
    """Verify P_adj formula: (TP + rho_B * FP) / (TP + FP) where rho_B = n_B / (n_A + n_B + n_C)."""
    # 20 TP, 80 FP. Audit of 50 samples: 30 CAT-A (FP), 15 CAT-B (TP), 5 CAT-C (Nitpick)
    # rho_B = 15 / 50 = 0.30
    # P_adj = (20 + 0.30 * 80) / (20 + 80) = 44 / 100 = 0.44
    res = calculate_adjusted_precision(
        tp_benchmark=20,
        fp_benchmark=80,
        n_A=30,
        n_B=15,
        n_C=5,
    )
    assert res["rho_B_latent_rate"] == 0.30
    assert res["apparent_precision"] == 0.20
    assert res["adjusted_precision"] == 0.44
    assert "Not a mathematically guaranteed lower bound" in res["precision_interpretation"]


def test_proportional_stratified_sampling():
    """Verify sample_candidates uses true proportional stratification across strata."""
    fps = []
    # Create 80 High severity, 20 Low severity (4:1 ratio)
    for i in range(80):
        fps.append({"finding_id": f"h_{i}", "severity": "HIGH", "category": "security"})
    for i in range(20):
        fps.append({"finding_id": f"l_{i}", "severity": "LOW", "category": "style"})

    sampled = sample_candidates(fps, n_samples=50, seed=42)
    assert len(sampled) == 50
    high_count = sum(1 for s in sampled if s["severity"] == "HIGH")
    low_count = sum(1 for s in sampled if s["severity"] == "LOW")

    # In proportional allocation of 50: High should be 40 (80%), Low should be 10 (20%)
    assert high_count == 40
    assert low_count == 10


def test_sample_adequacy_shortfall_flagging():
    """Verify sample shortfall is honestly flagged when candidate count is below minimum."""
    from evaluation.sample_audit import generate_audit_samples
    with tempfile.TemporaryDirectory() as tmp_audit:
        manifest = generate_audit_samples(
            audit_dir=Path(tmp_audit),
            tier1_size=50,
            tier2_per_config=15,
            seed=42,
        )
        assert "tier1_sample_adequacy" in manifest
        # If no E6 predictions exist in repo raw results, it must flag INSUFFICIENT honestly
        if manifest["tier1_e6_total_fps"] < 50:
            assert "INSUFFICIENT" in manifest["tier1_sample_adequacy"]
        else:
            assert manifest["tier1_sample_adequacy"] == "SUFFICIENT"


# =====================================================================
# 5. RAG EVALUATION & RELEVANCE ANNOTATION
# =====================================================================

def test_extract_files_from_diff():
    """Verify diff parser extracts modified file paths."""
    files = extract_files_from_diff(SAMPLE_DIFF)
    assert "sentinel/auth.py" in files


def test_dfor_dcg_ndcg_computation():
    """Verify DCG and nDCG computation."""
    scores = [2.0, 1.0, 0.0]
    dcg = compute_dcg_at_k(scores, k=3)
    ndcg = compute_ndcg_at_k(scores, k=3)
    assert dcg > 0.0
    assert ndcg == 1.0  # Perfect ordering -> nDCG is 1.0


def test_rag_pending_human_annotation_status():
    """Verify semantic MRR and nDCG are reported as PENDING_HUMAN_ANNOTATION when annotations absent."""
    with tempfile.TemporaryDirectory() as tmp_raw:
        # Create dummy configuration directory with a result file
        cfg_dir = Path(tmp_raw) / "E6"
        cfg_dir.mkdir(parents=True)
        res_file = cfg_dir / "pr_test_01.json"
        res_file.write_text(json.dumps({
            "rag_context": [{"file_path": "sentinel/auth.py", "chunk_id": "c1"}],
            "findings": [],
        }), encoding="utf-8")

        summary = evaluate_rag_retrieval(
            config_id="E6",
            raw_dir=Path(tmp_raw),
            manifest_path=Path("nonexistent_manifest.csv"),
            annotations_path=Path("nonexistent_annotations.csv"),
        )

        assert "dfor_coverage_proxy@1" in summary
        assert summary["semantic_mrr"] == "PENDING_HUMAN_ANNOTATION"
        assert summary["semantic_ndcg@1"] == "PENDING_HUMAN_ANNOTATION"
        assert summary["semantic_annotation_status"] == "PENDING_HUMAN_ANNOTATION"


def test_rag_annotation_template_exists():
    """Verify RAG human annotation template file exists with required schema."""
    template_path = Path("evaluation/datasets/rag_relevance_annotation_template.csv")
    assert template_path.exists()
    content = template_path.read_text(encoding="utf-8")
    assert "pr_id,chunk_id,file_path,retrieval_rank,query_type,relevance_score,annotator_id,notes" in content
