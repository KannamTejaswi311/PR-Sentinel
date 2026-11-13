# PR-Sentinel — Prompt 2 Final Status Report

**Document Version:** 1.0  
**Date:** November 13, 2026  
**Status:** All Prompt 2 Repairs Implemented & Verified (35/35 Tests Passing)  
**Methodological Protocol:** Frozen Experimental Protocol v1.0 & Amendment v1.1  

---

## 1. Files Modified and Created

### Modified Source & Evaluation Modules
1. [sentinel/codebert_analyzer.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/sentinel/codebert_analyzer.py)
   - Implemented supervised `CodeBERTClassificationHead` architecture (`1536 -> 256 -> 2`, GELU, Dropout 0.1).
   - Implemented `CodeBERTDefectClassifier` with `load_codexglue_dataset()` supporting JSONL splits and anti-leakage guards against evaluation benchmarks.
   - Implemented Youden's J calibration ($J = \text{TPR} - \text{FPR}$) optimizing decision threshold on validation data.
   - Implemented checkpoint persistence (`save_checkpoint`, `load_checkpoint`) with provenance metadata.
   - Implemented `analyze_diff_hunks()` producing `ModelFinding` records with `HUNK_SPAN` granularity and hunk line bounds.
   - Implemented strict research mode rejection of synthetic vectors and untrained inference.
2. [sentinel/multi_agent.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/sentinel/multi_agent.py)
   - Implemented `run_monolithic_review_sync()` enabling genuine monolithic single-prompt LLM execution for E1, E4, and E6-A4.
3. [sentinel/pipeline.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/sentinel/pipeline.py)
   - Integrated hunk-level CodeBERT findings ingestion into review synthesis.
   - Added target repository directory validation and base commit SHA isolation for RAG.
   - Routed monolithic review execution for `enable_llm=True` with `enable_multi_agent=False`.
   - Recorded standardized execution manifest statuses (`codebert`, `static_analyzer`, `static_analysis`, `rag_pipeline`, `multi_agent`).
4. [sentinel/synthesizer.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/sentinel/synthesizer.py)
   - Ingested CodeBERT hunk findings alongside static and multi-agent findings for unified deduplication and risk scoring.
5. [sentinel/rag_pipeline.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/sentinel/rag_pipeline.py)
   - Eliminated default fallback to PR-Sentinel host repository; requires explicit target workspace snapshot.
   - Added repository provenance tracking (`repo_id`, `commit_sha`, `file_path`) on all retrieved chunks.
   - Isolated persistent vector indexes by repository ID and commit SHA (`.sentinel_rag/{safe_repo}_{safe_sha}_rag_index.json`).
   - Implemented strict research mode rejection of synthetic embeddings when real CodeBERT weights are missing.
6. [evaluation/sample_audit.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/evaluation/sample_audit.py)
   - Restored frozen protocol audit rubric definitions:
     - **CAT-A**: Hallucination / Erroneous Finding (Confirmed False Positive).
     - **CAT-B**: Latent Valid Defect missed by benchmark ground truth (Reclassified True Positive).
     - **CAT-C**: Benign Quality Observation / Nitpick.
   - Replaced quadratic density weighting with true proportional stratified sampling across severity and repository strata.
   - Implemented honest sample shortfall reporting (`INSUFFICIENT (M=X < 50)`) without label fabrication.
   - Added `calculate_adjusted_precision()` computing $\hat{P}_{\text{adj}} = \frac{TP + \hat{\rho}_B \cdot FP_{\text{app}}}{TP + FP_{\text{app}}}$ with explicit non-lower-bound disclosure.
7. [evaluation/evaluate_rag.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/evaluation/evaluate_rag.py)
   - Enforced strict distinction between DFOR@K (diff-file overlap ratio, retrieval coverage proxy) and semantic relevance.
   - Configured semantic MRR and nDCG to report `PENDING_HUMAN_ANNOTATION` when human relevance judgments are absent.

### Created Assets & Test Suites
8. [evaluation/datasets/rag_relevance_annotation_template.csv](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/evaluation/datasets/rag_relevance_annotation_template.csv)
   - Standardized template for two-annotator graded relevance scoring (0 = Irrelevant, 1 = Contextually Helpful, 2 = Directly Relevant).
9. [tests/test_prompt2_repairs.py](file:///c:/Users/KannamTejaswi/Major%20Project/References/pr-agent/tests/test_prompt2_repairs.py)
   - Comprehensive 35-test unit and integration test suite verifying CodeBERT architecture, pipeline ablations, RAG isolation, audit rubric, and RAG evaluation metrics.

---

## 2. CodeBERT Pretrained Weights and Trained Classifier Status

- **Architecture Implementation:** **COMPLETE.** The two-layer classification head (`1536 -> 256 -> 2`, GELU activation, Dropout 0.1), feature concatenation (`[h_[CLS] || mean_pool]`), Youden's J calibration algorithm, and checkpoint serialization/deserialization are fully coded and tested.
- **Runtime Pretrained Weights:** **NOT LOADED IN ENVIRONMENT.** PyTorch is not installed in the Windows virtual environment (`ModuleNotFoundError: No module named 'torch'`). No trained weights file (`codebert_defect_head.pt`) exists on disk.
- **Research Safeguard:** **VERIFIED.** In research mode (`require_real_weights=True`), the system strictly rejects synthetic or heuristic fallback embeddings, raising `RuntimeError` or outputting `TOOL_ERROR` rather than silently generating invalid data. To execute genuine E3 experiments, PyTorch must be installed and the head weights trained on the CodeXGLUE defect detection dataset.

---

## 3. Monolithic LLM Invocation in E1, E4, and E6-A4

- **Status:** **CONFIRMED AND VERIFIED.**
- **Implementation:** In `sentinel/pipeline.py` (lines 144–167), when `config.enable_llm=True` and `config.enable_multi_agent=False`:
  - The pipeline calls `self.orchestrator.run_monolithic_review_sync()`.
  - A single comprehensive prompt is dispatched to the LLM (or mock offline reviewer) without invoking the 4 specialized agent roles.
  - Manifest records `"MONOLITHIC_MODE:SUCCESS"` and `agents_executed: ["Monolithic Reviewer"]`.
- **Test Evidence:**
  - `test_e1_monolithic_review_routing`: **PASSED** (Monolithic review invoked, CodeBERT/Static/RAG skipped).
  - `test_e4_monolithic_plus_rag`: **PASSED** (Monolithic review invoked + RAG enabled, CodeBERT/Static skipped).
  - `test_e6_a4_monolithic_ablation`: **PASSED** (Monolithic review invoked with full RAG + Static + CodeBERT pipeline).

---

## 4. Target Repository and PR Base Commit RAG Retrieval

- **Status:** **CONFIRMED AND VERIFIED.**
- **Implementation:**
  - `RepositoryRAGPipeline` in `sentinel/rag_pipeline.py` no longer defaults to indexing the PR-Sentinel host repository (`self.workspace_dir = None`).
  - If no target workspace is provided, `index_repository()` aborts with `"status": "SKIPPED_NO_TARGET_WORKSPACE"` and `pipeline.py` flags `"TOOL_ERROR: Target repository snapshot directory not specified or does not exist."`.
  - When target workspace and PR commit are provided, indexed chunks store `repo_id`, `commit_sha`, and `file_path`.
  - Indexes are strictly isolated on disk as `.sentinel_rag/{safe_repo}_{safe_sha}_rag_index.json` to prevent cross-repository or cross-commit contamination.
- **Test Evidence:**
  - `test_rag_unconfigured_workspace`: **PASSED**.
  - `test_rag_target_isolation`: **PASSED** (Two distinct repos generate separate index files).
  - `test_rag_provenance_recording`: **PASSED** (Indexed chunks contain `custom_repo` and `c0ffee1234`).
  - `test_rag_research_mode_rejects_synthetic_embeddings`: **PASSED**.

---

## 5. E6 Full Pipeline Execution

- **Status:** **CONFIRMED AND VERIFIED.**
- **Execution Workflow:**
  1. **STEP 1 (CodeBERT):** Analyzes diff hunks and extracts 768-D/1536-D representations and hunk-level `ModelFinding` records.
  2. **STEP 2 (Static Analysis):** Executes AST analysis, Bandit security rules, and Flake8 linter rules.
  3. **STEP 3 (Repository RAG):** Retrieves relevant chunks from the target repository snapshot.
  4. **STEP 4 (Multi-Agent Review):** Orchestrates all 4 specialized review agents:
     - Bug & Logic Reviewer
     - Security Specialist
     - Performance & Resource Reviewer
     - Code Quality & Clean Code Reviewer
  5. **STEP 5 (Review Synthesis):** Combines and deduplicates findings from CodeBERT, Static Analyzer, and all 4 agents into standardized `ModelFinding` entities and calculates PR risk score (0–100).
  6. **STEP 6 (Test Generation):** Produces automated test recommendations.
  7. **STEP 7 (Persistence):** Saves review history to SQLite.
- **Test Evidence:**
  - `test_all_ablation_configurations_instantiate_and_run[E6]`: **PASSED**.

---

## 6. Prompt 2 Repair Test Results

The prompt 2 verification test suite was executed via pytest:
```powershell
$env:PYTHONPATH="."
$env:PR_SENTINEL_OFFLINE="1"
.\.venv\Scripts\pytest.exe tests/test_prompt2_repairs.py -v -p no:asyncio -p no:anyio
```

### Exact Result: **35 passed, 0 failed, 2 warnings in 10.31s**

| Category | Test Function Name | Result |
| :--- | :--- | :--- |
| **CodeBERT** | `test_codebert_architecture` | **PASSED** |
| **CodeBERT** | `test_codebert_representation_dimensions` | **PASSED** |
| **CodeBERT** | `test_codebert_untrained_rejection` | **PASSED** |
| **CodeBERT** | `test_codebert_research_mode_rejection` | **PASSED** |
| **CodeBERT** | `test_codebert_hunk_findings_schema` | **PASSED** |
| **CodeBERT** | `test_codebert_dataset_loading` | **PASSED** |
| **CodeBERT** | `test_codebert_youden_j_calibration` | **PASSED** |
| **Pipeline Ablations** | `test_all_ablation_configurations_instantiate_and_run[E1..E6_A4]` (10 configs) | **PASSED** |
| **Pipeline Routing** | `test_e1_monolithic_review_routing` | **PASSED** |
| **Pipeline Routing** | `test_e2_static_only` | **PASSED** |
| **Pipeline Routing** | `test_e3_codebert_only` | **PASSED** |
| **Pipeline Routing** | `test_e4_monolithic_plus_rag` | **PASSED** |
| **Pipeline Routing** | `test_e5_multi_agent_without_rag_codebert` | **PASSED** |
| **Pipeline Routing** | `test_e6_a4_monolithic_ablation` | **PASSED** |
| **RAG Target Isolation** | `test_rag_unconfigured_workspace` | **PASSED** |
| **RAG Target Isolation** | `test_rag_target_isolation` | **PASSED** |
| **RAG Target Isolation** | `test_rag_provenance_recording` | **PASSED** |
| **RAG Target Isolation** | `test_rag_research_mode_rejects_synthetic_embeddings` | **PASSED** |
| **Audit Rubric & Sampling**| `test_manual_audit_rubric_definitions` | **PASSED** |
| **Audit Rubric & Sampling**| `test_adjusted_precision_formula` | **PASSED** |
| **Audit Rubric & Sampling**| `test_proportional_stratified_sampling` | **PASSED** |
| **Audit Rubric & Sampling**| `test_sample_adequacy_shortfall_flagging` | **PASSED** |
| **RAG Evaluation** | `test_extract_files_from_diff` | **PASSED** |
| **RAG Evaluation** | `test_dfor_dcg_ndcg_computation` | **PASSED** |
| **RAG Evaluation** | `test_rag_pending_human_annotation_status` | **PASSED** |
| **RAG Evaluation** | `test_rag_annotation_template_exists` | **PASSED** |

---

## 7. Remaining Blockers Preventing Genuine Experiments

Before running genuine (non-offline/non-mock) experiments for the research paper, the following 4 environment and data dependencies must be resolved:

1. **PyTorch Installation (`torch`):**
   - *Status:* Missing from Python 3.13 `.venv`.
   - *Impact:* Required for real `microsoft/codebert-base` forward passes and GPU/CPU tensor operations.
2. **Trained CodeBERT Defect Checkpoint (`codebert_defect_head.pt`):**
   - *Status:* Unproduced.
   - *Impact:* While `load_codexglue_dataset()` and `calibrate_threshold()` are implemented, training weights on CodeXGLUE defect detection must be run to generate a valid checkpoint. Until trained, E3 strictly reports `TOOL_ERROR` in research mode.
3. **Cloned Target Repository Snapshots for Benchmark PRs:**
   - *Status:* The benchmark contains 101 unified `.diff` files in `evaluation/diffs/`, but not cloned target repository trees.
   - *Impact:* RAG requires a local repository checkout matching the PR base commit SHA. Without local checkouts, RAG reports `SKIPPED_NO_TARGET_WORKSPACE`.
4. **Live Ollama Service with Frozen Model:**
   - *Status:* Offline execution is currently active (`PR_SENTINEL_OFFLINE=1`).
   - *Impact:* For live LLM reviews (E1, E4, E5, E6), an active Ollama instance hosting `qwen2.5-coder:7b` (or 14b) must be reachable at `http://localhost:11434`.

---

## 8. Exact Command to Run a 5-PR Experiment

Once the environment prerequisites are met (or in offline mode for pipeline verification), execute `evaluation/experiment_runner.py`:

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe evaluation/experiment_runner.py --experiment E6 --pr_limit 5
```

To run a specific configuration (e.g. E1 Standalone LLM baseline) on 5 named PRs:
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe evaluation/experiment_runner.py --experiment E1 --pr_ids pr_001 pr_002 pr_003 pr_004 pr_005
```

To run detection evaluation on those generated predictions:
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe evaluation/evaluate_detection.py --configs E6 --matching_protocols M1 M2 M3
```
