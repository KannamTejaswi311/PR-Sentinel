# PR-Sentinel Implementation & Repair Changelog

**Release Version:** Research-Ready v1.0  
**Experimental Protocol:** PR-Sentinel Frozen Experimental Protocol v1.0 & Protocol Amendment v1.1  
**Target Venue:** IEEE Transactions on Software Engineering / IEEE International Conference on Software Engineering (ICSE)  
**Date:** November 2026  

---

## 1. Executive Summary

This changelog records the complete implementation, repair, standardization, and verification of the PR-Sentinel AI-Assisted Pull Request Review and Evaluation System. All components have been transitioned from proof-of-concept prototypes to a rigorous, scientifically defensible experimental research platform adhering strictly to the ratified Frozen Experimental Protocol v1.0 and Protocol Amendment v1.1.

---

## 2. Benchmark Dataset & Ground-Truth Normalization

| Component | Previous Status | Upgraded Research-Ready Status | Defensibility Rationale |
| :--- | :--- | :--- | :--- |
| **Benchmark Diffs** | Ambiguous count (100 vs 101) | Exactly **100 verified unified diffs** in `evaluation/diffs/` | Validated against `evaluation/diff_manifest.csv` (100 rows, all `SUCCESS`). Protocol amended to reflect the true count. |
| **Original Assets** | Risk of accidental overwrite | **Strictly Preserved** | `git_code_review_bench_100_w_open_prs.jsonl`, `ground_truth.csv`, `diff_manifest.csv`, `evaluation/diffs/`, and legacy outputs were preserved without modification. |
| **Ground-Truth Cohorts** | Unpartitioned list | Formally partitioned into 3 distinct cohorts in `evaluation/datasets/normalized/ground_truth_cohorts.csv` | - **Cohort 1 (Primary GT):** 537 confirmed, verifiable issues with exact file matching.<br>- **Cohort 2 (Candidate GT):** 16 candidate issues matched by substring.<br>- **Cohort 3 (Unresolved):** 27 records missing file/line references (reported separately). |
| **Vulnerability Schema** | Free-form heterogeneous text | Standardized 5-class taxonomy (`FindingCategory`) | Mapped to `BUG` (274), `CODE_QUALITY` (135), `UNKNOWN` (125), `SECURITY` (30), `PERFORMANCE` (16). |

---

## 3. Core Engine Upgrades (`sentinel/`)

### 3.1. Canonical Schemas (`sentinel/schemas.py`)
- Created standardized dataclasses for all model outputs and ground-truth issues:
  - `ModelFinding`: Enforces `pr_id`, `source`, `category`, `file_path`, `line_start`, `line_end`, `rule_id`, `severity`, `description`, `recommendation`, `granularity`, `execution_status`, and `metadata`.
  - `GroundTruthIssue`: Normalizes benchmark ground-truth records across repositories.
  - `PipelineConfig`: Implements factory method `for_experiment(exp_id)` parameterizing E1–E6 and ablations E6-A1 through E6-A4.
  - `ExecutionStatus`: Tracks per-step outcomes (`SUCCESS`, `SKIPPED`, `UNSUPPORTED_LANGUAGE`, `TIMEOUT`, `TOOL_ERROR`).
  - `FindingGranularity`: Distinguishes localization precision (`STATEMENT`, `HUNK_SPAN`, `FILE`).

### 3.2. CodeBERT Representation & Defect Classifier (`sentinel/codebert_analyzer.py`)
- **Real-Weight Constraint:** Added `require_real_weights: bool` enforcing `RuntimeError` on model loading failure during research mode, preventing synthetic embedding fabrication.
- **Hunk-Level Localization:** Implemented `analyze_diff_hunks()` emitting `ModelFinding` instances with `granularity = FindingGranularity.HUNK_SPAN`.
- **Deterministic Offline Fallback:** Replaced Python's non-deterministic `hash()` with SHA-256 modular mapping for reproducible offline execution.
- **Diff Parsing Integrity:** Restricted diff embedding extraction strictly to added lines (`+`).

### 3.3. Static Analyzer (`sentinel/static_analyzer.py`)
- **Multi-File Diff Splitting:** Added `_split_multi_file_diff()` to isolate per-file changes across complex PR diffs.
- **Accurate Hunk Line Mapping:** Implemented `_extract_code_with_line_map()` mapping 1-based relative temp-file lines to real-file 1-based target line numbers across multiple hunks.
- **Language Boundary Notification:** Explicitly flags non-Python files with `UNSUPPORTED_LANGUAGE` notices at `granularity = FILE`.
- **Tool Error Capture:** Traps Bandit and Flake8 execution failures, timeouts, and syntax errors into the canonical execution manifest.
- **Schema Mapping:** Added `analyze_diff_to_findings()` exporting canonical `ModelFinding` instances.

### 3.4. Multi-Agent Review Orchestration (`sentinel/multi_agent.py`)
- **Synchronous Execution:** Added `run_review_sync()` for deterministic, thread-safe evaluation immune to IOCP socket bindings.
- **Offline Environment Variable:** Integrated `PR_SENTINEL_OFFLINE` to cleanly bypass network socket probes in sandboxed test runners.
- **Monolithic LLM Review:** Added `run_monolithic_review()` for baseline E1 and ablation E6-A4.
- **Rule Enrichment:** Added `QUAL-02` (Bare except clause detection) to `CodeQualityAgent`.

### 3.5. Review Synthesis & PR Risk Scoring (`sentinel/synthesizer.py`)
- **Additive Penalty Correction:** Repaired compounding bug in `calculate_pr_risk_score` where CodeBERT penalties multiplied exponentially; now applied as a well-calibrated additive term.
- **Provenance Preservation:** Retained original `source` and `all_sources` in findings during cross-agent deduplication.
- **Operator Precedence:** Clarified operator grouping in security issue aggregation.

### 3.6. Automated Test Recommendation Engine (`sentinel/test_generator.py`)
- **Clean Diff Parsing:** Updated `_extract_modified_functions` to parse only added lines (`+`) and hunk context headers (`@@`).
- **Traceability:** Linked generated test cases to specific `finding_id` identifiers.
- **Dead Code Removal:** Eliminated unused `import ast`.

### 3.7. GitHub Pull Request Integration (`sentinel/github_integration.py`)
- **Explicit Auth Failures:** Replaced silent fallback to `auth_service.py` with explicit `ConnectionError` unless `allow_demo=True` is passed.
- **Synchronous Client:** Added `fetch_pr_data_sync()` and `post_review_comment()` with full error capture.

### 3.8. Central Pipeline Orchestrator (`sentinel/pipeline.py`)
- **Config-Driven Architecture:** Ingests `PipelineConfig` to activate/deactivate components selectively across E1–E6.
- **Deterministic Sync Runner:** Added `execute_review_sync()` with comprehensive `execution_manifest`.
- **Null Safety:** Repaired synthesis bypass paths when individual modules are deactivated.

---

## 4. Evaluation Suite Implementation (`evaluation/`)

### 4.1. Bipartite Matching Engine (`evaluation/matching_engine.py`)
- Implemented globally optimal **1:1 Maximum Weight Bipartite Matching** using `scipy.optimize.linear_sum_assignment` (Hungarian algorithm).
- Supports three protocol modes:
  - **M1 (Strict - Primary):** Exact file match, $|\Delta \text{line}| \le 3$.
  - **M2 (Tolerant - Sensitivity):** Exact file match, $|\Delta \text{line}| \le 10$.
  - **M3 (Semantic - Sensitivity):** Exact file match, category match, token Jaccard similarity $\ge 0.30$.
- Implements Hunk Span localization distance: minimum distance from ground truth line to span $[line\_start, line\_end]$.

### 4.2. Central Experiment Runner (`evaluation/experiment_runner.py`)
- Orchestrates reproducible batch runs across all 10 experimental configurations:
  - Baselines: `E1`, `E2`, `E3`, `E4`, `E5`, `E6`.
  - Ablations: `E6_A1` (-CodeBERT), `E6_A2` (-Static), `E6_A3` (-RAG), `E6_A4` (-MultiAgent).
- Outputs immutable JSON predictions to `evaluation/results/frozen_protocol_v1/raw/{config_id}/{pr_id}.json`.

### 4.3. Research Metric Evaluator (`evaluation/evaluate_detection.py`)
- Replaced unprincipled Positive-Unlabeled assumptions:
  - Strictly reports **Ground-Truth Recall ($R_{\text{GT}}$)**, **Benchmark-Apparent Precision ($P_{\text{app}}$)**, and **Benchmark-Apparent F1 ($F1_{\text{app}}$)**.
  - **Prohibits** Accuracy and False Positive Rate (FPR).
- Implemented **PR-level Clustered Bootstrap Confidence Intervals** ($B = 1000$ iterations).
- Integrated **Paired Wilcoxon signed-rank tests** comparing E6 against each baseline.

### 4.4. Repository-Aware RAG Evaluator (`evaluation/evaluate_rag.py`)
- Measures context retrieval quality via:
  - **DFOR@K** (Diff-File Overlap Rate at top-1, top-3, top-5) as a retrieval-coverage proxy.
  - **MRR** (Mean Reciprocal Rank).
  - **nDCG@K** (Normalized Discounted Cumulative Gain).

### 4.5. Two-Tier Manual Audit Sampler (`evaluation/sample_audit.py`)
- **Tier 1:** Stratified sample of $n = 50$ apparent false positives from E6 for quantitative precision adjustment.
- **Tier 2:** Qualitative sample of $n = 15$ apparent false positives each from E1, E2, E3, E5 for cross-model error analysis.
- Generates blinded double-reviewer template with status `PENDING_HUMAN_REVIEW` without fabricating verdicts.
