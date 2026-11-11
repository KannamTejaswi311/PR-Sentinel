# PR-Sentinel Test Execution & Verification Report

**Date of Execution:** November 2026  
**Environment:** Python 3.13.15, pytest 9.0.2, Windows 11 AMD64  
**Test Framework:** Pytest (Sync runner with `-p no:asyncio -p no:anyio`)  
**Overall Status:** **27 PASSED, 0 FAILED (100% Pass Rate)**

---

## 1. Test Suite Summary Table

| Test Suite File | Scope & Tested Functionality | Total Tests | Passed | Failed | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests/test_matching_engine.py` | Hungarian Bipartite Matching, M1/M2/M3 constraints, Hunk Span distance, 1:1 attribution | 8 | 8 | 0 | 1.15s |
| `tests/test_review_analyzer.py` | Multi-hunk line mapping, unsupported language capture, CodeBERT real-weight check, heuristic agents, ablation matrix | 9 | 9 | 0 | 3.10s |
| `tests/test_pr_sentinel_suite.py` (Sync) | Unit testing of AST parser, Flake8/Bandit runners, CodeBERT similarity, RAG chunking, risk formulas, SQLite history | 10 | 10 | 0 | 3.62s |
| **TOTAL** | **Full System & Evaluation Suite** | **27** | **27** | **0** | **7.87s** |

---

## 2. Detailed Verification by Functional Area

### 2.1. Hungarian Bipartite Matching Engine (`tests/test_matching_engine.py`)
- `test_normalize_file_path`: PASSED. Verifies cross-platform path normalization, stripping `a/` and `b/` git prefixes and unifying backslashes.
- `test_compute_token_jaccard`: PASSED. Confirms token-level Jaccard similarity scoring across natural language descriptions.
- `test_compute_line_distance`: PASSED. Validates statement-specific line distance and hunk span line distance ($dist = 0$ if inside span $[line\_start, line\_end]$).
- `test_m1_strict_matching`: PASSED. Confirms exact file match and $|\Delta \text{line}| \le 3$ threshold boundary conditions.
- `test_m2_tolerant_matching`: PASSED. Confirms $|\Delta \text{line}| \le 10$ sensitivity threshold.
- `test_bipartite_one_to_one_attribution`: PASSED. Proves that when multiple predictions compete for a single ground-truth defect, Hungarian optimization awards the ground-truth issue to the closer prediction (1:1 bijective attribution) and marks the second prediction as an apparent false positive.
- `test_m3_semantic_matching`: PASSED. Verifies semantic matching with category agreement and token Jaccard similarity $\ge 0.30$.
- `test_hunk_span_granularity_matching`: PASSED. Verifies E3 CodeBERT hunk-level predictions match ground-truth lines located within the modified hunk.

### 2.2. Review Analyzer & Pipeline Ablations (`tests/test_review_analyzer.py`)
- `test_static_analyzer_multi_hunk_line_mapping`: PASSED. Verifies that static analysis correctly parses multiple hunks in a diff and extracts target file lines.
- `test_static_analyzer_unsupported_language`: PASSED. Confirms that non-Python source files (`.rs`, `.ts`, `.go`) emit `UNSUPPORTED_LANGUAGE` notices at `granularity = FILE` without crashing.
- `test_codebert_real_weight_enforcement`: PASSED. Confirms that attempting to run CodeBERT in research mode with `require_real_weights=True` when PyTorch or weight checkpoints are missing raises an explicit `RuntimeError` rather than fabricating synthetic vectors.
- `test_bug_detection_agent_heuristics`: PASSED. Validates division by zero detection rules.
- `test_security_review_agent_heuristics`: PASSED. Validates SQL injection and shell command injection detection rules.
- `test_performance_review_agent_heuristics`: PASSED. Validates unbounded query and loop query heuristics.
- `test_code_quality_agent_heuristics`: PASSED. Validates bare `except:` clause detection (`QUAL-02`) and non-descriptive identifier detection (`QUAL-01`).
- `test_pipeline_config_ablation_matrix`: PASSED. Validates component toggle flags across all 10 configurations: `E1`, `E2`, `E3`, `E4`, `E5`, `E6`, `E6_A1`, `E6_A2`, `E6_A3`, `E6_A4`.
- `test_test_generator_finding_id_association`: PASSED. Validates that the test generator preserves finding UUID traceability.

### 2.3. Core Sentinel Subsystems (`tests/test_pr_sentinel_suite.py`)
- `test_1_bug_detection`: PASSED. AST and heuristic detection of unhandled division by zero.
- `test_2_security_review`: PASSED. AST and heuristic detection of command injection.
- `test_3_code_quality`: PASSED. Detection of style and quality issues.
- `test_4_performance_review`: PASSED. Query inside loop detection.
- `test_5_static_analyzer`: PASSED. Bandit security vulnerability detection.
- `test_6_codebert_representation`: PASSED. Self-similarity computation ($sim = 1.0$) and 768-dimensional vector shape.
- `test_7_rag_pipeline`: PASSED. Repository indexing and snippet context retrieval.
- `test_9_review_synthesizer`: PASSED. Deduplication of multi-agent and static findings into unified finding records.
- `test_10_risk_assessment`: PASSED. PR risk scoring formula and transparent multiplier calculation.
- `test_11_review_history_and_feedback`: PASSED. SQLite persistence of review summaries and developer feedback ratings.

---

## 3. End-to-End Experiment Runner Smoke Verification

A dry run of `evaluation/experiment_runner.py` was executed across all 10 configurations on benchmark PRs:

| Configuration ID | Scheduled PRs | Successful PRs | Failed PRs | Generated Findings | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `E1` (Monolithic LLM) | 2 | 2 | 0 | 0 (Offline Mode) | **PASSED** |
| `E2` (Static Analysis) | 2 | 2 | 0 | 7 | **PASSED** |
| `E3` (CodeBERT Classifier) | 2 | 2 | 0 | 0 (Offline Mode) | **PASSED** |
| `E4` (RAG + LLM) | 2 | 2 | 0 | 0 (Offline Mode) | **PASSED** |
| `E5` (Multi-Agent Review) | 2 | 2 | 0 | 15 | **PASSED** |
| `E6` (Full PR-Sentinel) | 2 | 2 | 0 | 22 | **PASSED** |
| `E6_A1` (Ablation: -CodeBERT) | 2 | 2 | 0 | 22 | **PASSED** |
| `E6_A2` (Ablation: -Static) | 2 | 2 | 0 | 15 | **PASSED** |
| `E6_A3` (Ablation: -RAG) | 2 | 2 | 0 | 22 | **PASSED** |
| `E6_A4` (Ablation: -MultiAgent) | 2 | 2 | 0 | 7 | **PASSED** |

Raw JSON predictions and configuration execution manifests were persisted to `evaluation/results/frozen_protocol_v1/raw/`.

---

## 4. Evaluation Engine Smoke Verification

- `evaluation/evaluate_detection.py`:
  - Successfully loaded all 100 benchmark PRs and 537 Cohort 1 confirmed ground-truth issues.
  - Successfully executed bipartite matching under M1, M2, and M3.
  - Computed 95% bootstrap confidence intervals ($B = 1000$ iterations) and paired Wilcoxon signed-rank test placeholders.
- `evaluation/evaluate_rag.py`:
  - Successfully verified diff-file extraction, DFOR@K, MRR, and nDCG@K calculations.
- `evaluation/sample_audit.py`:
  - Successfully sampled Tier 1 ($n=50$) and Tier 2 ($n=60$) candidate false positives into blinded CSV templates with status `PENDING_HUMAN_REVIEW`.
