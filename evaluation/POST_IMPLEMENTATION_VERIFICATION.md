# PR-Sentinel: Post-Implementation Verification Audit (Step 1)

**Audit Type:** Post-Implementation Inspection & Consistency Audit  
**Target System:** PR-Sentinel AI-Assisted Pull Request Review System  
**Audited Artifacts:** Core Engine (`sentinel/`), Evaluation Suite (`evaluation/`), Test Suites (`tests/`), Reports (`evaluation/*.md`), and Ratified Protocols (`Frozen Protocol v1.0`, `Protocol Amendment v1.1`)  
**Date:** November 2026  
**Status:** Verification Complete — Code & Dataset Modifications Prohibited  

---

## 1. Executive Verdict & Readiness Statement

PR-Sentinel's architecture has achieved significant structural maturity: canonical schemas are established, the Hungarian 1:1 bipartite matching engine is mathematically verified, metric reporting strictly adheres to Positive-Unlabeled (PU) conventions (Accuracy and FPR are eliminated), and offline deterministic execution runs cleanly without unhandled exceptions.

However, **PR-Sentinel is NOT yet ready for full-scale benchmark evaluation**. Critical architectural disconnections and environment blockers currently prevent genuine experimental evaluation:
1. **Model Weights Missing:** Neither `microsoft/codebert-base` nor `torch` is loaded; only deterministic SHA-256 projections are active.
2. **Missing Supervised Head & Calibration:** The CodeBERT defect classification head is an analytical heuristic proxy, not a trained neural head, with no CodeXGLUE training, validation, or Youden's J calibration.
3. **Execution Disconnections:** `PipelineConfig` does not call `analyze_diff_hunks()` for E3, and monolithic review (`run_monolithic_review`) is never invoked for E1, E4, or E6-A4.
4. **RAG Context Target Inversion:** Target repositories are not checked out at their base commits; the RAG pipeline defaults to indexing PR-Sentinel's own codebase.
5. **Critical Audit Rubric Inversion:** CAT-A and CAT-B definitions in `sample_audit.py` and `RESEARCH_READINESS_REPORT.md` are inverted relative to the ratified Frozen Protocol v1.0.

---

## 2. Inconsistency Audits by Area

### 2.1. CodeBERT Verification (`sentinel/codebert_analyzer.py`)

| Aspect | Current Implementation Status | Code Evidence & File References | Audit Assessment |
| :--- | :--- | :--- | :--- |
| **Model Loading (`microsoft/codebert-base`)** | **Not Loaded (Fallback Active)** | `sentinel/codebert_analyzer.py:103-122` (`load_model`). In `.venv`, `import torch` raises `ModuleNotFoundError: No module named 'torch'`. Status returns: `is_loaded: True, using_pretrained_weights: False, fallback_active: True`. | **UNVERIFIED & BLOCKED:** Real weights are not running due to missing PyTorch dependency in the environment. |
| **Supervised Classification Head** | **Analytical Heuristic Proxy Only** | `sentinel/codebert_analyzer.py:22-64` (`CodeBERTDefectClassifier`). Docstring claims `MLP(1536->256->2)`, but `predict_proba()` implements: `score = sum(abs(x) for x in v[:32]) / 32.0; prob = 1.0 / (1.0 + math.exp(-6.0 * (score - 0.05)))`. `self.is_trained = False`. | **UNIMPLEMENTED:** No PyTorch weights, no trained parameters, no MLP layer. |
| **CodeXGLUE Splits** | **Unused / Absent** | `sentinel/codebert_analyzer.py:39` contains metadata string `"trained_on": "CodeXGLUE_defect_detection"`. `sentinel/datasets_info.py:10-24` documents the dataset URL. | **UNIMPLEMENTED:** No CodeXGLUE dataset files, splits (train/val/test), or loading scripts exist in the repository. |
| **Threshold Calibration** | **Hardcoded Placeholder** | `sentinel/codebert_analyzer.py:30-38`. `self.threshold = 0.5` is statically initialized. Metadata cites `Youden_J`, but no validation ROC curve or calibration code exists. | **UNIMPLEMENTED:** The 0.5 threshold is an arbitrary default. |
| **Research Mode Fallback Rejection** | **Partially Enforced** | `load_model()` (lines 117-121), `tokenize_code()` (lines 164-165), and `extract_code_representation()` (lines 240-241) raise `RuntimeError` if `require_real_weights=True`. However, `analyze_diff_hunks()` (lines 346-358) traps missing weights and returns a `TOOL_ERROR` finding rather than halting. | **PARTIALLY VERIFIED:** Unit test passes, but pipeline wraps errors in `try...except`, absorbing exceptions. |
| **Accidental Fallback in E3** | **Guarded in Schema, Absorbed in Pipeline** | In `sentinel/schemas.py:138`, E3 sets `require_real_weights=True`. However, in `sentinel/pipeline.py:68-73`, the call is wrapped in `try...except Exception as e: manifest["codebert"] = f"ERROR: {str(e)}"`. In E6, `require_real_weights=False` (line 178), so E6 silently uses SHA-256 embeddings without error. | **VERIFIED CONCERN:** E3 correctly rejects fallback by failing, but E6 runs on fallback embeddings without warning. |
| **Hunk-Level Localization** | **Implemented in Analyzer, Disconnected in Pipeline** | `sentinel/codebert_analyzer.py:335-428` (`analyze_diff_hunks`) implements regex hunk parsing (`@@ -\d+ \+(\d+) @@`), assigns `granularity = HUNK_SPAN`, `line_start`, and `line_end`. BUT `sentinel/pipeline.py:70` calls `analyze_code_diff()`, NOT `analyze_diff_hunks()`. | **DISCONNECTED:** Hunk localization logic is functional in unit tests, but bypassed in the main pipeline. |

---

### 2.2. Offline Experiment Verification (`evaluation/experiment_runner.py` & Pipeline)

#### Why E1, E3, and E4 Returned Zero Findings in Offline Mode:
1. **Configuration E1 (Standalone LLM):**
   - In `sentinel/pipeline.py` (lines 121–123):
     ```python
     elif self.config.enable_llm:
         manifest["multi_agent"] = "MONOLITHIC_MODE"
     ```
     `pipeline.py` contains no invocation of `run_monolithic_review()`. The block only sets a manifest label. Furthermore, in offline mode (`PR_SENTINEL_OFFLINE=1`), `_check_ollama()` returns `False`. Because `enable_synthesis=False` for E1, the finding list is empty (`[]`).
2. **Configuration E3 (CodeBERT Only):**
   - In `sentinel/pipeline.py` (line 70), `codebert_result = self.codebert.analyze_code_diff(diff_content)`.
   - `analyze_code_diff()` returns a dictionary of representation statistics (`token_count`, `embedding_sample`, `semantic_indicators`), but **no `"findings"` list**.
   - `analyze_diff_hunks()` is never called by `pipeline.py`.
   - In `evaluation/experiment_runner.py` (lines 60–62), `extract_model_findings()` checks for `review_result.get("codebert_analysis", {}).get("findings")`, which is `None`. Therefore, 0 findings are produced.
3. **Configuration E4 (LLM + RAG):**
   - In `PipelineConfig.for_experiment("E4")`: `enable_rag=True, enable_llm=True, enable_multi_agent=False, enable_static=False, enable_synthesis=False`.
   - `pipeline.py` retrieves RAG context, but because `enable_multi_agent=False`, execution enters `elif self.config.enable_llm:`, which performs no LLM call. Synthesis is skipped, yielding exactly 0 findings.

#### Execution State of Smoke Test vs. Real Model Inference:
- **Qwen Model Inference:** **NOT EXECUTED.** `PR_SENTINEL_OFFLINE=1` bypassed all Ollama HTTP client requests. Zero tokens were generated.
- **Real CodeBERT Inference:** **NOT EXECUTED.** `torch` was unavailable; zero PyTorch tensor operations were performed.
- **Target-Repository RAG Retrieval:** **NOT EXECUTED.** No benchmark repository was cloned or checked out at its base commit.
- **Distinction:** The smoke test demonstrated **harness and schema compatibility** (CSV parsing, diff reading, JSON output serialization, metric calculation, error handling). It did **not** execute real AI model inference.

---

### 2.3. Ablation Verification

| Configuration | Configured Toggles | Pipeline Execution Check | Smoke Test Finding Count | Finding Count Explanation |
| :--- | :--- | :--- | :---: | :--- |
| **`E6`** | All components enabled (`codebert=T, static=T, rag=T, multi_agent=T`) | CodeBERT (metrics only) + Static + RAG (unused offline) + Multi-Agent Heuristics | **22** | 7 static findings + 15 heuristic agent findings. |
| **`E6_A1`** (-CodeBERT) | `codebert=F, static=T, rag=T, multi_agent=T` | CodeBERT skipped; Static + RAG + Multi-Agent Heuristics run. | **22** | CodeBERT produced 0 findings in E6 (calling `analyze_code_diff`). Thus, disabling CodeBERT removes 0 findings ($22 - 0 = 22$). |
| **`E6_A2`** (-Static) | `codebert=T, static=F, rag=T, multi_agent=T` | Static skipped; CodeBERT + RAG + Multi-Agent Heuristics run. | **15** | Static analysis contributed 7 findings in E6. Disabling it drops finding count by exactly 7 ($22 - 7 = 15$). |
| **`E6_A3`** (-RAG) | `codebert=T, static=T, rag=F, multi_agent=T` | RAG skipped; CodeBERT + Static + Multi-Agent Heuristics run. | **22** | In offline mode, the LLM is skipped. RAG context is only passed to `_call_agent_llm()`. Heuristic analysis (`agent.analyze_heuristically`) does not read RAG context. Disabling RAG removes 0 findings ($22 - 0 = 22$). |
| **`E6_A4`** (-MultiAgent) | `codebert=T, static=T, rag=T, multi_agent=F, llm=T` | Multi-Agent skipped; Monolithic LLM review not called; Static runs. | **7** | Multi-Agent heuristics contributed 15 findings. Disabling multi-agent leaves only the 7 static findings. |

**Audit Conclusion:** Disabled components are truly bypassed via execution toggles. The identical counts between E6, E6-A1, and E6-A3 are mathematically explained by CodeBERT producing no finding objects and RAG context affecting only the offline-disabled LLM.

---

### 2.4. Test Report Verification

#### Exclusion of `test_8`, `test_12`, and `test_13`:
- **Files & Functions:** `tests/test_pr_sentinel_suite.py`:
  - `test_8_multi_agent_review` (`@pytest.mark.asyncio`, calls `await orchestrator.run_review`)
  - `test_12_end_to_end_pipeline` (`@pytest.mark.asyncio`, calls `await pipeline.execute_review`)
  - `test_13_github_pr_integration` (`@pytest.mark.asyncio`, calls `await github.fetch_pr_data`)
- **Reason for Exclusion:** In Python 3.13 on Windows inside the sandbox, the `pytest-asyncio` plugin initializes a Proactor event loop that attempts internal loopback socket bindings (IOCP completion ports). Due to sandbox network isolation, this causes the test runner to hang indefinitely. Running with `-p no:asyncio -p no:anyio` and `-k "not (test_8 or test_12 or test_13)"` bypassed the plugin stall.
- **Async Tests Skipped:** Exactly 3 asynchronous integration tests were skipped. Zero asynchronous tests were executed.

#### Discrepancy in Execution Times (6.13s vs. 7.87s):
- **7.87 seconds:** The arithmetic sum of three separate test commands executed during staged validation:
  - `test_matching_engine.py`: 1.15s
  - `test_review_analyzer.py`: 3.10s
  - `test_pr_sentinel_suite.py` (sync subset): 3.62s
  - Total: $1.15 + 3.10 + 3.62 = 7.87\text{s}$.
- **6.13 seconds:** The wall-clock execution time when all test files were invoked together in a single command (`pytest tests/test_matching_engine.py tests/test_review_analyzer.py tests/test_pr_sentinel_suite.py ...`).

#### Test Categorization:
- **Unit Tests (23 passed):**
  - `test_matching_engine.py` (8 tests): path normalization, token Jaccard, line distance, M1 strict, M2 tolerant, 1:1 Hungarian attribution, M3 semantic, hunk span distance.
  - `test_review_analyzer.py` (7 tests): CodeBERT real-weight check, 4 agent heuristic tests, ablation matrix check, test generator finding-ID association.
  - `test_pr_sentinel_suite.py` (8 tests): bug detection, security review, code quality, performance review, static analyzer, CodeBERT representation, review synthesizer, risk assessment.
- **Integration Tests (4 passed, 3 skipped):**
  - `test_review_analyzer.py` (2 passed): multi-hunk line mapping (runs real tempfile AST/Bandit/Flake8), unsupported language handling.
  - `test_pr_sentinel_suite.py` (2 passed): RAG pipeline indexing (filesystem traversal), review history & feedback (SQLite DB write/read).
  - `test_pr_sentinel_suite.py` (3 skipped): `test_8`, `test_12`, `test_13`.
- **Smoke Tests (4 automated scripts):**
  - `experiment_runner.py` (10 configurations on 2 PRs).
  - `evaluate_detection.py` (Hungarian matching on smoke outputs).
  - `evaluate_rag.py` (retrieval metric extraction).
  - `sample_audit.py` (audit record generation).

---

### 2.5. RAG Evaluation Verification (`sentinel/rag_pipeline.py` & `evaluation/evaluate_rag.py`)

| Requirement | Implementation State | Evidence & References |
| :--- | :--- | :--- |
| **Base Commit Repository Indexing** | **NOT IMPLEMENTED** | `sentinel/rag_pipeline.py:58-69`. The pipeline accepts `workspace_dir`, but does not clone, fetch, or checkout the benchmark repositories at their base commit SHAs. |
| **Host Repository Accidental Indexing** | **CONFIRMED DEFECT** | `sentinel/rag_pipeline.py:59`: `self.workspace_dir = Path(workspace_dir) if workspace_dir else Path(__file__).resolve().parent.parent`. In `PipelineConfig.for_experiment("E6")`, `target_repo_dir=None`. The RAG engine indexes `c:\Users\...\pr-agent` (the host PR-Sentinel codebase), retrieving pr-agent files as context for external benchmark PRs. |
| **Real CodeBERT Embeddings in RAG** | **NOT USED** | In `sentinel/rag_pipeline.py:140`, chunk embeddings call `self.codebert.extract_code_representation()`, which falls back to deterministic SHA-256 embeddings because `torch` is not installed. |
| **DFOR@K, MRR, nDCG@K Formulas** | **Mathematically Correct** | `evaluation/evaluate_rag.py:51-66` correctly implements DCG, nDCG, and reciprocal rank calculations. |
| **Retrieval Coverage vs. Semantic Relevance** | **Conflated in Code** | In `evaluation/evaluate_rag.py:153-159`, binary relevance is computed as: `is_relevant = 1.0 if any(norm_chunk_file in mf or mf in norm_chunk_file for mf in modified_files) else 0.0`. nDCG and MRR are computed purely on file overlap, measuring **retrieval coverage only**. |
| **Frozen 25-PR Human Relevance Annotations** | **MISSING** | No human semantic relevance dataset or annotation file exists in `evaluation/` or `evaluation/datasets/`. |

---

### 2.6. Manual Audit Protocol Consistency

#### Critical Rubric Contradiction:
A direct semantic contradiction exists between the ratified Frozen Protocol and the implementation in `sample_audit.py` / `RESEARCH_READINESS_REPORT.md`:

```
RATIFIED FROZEN PROTOCOL v1.0 (Table 4 & Section 5.3):
--------------------------------------------------------------------------------
CAT-A  -->  True Hallucination / Erroneous Finding (Confirmed False Positive)
CAT-B  -->  Latent Valid Defect (Omitted by benchmark annotators -> Valid TP)
CAT-C  -->  Benign Quality Observation / Nitpick (Subjective style / convention)

IMPLEMENTATION IN sample_audit.py (Lines 25-30) & RESEARCH_READINESS_REPORT.md:
--------------------------------------------------------------------------------
CAT-A  -->  "Valid Finding: Genuine defect... (True Positive)"      [INVERTED]
CAT-B  -->  "True False Positive: Hallucination..."                [INVERTED]
CAT-C  -->  "Borderline / Subjective: Stylistic suggestion..."
```

**Impact:** If manual auditors follow the rubric in `sample_audit.py`, their labels will invert the formula defined in the Frozen Protocol ($P_{\text{adj}} = \frac{TP + \hat{\rho}_B \cdot FP_{\text{app}}}{Total}$), counting hallucinations as valid defects.

#### Sample Size & Shortfall Handling:
- In `evaluation/sample_audit.py:55-65` (`sample_candidates`):
  ```python
  if len(fps) <= n_samples:
      return list(fps)
  ```
  When fewer than $n$ apparent false positives exist (e.g., E6 smoke test produced 22 apparent FPs vs. $n=50$ target; offline E1/E3 produced 0), the function returns whatever exists (or an empty list) without notification, padding, or warning.
- **Stratification Bias:** In `sample_audit.py:73-76`, probability weights are calculated as:
  `weights = df.groupby(["category", "severity"])["finding_id"].transform("count"); prob = weights / weights.sum()`.
  Because `prob` is assigned per-item based on stratum count, sampling with `p=prob` applies **quadratic weighting** ($count^2$), over-sampling dense strata rather than achieving true proportional stratified sampling.

---

## 3. Verified vs. Unverified Functionality Matrix

| System Component | Implemented & Verified | Unverified / Blocked / Incomplete |
| :--- | :--- | :--- |
| **Hungarian Bipartite Matching** | **VERIFIED:** Strict M1 ($\le 3$), tolerant M2 ($\le 10$), semantic M3 (Jaccard $\ge 0.3$), 1:1 bijective attribution, hunk span distance. | None. |
| **Metric Calculation Engine** | **VERIFIED:** $R_{\text{GT}}$, $P_{\text{app}}$, $F1_{\text{app}}$, $B=1000$ PR-level clustered bootstrap 95% CIs, Wilcoxon signed-rank test. Accuracy and FPR strictly eliminated. | None. |
| **Static Analyzer** | **VERIFIED:** Multi-file diff splitting, multi-hunk relative-to-actual line mapping, `UNSUPPORTED_LANGUAGE` notices. | Live execution on non-Python languages (ESLint, SwiftLint). |
| **Multi-Agent Heuristics** | **VERIFIED:** AST division by zero, command injection, SQL injection, bare except clause (`QUAL-02`), non-descriptive names (`QUAL-01`). | Live LLM generation via Ollama; interaction between RAG and agents. |
| **Pipeline Ablation Toggles** | **VERIFIED:** Clean execution bypass for `enable_codebert`, `enable_static`, `enable_rag`. | Monolithic review execution path in `enable_llm` when multi-agent is disabled. |
| **CodeBERT Classifier** | **TESTED:** Schema fields, `require_real_weights` exception in unit test, HUNK_SPAN data structure. | **UNVERIFIED:** Neural weights loading, CodeXGLUE training, validation calibration, pipeline hunk analysis connection. |
| **Repository RAG Pipeline** | **TESTED:** Text chunking, AST metadata tagging, retrieval formula calculation. | **UNVERIFIED:** Target repo checkout at base commit, real embeddings, human semantic relevance annotations. |
| **Manual Audit Sampler** | **TESTED:** Blinded CSV export format, tier separation (Tier 1 vs. Tier 2). | **BLOCKED:** Rubric definitions inverted; sampling probability weighting biased; shortfall handling absent. |

---

## 4. Critical Blockers Before Real Inference

1. **Environment Dependency Blockers:**
   - **PyTorch Missing:** PyTorch is not installed in `.venv`. CodeBERT cannot load real weights or execute neural tensor operations.
   - **Live Ollama Service Inactive:** No local Ollama daemon hosting `qwen2.5-coder:3b-instruct` was running; all LLM calls were bypassed via `PR_SENTINEL_OFFLINE=1`.
2. **Pipeline Wiring Blockers:**
   - **CodeBERT Diff Hunk Disconnection:** `sentinel/pipeline.py` calls `analyze_code_diff()` instead of `analyze_diff_hunks()`, generating zero findings for E3.
   - **Monolithic LLM Review Disconnection:** `sentinel/pipeline.py` contains no call to `run_monolithic_review()` when `enable_multi_agent=False` and `enable_llm=True`, generating zero findings for E1, E4, and E6-A4.
3. **Repository Context Inversion:**
   - Benchmark PR target repositories are not checked out. The RAG pipeline indexes the PR-Sentinel host repo.
4. **Audit Rubric Semantic Inversion:**
   - Contradiction between Frozen Protocol (CAT-A = FP, CAT-B = Valid Latent) and `sample_audit.py` (CAT-A = Valid, CAT-B = FP).

---

## 5. Recommended Order of Corrective Actions

Before launching any real-model testing or benchmark experiments, execute corrections in the following strict sequential order:

```
[Phase A: Protocol & Rubric Alignment]
  1. Correct CAT-A, CAT-B, CAT-C definitions in sentinel/sample_audit.py,
     manual_audit_manifest.json, and evaluation/RESEARCH_READINESS_REPORT.md
     to match Frozen Protocol v1.0 (CAT-A = Hallucination/FP, CAT-B = Latent Valid Defect).
  2. Fix stratified sampling logic in sample_candidates() to use true proportional allocation
     and add explicit warnings when apparent FPs < sample_size.

[Phase B: Pipeline Wiring Repairs]
  3. Wire CodeBERT hunk analysis: Update sentinel/pipeline.py to invoke
     self.codebert.analyze_diff_hunks(diff_content, pr_id=pr_id) when enable_codebert=True,
     populating codebert_result["findings"].
  4. Wire Monolithic LLM review: Update sentinel/pipeline.py line 122 to invoke
     self.orchestrator.run_monolithic_review() when enable_llm=True and enable_multi_agent=False.

[Phase C: Environment & Real Model Enablement]
  5. Install PyTorch in the virtual environment (CPU or CUDA).
  6. Verify CodeBERT model loading (microsoft/codebert-base) with require_real_weights=True.
  7. Start local Ollama daemon (ollama serve) with qwen2.5-coder:3b-instruct.

[Phase D: RAG Target Repository Isolation]
  8. Update evaluation/experiment_runner.py to provide isolated target_repo_dir paths
     for benchmark PRs rather than defaulting to the PR-Sentinel host directory.
  9. For semantic RAG evaluation, acquire or produce the required 25-PR human relevance annotations.
```

---

## 6. Small-Scale Real-Model Testing Readiness Assessment

### Is PR-Sentinel Ready for Small-Scale Real-Model Testing?
**NO, NOT IN ITS CURRENT STATE.**

While the deterministic harness (static analysis, regex heuristics, Hungarian bipartite matching, and bootstrap metrics) is verified and passes 27/27 unit tests, attempting small-scale real-model testing today will fail for two distinct reasons:
1. **Environment Failure:** PyTorch is absent, causing CodeBERT to fail with `RuntimeError` when `require_real_weights=True`, and Ollama was offline.
2. **Code Wiring Failure:** Even if Ollama and PyTorch were active, E1, E3, and E4 would still return 0 findings because `pipeline.py` does not invoke `analyze_diff_hunks()` for CodeBERT or `run_monolithic_review()` for monolithic LLM review.

**Immediate Next Step:** Once Phase A and Phase B code wiring repairs are authorized and performed, and the Ollama/PyTorch environment dependencies are verified, PR-Sentinel will be fully ready for small-scale (e.g. 5-PR) live model validation.
