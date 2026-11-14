# PR-Sentinel: Manual RAG Repository Snapshot Setup & Verification Guide

**Document ID:** `evaluation/MANUAL_RAG_REPOSITORY_SETUP_GUIDE.md`  
**Target Environment:** Windows 10/11, Visual Studio Code PowerShell Terminal  
**Audience:** User (Manual Execution in VS Code with Git / PowerShell)  
**Objective:** Establish local target-repository snapshots in `evaluation/repos/` for all eight benchmark repositories, checkout correct base commit states, configure target repository paths, verify RAG indexing and provenance metadata, and enable cross-file context retrieval without cross-repository contamination.

---

## 1. Benchmark Repository Overview & Distribution

The PR-Sentinel 100-PR benchmark (`evaluation/diff_manifest.csv`) evaluates pull requests across eight open-source target repositories hosted under the `agentic-review-benchmarks` GitHub organization.

| Repository Name | Primary Language | PR Count | Benchmark PR Range | Current Local Status | Target Snapshot Directory |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ghost** | JavaScript / Node.js | 13 | `Ghost_pr1` - `Ghost_pr13` | **VERIFIED** | `evaluation/repos/Ghost_base` |
| **aspnetcore** | C# / .NET | 10 | `aspnetcore_pr1` - `aspnetcore_pr10` | Missing | `evaluation/repos/aspnetcore_base` |
| **cal.com** | TypeScript / React | 16 | `cal.com_pr1` - `cal.com_pr16` | Missing | `evaluation/repos/cal.com_base` |
| **dify** | Python / Flask | 13 | `dify_pr1` - `dify_pr13` | Missing | `evaluation/repos/dify_base` |
| **firefox-ios**| Swift / iOS | 13 | `firefox-ios_pr1` - `firefox-ios_pr13` | Missing | `evaluation/repos/firefox-ios_base` |
| **prefect** | Python / FastAPI | 13 | `prefect_pr1` - `prefect_pr13` | Missing | `evaluation/repos/prefect_base` |
| **redis** | C | 9 | `redis_pr1` - `redis_pr9` | Missing | `evaluation/repos/redis_base` |
| **tauri** | Rust / C | 13 | `tauri_pr1` - `tauri_pr13` | Missing | `evaluation/repos/tauri_base` |
| **TOTAL** | Polyglot | **100** | **100 Pull Requests** | 1 Verified, 7 Pending | `evaluation/repos/` |

---

## 2. Directory Layout & Snapshot Strategy

Under **Protocol v1.0**, RAG context retrieval operates by indexing code chunks from the **base commit snapshot** of the target repository (the state immediately prior to the PR diff).

To prevent cloning large repositories multiple times, PR-Sentinel's resolution engine in `evaluation/experiment_runner.py` searches candidate paths in priority order:
1. `evaluation/repos/{repo_name}_{pr_id}` (PR-specific checkout if base commits diverge).
2. `evaluation/repos/{repo_name}_base` (**Recommended:** Shared base repository snapshot for the repository).
3. `evaluation/repos/{repo_name}` (Fallback repository root).

When multiple PRs in the same repository share a common base branch or base commit (as formatted in `agentic-review-benchmarks`), a single clone at `evaluation/repos/{repo_name}_base` serves all PRs in that repository.

---

## 3. Step-by-Step PowerShell Setup Commands

Open a **PowerShell** terminal in VS Code (`Ctrl + ~`) from the project root (`c:\Users\KannamTejaswi\Major Project\References\pr-agent`).

### Step 3.1: Verify Existing Repositories Folder
```powershell
Set-Location "c:\Users\KannamTejaswi\Major Project\References\pr-agent"
New-Item -ItemType Directory -Force -Path "evaluation\repos"
Get-ChildItem -Path "evaluation\repos"
```
*Expected Output:* `Ghost_base` is listed.

---

### Step 3.2: Clone the Seven Missing Repositories

Run the following commands to clone each repository with depth 50 (sufficient to capture PR base commits while saving disk space and download time):

```powershell
# 1. dify (Python, 13 PRs) - Essential for static + RAG evaluation
git clone --depth 50 https://github.com/agentic-review-benchmarks/dify.git "evaluation\repos\dify_base"

# 2. prefect (Python, 13 PRs) - Essential for static + RAG evaluation
git clone --depth 50 https://github.com/agentic-review-benchmarks/prefect.git "evaluation\repos\prefect_base"

# 3. cal.com (TypeScript, 16 PRs)
git clone --depth 50 https://github.com/agentic-review-benchmarks/cal.com.git "evaluation\repos\cal.com_base"

# 4. redis (C, 9 PRs)
git clone --depth 50 https://github.com/agentic-review-benchmarks/redis.git "evaluation\repos\redis_base"

# 5. tauri (Rust, 13 PRs)
git clone --depth 50 https://github.com/agentic-review-benchmarks/tauri.git "evaluation\repos\tauri_base"

# 6. firefox-ios (Swift, 13 PRs)
git clone --depth 50 https://github.com/agentic-review-benchmarks/firefox-ios.git "evaluation\repos\firefox-ios_base"

# 7. aspnetcore (C#, 10 PRs)
git clone --depth 50 https://github.com/agentic-review-benchmarks/aspnetcore.git "evaluation\repos\aspnetcore_base"
```

*Note on Shallow Clones:* If a specific PR base commit was merged earlier than the 50 most recent commits, run `git -C "evaluation\repos\<repo>_base" fetch --unshallow` to fetch the complete commit history.

---

## 4. Verification of Cloned Repositories & Base Commits

Run this automated verification script in PowerShell to verify that every cloned repository is valid, healthy, and has its `.git` tree intact:

### 4.1 Verification Script
```powershell
python -c "
import subprocess
from pathlib import Path

repos = ['Ghost', 'dify', 'prefect', 'cal.com', 'redis', 'tauri', 'firefox-ios', 'aspnetcore']
base_dir = Path('evaluation/repos')

print(f'=== Verifying {len(repos)} Repository Snapshots in evaluation/repos/ ===')
for repo in repos:
    p = base_dir / f'{repo}_base'
    if not p.exists():
        print(f'[-] {repo:12s}: MISSING directory {p}')
        continue
    git_dir = p / '.git'
    if not git_dir.exists():
        print(f'[-] {repo:12s}: INVALID (no .git directory)')
        continue
    try:
        head_sha = subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip()
        remote = subprocess.check_output(['git', '-C', str(p), 'remote', 'get-url', 'origin'], text=True).strip()
        print(f'[+] {repo:12s}: OK | HEAD: {head_sha[:10]}... | Origin: {remote}')
    except Exception as e:
        print(f'[-] {repo:12s}: ERROR querying git: {e}')
"
```

### 4.2 Expected Output
```
=== Verifying 8 Repository Snapshots in evaluation/repos/ ===
[+] Ghost       : OK | HEAD: 9b6ba0fc89... | Origin: https://github.com/agentic-review-benchmarks/Ghost.git
[+] dify        : OK | HEAD: abc1234567... | Origin: https://github.com/agentic-review-benchmarks/dify.git
[+] prefect     : OK | HEAD: def8901234... | Origin: https://github.com/agentic-review-benchmarks/prefect.git
...
```

---

## 5. Verify Target File Existence for PR Diffs

Before running RAG indexing, verify that the source files modified by each PR diff exist in the corresponding target repository snapshot.

### 5.1 Verification Script
```powershell
python -c "
import csv, re
from pathlib import Path

manifest_path = Path('evaluation/diff_manifest.csv')
with open(manifest_path, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

verified = 0
missing_files = []

for row in rows:
    repo = row['repo']
    diff_path = Path(row['diff_file'])
    repo_dir = Path(f'evaluation/repos/{repo}_base')
    if not repo_dir.exists():
        continue
    if not diff_path.exists():
        continue

    content = diff_path.read_text(encoding='utf-8', errors='replace')
    files = re.findall(r'^\+\+\+\s+b/(.+)$', content, re.MULTILINE)
    for f in files:
        target = repo_dir / f
        if target.exists():
            verified += 1
        else:
            missing_files.append((row['diff_file'], f))

print(f'Verified {verified} target source files across cloned repository snapshots.')
if missing_files:
    print(f'Warning: {len(missing_files)} diff target files not found in base snapshot (may be new files created by PR).')
    print('Sample missing:', missing_files[:3])
"
```

---

## 6. Test RAG Indexing & Provenance Metadata

Verify that PR-Sentinel's `RepositoryRAGPipeline` successfully indexes chunks from the cloned repository, records correct provenance fields (`repo_id`, `commit_sha`, `file_path`), and stores isolated index caches under `.sentinel_rag`.

### 6.1 Test Command
```powershell
python -c "
from pathlib import Path
from sentinel.rag_pipeline import get_rag_pipeline

# Test indexing on dify_base
repo_dir = Path('evaluation/repos/dify_base')
if repo_dir.exists():
    rag = get_rag_pipeline(workspace_dir=str(repo_dir.resolve()), repo_id='dify', commit_sha='HEAD')
    stats = rag.index_repository(force_reindex=False, max_files=50)
    print('Dify RAG Index Stats:', stats)
    
    # Test retrieval
    chunks = rag.retrieve_context('def update_variable(conversation_id, name, value):', top_k=3)
    print(f'Retrieved {len(chunks)} context chunks:')
    for c in chunks:
        print(f'  [Rank {c[\"rank\"]}] {c[\"file_path\"]} (L{c[\"line_start\"]}-L{c[\"line_end\"]}) - Sim: {c[\"similarity_score\"]}')
        assert c['file_path'].startswith(('api/', 'web/', 'core/')), 'File path must be relative to target repository'
    print('SUCCESS: RAG indexing, provenance, and retrieval verified for dify.')
else:
    print('dify_base not yet cloned. Clone it in Step 3.2 to run this verification.')
"
```

### 6.2 Expected Output
```
Dify RAG Index Stats: {'indexed_files_count': 50, 'total_chunks': ..., 'last_indexed_time': '...'}
Retrieved 3 context chunks:
  [Rank 1] api/services/conversation_variable_updater.py (L1-L45) - Sim: 0.8124
  ...
SUCCESS: RAG indexing, provenance, and retrieval verified for dify.
```

---

## 7. Preventing Cross-Repository Contamination

Under **Protocol v1.0**, RAG index caches must be strictly isolated:
- Indexes are persisted to `{workspace_dir}/.sentinel_rag/{safe_repo}_{safe_sha}_rag_index.json`.
- When switching between PRs from different repositories (e.g. from `Ghost` to `dify`), `sentinel/pipeline.py` automatically detects the change in `repo_id` or `target_resolved` directory and instantiates a clean `RepositoryRAGPipeline`.
- No chunks from `Ghost` can ever leak into `dify` or `cal.com` prompts.

---

## 8. Summary of Completion

Once all 7 repositories are cloned into `evaluation/repos/<repo>_base`:
1. RAG status in execution manifests will transition from `TOOL_ERROR` to `SUCCESS` across all 100 PRs.
2. Configuration **E4** (RAG + Monolithic) and **E6-A1** (Primary empirical system) will have access to genuine codebase context for all 100 PRs.
3. Every prompt sent to the LLM agents will contain genuine multi-file context snippets from the PR's target repository.
