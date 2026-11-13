"""
PR-Sentinel: Research RAG Evaluation Engine (Protocol v1.0 & Amendment v1.1)
Evaluates context retrieval quality for repository-aware RAG configurations (E4, E6).
Computes:
- DFOR@K (Diff-File Overlap Rate at top-K) — retrieval-coverage proxy
- MRR (Mean Reciprocal Rank) of relevant context chunks
- nDCG@K (Normalized Discounted Cumulative Gain at rank K)
STRICT DISTINCTION: DFOR is documented as a retrieval-coverage measure,
not semantic relevance. Graded relevance is evaluated separately when annotated.
"""

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateRAG")

BASE_DIR = Path(__file__).resolve().parent
DIFFS_DIR = BASE_DIR / "diffs"
MANIFEST_PATH = BASE_DIR / "diff_manifest.csv"
RAW_RESULTS_DIR = BASE_DIR / "results" / "frozen_protocol_v1" / "raw"
OUTPUT_DIR = BASE_DIR / "results" / "frozen_protocol_v1"
ANNOTATIONS_DIR = BASE_DIR / "datasets"
HUMAN_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "rag_relevance_annotations.csv"


def extract_files_from_diff(diff_content: str) -> Set[str]:
    """Extract set of normalized modified file paths from a unified diff."""
    files = set()
    for line in diff_content.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                files.add(path.replace("\\", "/").lower())
        elif line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                b_path = re.sub(r"^[ab]/", "", parts[3]).strip()
                files.add(b_path.replace("\\", "/").lower())
    return files


def load_human_relevance_annotations(annotations_path: Path = HUMAN_ANNOTATIONS_PATH) -> Dict[str, Dict[str, float]]:
    """
    Load validated human graded relevance annotations if present.
    Returns mapping: pr_id -> {chunk_id or file_path -> relevance_score (0, 1, or 2)}.
    """
    if not annotations_path.exists():
        return {}

    judgments: Dict[str, Dict[str, float]] = {}
    try:
        df = pd.read_csv(annotations_path, comment="#")
        for _, row in df.iterrows():
            pr_id = str(row.get("pr_id", "")).strip()
            cid = str(row.get("chunk_id", "")).strip()
            fpath = str(row.get("file_path", "")).strip().replace("\\", "/").lower()
            try:
                score = float(row.get("relevance_score", 0.0))
            except Exception:
                continue

            if pr_id not in judgments:
                judgments[pr_id] = {}
            if cid:
                judgments[pr_id][cid] = score
            if fpath:
                judgments[pr_id][fpath] = score
    except Exception as e:
        logger.warning(f"Failed to load human relevance annotations from {annotations_path}: {e}")

    return judgments


def compute_dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K using graded relevance."""
    scores = relevance_scores[:k]
    dcg = 0.0
    for i, rel in enumerate(scores):
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return dcg


def compute_ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = compute_dcg_at_k(ideal_scores, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_rag_retrieval(
    config_id: str = "E6",
    raw_dir: Path = RAW_RESULTS_DIR,
    diffs_dir: Path = DIFFS_DIR,
    manifest_path: Path = MANIFEST_PATH,
    annotations_path: Path = HUMAN_ANNOTATIONS_PATH,
    top_ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality across all PRs for a given RAG configuration.
    Strictly distinguishes:
    1. DFOR@K: diff-file overlap ratio (retrieval-coverage proxy).
    2. Semantic MRR / nDCG@K: Requires human relevance judgments. If absent,
       explicitly marked as PENDING_HUMAN_ANNOTATION.
    """
    cfg_dir = raw_dir / config_id
    if not cfg_dir.exists():
        logger.warning(f"Raw results directory not found for {config_id}: {cfg_dir}")
        return {}

    # Load human annotations if available
    human_judgments = load_human_relevance_annotations(annotations_path)
    has_human_annotations = len(human_judgments) > 0

    # Load diff files map
    pr_diff_files: Dict[str, Set[str]] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status", "").upper() == "SUCCESS":
                    diff_file = Path(row["diff_file"])
                    pr_id = diff_file.stem
                    if diff_file.exists():
                        content = diff_file.read_text(encoding="utf-8", errors="replace")
                        pr_diff_files[pr_id] = extract_files_from_diff(content)

    dfor_at_k: Dict[int, List[float]] = {k: [] for k in top_ks}
    semantic_reciprocal_ranks = []
    semantic_ndcg_at_k: Dict[int, List[float]] = {k: [] for k in top_ks}
    total_chunks_retrieved = 0
    prs_with_context = 0

    for json_file in sorted(cfg_dir.glob("*.json")):
        if json_file.name == "execution_manifest.json":
            continue

        pr_id = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {json_file}: {e}")
            continue

        # Look for retrieved RAG chunks in execution_manifest or findings
        retrieved_chunks = []
        exec_manifest = data.get("execution_manifest", {})
        if "rag_context_snippets" in exec_manifest:
            retrieved_chunks = exec_manifest["rag_context_snippets"]
        elif "rag_context" in data:
            retrieved_chunks = data["rag_context"]

        if not retrieved_chunks:
            for finding in data.get("findings", []):
                chunks = finding.get("related_context_chunks", [])
                if chunks:
                    retrieved_chunks.extend(chunks)

        if not retrieved_chunks:
            continue

        prs_with_context += 1
        total_chunks_retrieved += len(retrieved_chunks)

        # 1. DFOR@K (Coverage proxy)
        modified_files = pr_diff_files.get(pr_id, set())
        overlap_vector = []
        for chunk in retrieved_chunks:
            chunk_file = ""
            if isinstance(chunk, dict):
                chunk_file = chunk.get("file_path", "") or chunk.get("file", "")
            elif isinstance(chunk, str):
                m = re.search(r"File:\s*([^\s\n]+)", chunk)
                if m:
                    chunk_file = m.group(1)

            norm_chunk_file = chunk_file.strip().replace("\\", "/").lower()
            is_overlap = 1.0 if (norm_chunk_file and any(
                norm_chunk_file in mf or mf in norm_chunk_file for mf in modified_files
            )) else 0.0
            overlap_vector.append(is_overlap)

        for k in top_ks:
            top_k_overlap = overlap_vector[:k]
            overlap_count = sum(top_k_overlap)
            dfor = (overlap_count / min(k, len(top_k_overlap))) if top_k_overlap else 0.0
            dfor_at_k[k].append(dfor)

        # 2. Semantic Relevance (if human annotations exist)
        if has_human_annotations and pr_id in human_judgments:
            pr_scores = human_judgments[pr_id]
            relevance_vector = []
            first_rel_rank = 0
            for rank_idx, chunk in enumerate(retrieved_chunks, 1):
                cid = chunk.get("chunk_id", "") if isinstance(chunk, dict) else ""
                cfile = (chunk.get("file_path", "") if isinstance(chunk, dict) else "").strip().replace("\\", "/").lower()
                rel_score = pr_scores.get(cid, pr_scores.get(cfile, 0.0))
                relevance_vector.append(rel_score)
                if rel_score > 0 and first_rel_rank == 0:
                    first_rel_rank = rank_idx

            semantic_reciprocal_ranks.append(1.0 / first_rel_rank if first_rel_rank > 0 else 0.0)
            for k in top_ks:
                semantic_ndcg_at_k[k].append(compute_ndcg_at_k(relevance_vector, k))

    summary: Dict[str, Any] = {
        "config_id": config_id,
        "prs_evaluated": prs_with_context,
        "total_chunks_evaluated": total_chunks_retrieved,
        "dfor_description": "Diff-File Overlap Ratio: retrieval coverage proxy (Protocol Sec 6.1)",
    }

    for k in top_ks:
        summary[f"dfor_coverage_proxy@{k}"] = round(float(np.mean(dfor_at_k[k])), 4) if dfor_at_k[k] else 0.0

    if has_human_annotations and semantic_reciprocal_ranks:
        summary["semantic_mrr"] = round(float(np.mean(semantic_reciprocal_ranks)), 4)
        for k in top_ks:
            summary[f"semantic_ndcg@{k}"] = round(float(np.mean(semantic_ndcg_at_k[k])), 4) if semantic_ndcg_at_k[k] else 0.0
        summary["semantic_annotation_status"] = "COMPLETE"
    else:
        summary["semantic_mrr"] = "PENDING_HUMAN_ANNOTATION"
        for k in top_ks:
            summary[f"semantic_ndcg@{k}"] = "PENDING_HUMAN_ANNOTATION"
        summary["semantic_annotation_status"] = "PENDING_HUMAN_ANNOTATION"
        summary["annotation_template"] = "evaluation/datasets/rag_relevance_annotation_template.csv"

    return summary


def run_rag_evaluations(
    configs: Tuple[str, ...] = ("E4", "E6"),
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Run RAG evaluation across configurations that employ repository-aware RAG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for cfg in configs:
        logger.info(f"Evaluating RAG metrics for: {cfg}")
        res = evaluate_rag_retrieval(config_id=cfg)
        if res:
            rows.append(res)

    if not rows:
        # Default placeholder if runs have not been completed yet
        rows.append({
            "config_id": "E6 (No data yet)",
            "prs_evaluated": 0,
            "total_chunks_evaluated": 0,
            "dfor_description": "Diff-File Overlap Ratio: retrieval coverage proxy (Protocol Sec 6.1)",
            "dfor_coverage_proxy@1": 0.0,
            "dfor_coverage_proxy@3": 0.0,
            "dfor_coverage_proxy@5": 0.0,
            "semantic_mrr": "PENDING_HUMAN_ANNOTATION",
            "semantic_ndcg@1": "PENDING_HUMAN_ANNOTATION",
            "semantic_ndcg@3": "PENDING_HUMAN_ANNOTATION",
            "semantic_ndcg@5": "PENDING_HUMAN_ANNOTATION",
            "semantic_annotation_status": "PENDING_HUMAN_ANNOTATION",
            "annotation_template": "evaluation/datasets/rag_relevance_annotation_template.csv",
        })

    df = pd.DataFrame(rows)
    out_csv = output_dir / "rag_evaluation_metrics.csv"
    df.to_csv(out_csv, index=False)

    print("\n===== REPOSITORY-AWARE RAG EVALUATION RESULTS =====\n")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="PR-Sentinel RAG Evaluation")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["E4", "E6"],
        help="Configurations to evaluate for RAG metrics (default: E4 E6)",
    )
    args = parser.parse_args()
    run_rag_evaluations(configs=tuple(args.configs))


if __name__ == "__main__":
    main()
