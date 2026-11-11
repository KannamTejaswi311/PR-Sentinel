"""
PR-Sentinel: Research Metric Evaluation (Protocol v1.0 & Amendment v1.1)
Evaluates defect detection performance across configurations E1-E6 and ablations
against normalized ground-truth cohorts using Maximum Bipartite Matching (M1/M2/M3).
Computes:
- Ground-Truth Recall (R_GT)
- Benchmark-Apparent Precision (P_app)
- Benchmark-Apparent F1 (F1_app)
- 95% Clustered Bootstrap Confidence Intervals (B=1000 PR-level resamples)
- Paired Wilcoxon signed-rank tests against baselines
STRICT CONSTRAINT: Never reports Accuracy, False Positive Rate (FPR), or True Negatives.
"""

import argparse
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.matching_engine import (
    BipartiteMatchingEngine,
    MatchingProtocol,
    MatchingResult,
)
from sentinel.schemas import (
    GroundTruthIssue,
    ModelFinding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateDetection")

BASE_DIR = Path(__file__).resolve().parent
COHORTS_FILE = BASE_DIR / "datasets" / "normalized" / "ground_truth_cohorts.csv"
RAW_RESULTS_DIR = BASE_DIR / "results" / "frozen_protocol_v1" / "raw"
OUTPUT_DIR = BASE_DIR / "results" / "frozen_protocol_v1"

ALL_CONFIGS = ["E1", "E2", "E3", "E4", "E5", "E6", "E6_A1", "E6_A2", "E6_A3", "E6_A4"]


def load_ground_truth(
    cohorts_path: Path = COHORTS_FILE,
    include_candidate_cohort2: bool = False,
) -> Dict[str, List[GroundTruthIssue]]:
    """
    Load ground-truth issues from normalized cohorts CSV, grouped by PR ID.
    By default loads Cohort 1 (Confirmed, verifiable).
    If include_candidate_cohort2 is True, loads Cohort 1 + Cohort 2 for sensitivity analysis.
    """
    if not cohorts_path.exists():
        raise FileNotFoundError(f"Cohort ground truth file not found: {cohorts_path}")

    df = pd.read_csv(cohorts_path)
    allowed_cohorts = ["COHORT_1_CONFIRMED"]
    if include_candidate_cohort2:
        allowed_cohorts.append("COHORT_2_CANDIDATE")

    # Filter to requested cohorts and verifiable records
    filtered = df[df["cohort"].isin(allowed_cohorts) & (df["verifiable"] == True)]

    pr_to_gt: Dict[str, List[GroundTruthIssue]] = {}
    for _, row in filtered.iterrows():
        repo = str(row.get("repo", "")).strip()
        pr_num = row.get("pr_number", 0)
        pr_id = f"{repo}_pr{pr_num}"
        issue = GroundTruthIssue(
            issue_id=str(row["issue_id"]),
            repo=repo,
            pr_number=int(pr_num),
            pr_url=str(row.get("pr_url", "")),
            cohort=str(row["cohort"]),
            verifiable=bool(row["verifiable"]),
            file_path=str(row["file_path"]) if pd.notna(row["file_path"]) else None,
            start_line=int(row["start_line"]) if pd.notna(row["start_line"]) else None,
            end_line=int(row["end_line"]) if pd.notna(row["end_line"]) else None,
            rule_name=str(row["rule_name"]) if pd.notna(row["rule_name"]) else "",
            problematic_code=str(row.get("problematic_code", "")) if pd.notna(row.get("problematic_code")) else "",
            standardized_category=str(row["standardized_category"]) if pd.notna(row["standardized_category"]) else "UNKNOWN",
        )
        pr_to_gt.setdefault(pr_id, []).append(issue)

    return pr_to_gt


def load_raw_predictions(
    config_id: str,
    raw_dir: Path = RAW_RESULTS_DIR,
) -> Dict[str, List[ModelFinding]]:
    """Load raw JSON prediction files for a given configuration, grouped by PR ID."""
    cfg_dir = raw_dir / config_id
    if not cfg_dir.exists():
        logger.warning(f"Raw results directory not found for {config_id}: {cfg_dir}")
        return {}

    pr_to_preds: Dict[str, List[ModelFinding]] = {}
    for json_file in cfg_dir.glob("*.json"):
        if json_file.name == "execution_manifest.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            pr_id = data.get("pr_id") or json_file.stem
            preds: List[ModelFinding] = []
            for item in data.get("findings", []):
                preds.append(
                    ModelFinding(
                        pr_id=pr_id,
                        source=item.get("source", config_id),
                        category=item.get("category", "UNKNOWN"),
                        file_path=item.get("file_path", ""),
                        line_start=int(item.get("line_start", 1)),
                        line_end=int(item.get("line_end", item.get("line_start", 1))),
                        evidence=item.get("evidence", ""),
                        execution_status=item.get("execution_status", "SUCCESS"),
                        finding_id=item.get("finding_id", ""),
                        rule_id=item.get("rule_id", ""),
                        severity=item.get("severity", "MEDIUM"),
                        confidence=item.get("confidence"),
                        description=item.get("description", ""),
                        recommendation=item.get("recommendation", ""),
                        granularity=item.get("granularity", "STATEMENT"),
                        metadata=item.get("metadata", {}),
                    )
                )
            pr_to_preds[pr_id] = preds
        except Exception as e:
            logger.error(f"Error reading prediction file {json_file}: {e}")

    return pr_to_preds


def compute_aggregate_metrics(pr_results: List[MatchingResult]) -> Dict[str, Any]:
    """Aggregate TP, FP, FN across all PRs and compute aggregate precision, recall, F1."""
    total_tp = sum(r.tp for r in pr_results)
    total_fp = sum(r.fp for r in pr_results)
    total_fn = sum(r.fn for r in pr_results)

    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "gt_recall": recall,
        "benchmark_apparent_precision": precision,
        "benchmark_apparent_f1": f1,
    }


def compute_bootstrap_cis(
    pr_results: List[MatchingResult],
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% confidence intervals via clustered PR-level bootstrap resampling.
    Resamples entire PRs with replacement B=1000 times.
    """
    if not pr_results:
        return {
            "gt_recall_ci": (0.0, 0.0),
            "precision_ci": (0.0, 0.0),
            "f1_ci": (0.0, 0.0),
        }

    rng = np.random.default_rng(seed)
    n_prs = len(pr_results)

    recalls = []
    precisions = []
    f1s = []

    for _ in range(num_bootstraps):
        sampled_indices = rng.integers(0, n_prs, size=n_prs)
        sampled_results = [pr_results[i] for i in sampled_indices]
        agg = compute_aggregate_metrics(sampled_results)
        recalls.append(agg["gt_recall"])
        precisions.append(agg["benchmark_apparent_precision"])
        f1s.append(agg["benchmark_apparent_f1"])

    return {
        "gt_recall_ci": (float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))),
        "precision_ci": (float(np.percentile(precisions, 2.5)), float(np.percentile(precisions, 97.5))),
        "f1_ci": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
    }


def evaluate_configuration(
    config_id: str,
    gt_by_pr: Dict[str, List[GroundTruthIssue]],
    preds_by_pr: Dict[str, List[ModelFinding]],
    protocol: MatchingProtocol = MatchingProtocol.M1_STRICT,
) -> Tuple[Dict[str, Any], List[MatchingResult]]:
    """Match predictions to ground truth for all PRs under the chosen protocol."""
    engine = BipartiteMatchingEngine(primary_protocol=protocol)
    pr_results: List[MatchingResult] = []

    # All unique PR IDs in either GT or Predictions
    all_pr_ids = sorted(set(gt_by_pr.keys()).union(set(preds_by_pr.keys())))

    for pr_id in all_pr_ids:
        prs_gt = gt_by_pr.get(pr_id, [])
        prs_preds = preds_by_pr.get(pr_id, [])
        res = engine.match(predictions=prs_preds, ground_truth=prs_gt, protocol=protocol, pr_id=pr_id)
        pr_results.append(res)

    agg = compute_aggregate_metrics(pr_results)
    cis = compute_bootstrap_cis(pr_results)

    summary = {
        "config_id": config_id,
        "protocol": protocol.value,
        "evaluated_prs": len(all_pr_ids),
        "tp": agg["tp"],
        "fp": agg["fp"],
        "fn": agg["fn"],
        "gt_recall": round(agg["gt_recall"] * 100, 2),
        "gt_recall_ci_95": f"[{round(cis['gt_recall_ci'][0] * 100, 2)}, {round(cis['gt_recall_ci'][1] * 100, 2)}]",
        "benchmark_apparent_precision": round(agg["benchmark_apparent_precision"] * 100, 2),
        "precision_ci_95": f"[{round(cis['precision_ci'][0] * 100, 2)}, {round(cis['precision_ci'][1] * 100, 2)}]",
        "benchmark_apparent_f1": round(agg["benchmark_apparent_f1"] * 100, 2),
        "f1_ci_95": f"[{round(cis['f1_ci'][0] * 100, 2)}, {round(cis['f1_ci'][1] * 100, 2)}]",
    }

    return summary, pr_results


def run_evaluation(
    protocols: Optional[List[MatchingProtocol]] = None,
    include_candidate_cohort2: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, pd.DataFrame]:
    """Run full evaluation matrix across all available configurations and protocols."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_protocols = protocols or [
        MatchingProtocol.M1_STRICT,
        MatchingProtocol.M2_TOLERANT,
        MatchingProtocol.M3_SEMANTIC,
    ]

    cohort_label = "Cohort1_Confirmed" if not include_candidate_cohort2 else "Cohort1_and_Cohort2"
    logger.info(f"Loading Ground Truth ({cohort_label})...")
    gt_by_pr = load_ground_truth(include_candidate_cohort2=include_candidate_cohort2)
    logger.info(f"Loaded ground truth for {len(gt_by_pr)} PRs.")

    protocol_dfs: Dict[str, pd.DataFrame] = {}

    for proto in target_protocols:
        logger.info(f"\n--- Evaluating Protocol: {proto.value} ---")
        rows = []
        per_config_pr_results: Dict[str, List[MatchingResult]] = {}

        for cfg in ALL_CONFIGS:
            preds_by_pr = load_raw_predictions(cfg)
            if not preds_by_pr:
                continue

            summary, pr_res = evaluate_configuration(cfg, gt_by_pr, preds_by_pr, protocol=proto)
            rows.append(summary)
            per_config_pr_results[cfg] = pr_res

        if not rows:
            logger.warning(f"No prediction data found for any configuration under {proto.value}.")
            continue

        df_summary = pd.DataFrame(rows)

        # Compute Wilcoxon signed-rank test comparing E6 against each baseline
        if "E6" in per_config_pr_results:
            e6_prs = {r.pr_id: r.benchmark_apparent_f1 for r in per_config_pr_results["E6"]}
            wilcoxon_results = {}
            for other_cfg, other_res in per_config_pr_results.items():
                if other_cfg == "E6":
                    continue
                other_prs = {r.pr_id: r.benchmark_apparent_f1 for r in other_res}
                common_ids = sorted(set(e6_prs.keys()).intersection(set(other_prs.keys())))
                diffs = [e6_prs[pid] - other_prs[pid] for pid in common_ids]
                
                # Wilcoxon requires non-zero differences
                non_zeros = [d for d in diffs if d != 0]
                if len(non_zeros) >= 5:
                    stat, p_val = stats.wilcoxon(diffs, zero_method="pratt")
                    wilcoxon_results[other_cfg] = round(p_val, 4)
                else:
                    wilcoxon_results[other_cfg] = "N/A (<5 non-zero diffs)"

            df_summary["wilcoxon_p_vs_E6"] = df_summary["config_id"].map(
                lambda c: wilcoxon_results.get(c, "Baseline") if c != "E6" else "Reference (E6)"
            )

        csv_name = f"metrics_summary_{proto.value.lower()}_{cohort_label.lower()}.csv"
        csv_path = output_dir / csv_name
        df_summary.to_csv(csv_path, index=False)
        protocol_dfs[proto.value] = df_summary

        print(f"\n===== RESULTS SUMMARY [{proto.value}] ({cohort_label}) =====\n")
        print(df_summary.to_string(index=False))

    return protocol_dfs


def main():
    parser = argparse.ArgumentParser(description="PR-Sentinel Research Metric Evaluator")
    parser.add_argument(
        "--protocol",
        choices=["M1", "M2", "M3", "ALL"],
        default="ALL",
        help="Matching protocol to evaluate (default: ALL)",
    )
    parser.add_argument(
        "--include_candidate_cohort2",
        action="store_true",
        help="Include Cohort 2 (candidate 16 issues) for sensitivity analysis",
    )
    args = parser.parse_args()

    protos = None if args.protocol == "ALL" else [MatchingProtocol(args.protocol)]
    run_evaluation(protocols=protos, include_candidate_cohort2=args.include_candidate_cohort2)


if __name__ == "__main__":
    main()
