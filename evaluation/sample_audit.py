"""
PR-Sentinel: Research Manual Audit Sampling Engine (Protocol v1.0 & Amendment v1.1)
Samples apparent false positives (unmatched predictions under M1) for blinded,
two-reviewer manual human audit verification without fabricating review labels.

Implements Two-Tier Architecture:
- Tier 1: n=50 sample exclusively from primary system E6 for quantitative adjusted precision.
- Tier 2: n=15 sample each from E1, E2, E3, E5 (total 60) for qualitative error taxonomy.
STRICT RULE: Never pool Tier 2 into Tier 1 precision estimates.
STRICT RULE: Never fabricate reviewer verdicts; marks all samples as PENDING_HUMAN_REVIEW.
"""

import argparse
import csv
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.evaluate_detection import load_ground_truth, load_raw_predictions
from evaluation.matching_engine import BipartiteMatchingEngine, MatchingProtocol
from sentinel.schemas import ModelFinding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SampleAudit")

BASE_DIR = Path(__file__).resolve().parent
RAW_RESULTS_DIR = BASE_DIR / "results" / "frozen_protocol_v1" / "raw"
AUDIT_DIR = BASE_DIR / "results" / "frozen_protocol_v1" / "audit"

AUDIT_RUBRIC = {
    "CAT-A": "Valid Finding: Genuine defect or security flaw missed by benchmark ground truth annotations (True Positive).",
    "CAT-B": "True False Positive: Hallucination, invalid rule firing, syntax misunderstanding, or incorrect advice.",
    "CAT-C": "Borderline / Subjective: Stylistic suggestion, minor non-blocking convention, or debatable best practice.",
}


def extract_apparent_fps(
    config_id: str,
    gt_by_pr: Dict[str, Any],
    raw_dir: Path = RAW_RESULTS_DIR,
) -> List[Dict[str, Any]]:
    """Run bipartite matching under M1 and extract all unmatched predictions (apparent FPs)."""
    preds_by_pr = load_raw_predictions(config_id, raw_dir=raw_dir)
    if not preds_by_pr:
        logger.warning(f"No predictions found for configuration: {config_id}")
        return []

    engine = BipartiteMatchingEngine(primary_protocol=MatchingProtocol.M1_STRICT)
    apparent_fps: List[Dict[str, Any]] = []

    for pr_id, preds in preds_by_pr.items():
        gts = gt_by_pr.get(pr_id, [])
        match_res = engine.match(predictions=preds, ground_truth=gts, protocol=MatchingProtocol.M1_STRICT, pr_id=pr_id)
        for pred in match_res.unmatched_predictions:
            d = asdict(pred) if hasattr(pred, "to_dict") else dict(pred)
            d["configuration"] = config_id
            d["unmatched_protocol"] = "M1_STRICT"
            apparent_fps.append(d)

    return apparent_fps


def sample_candidates(
    fps: List[Dict[str, Any]],
    n_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample candidates using reproducible seeded pseudo-random selection with stratification."""
    if not fps:
        return []

    if len(fps) <= n_samples:
        return list(fps)

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(fps)

    # Attempt stratified sampling by category and severity
    if "category" in df.columns and "severity" in df.columns:
        groups = df.groupby(["category", "severity"], group_keys=False)
        # Proportionate allocation
        weights = df.groupby(["category", "severity"])["finding_id"].transform("count")
        prob = weights / weights.sum()
        chosen_indices = rng.choice(len(df), size=n_samples, replace=False, p=prob.values / prob.values.sum())
    else:
        chosen_indices = rng.choice(len(df), size=n_samples, replace=False)

    sampled_df = df.iloc[chosen_indices].copy()
    return sampled_df.to_dict(orient="records")


def format_audit_records(
    samples: List[Dict[str, Any]],
    tier_label: str,
) -> List[Dict[str, Any]]:
    """Format sampled findings with two-reviewer blinded rubric fields."""
    formatted = []
    for idx, s in enumerate(samples, 1):
        formatted.append({
            "sample_id": f"{tier_label}_{idx:03d}",
            "tier": tier_label,
            "configuration": s.get("configuration", ""),
            "pr_id": s.get("pr_id", ""),
            "finding_id": s.get("finding_id", ""),
            "file_path": s.get("file_path", ""),
            "line_start": s.get("line_start", ""),
            "line_end": s.get("line_end", ""),
            "category": s.get("category", ""),
            "severity": s.get("severity", ""),
            "rule_id": s.get("rule_id", ""),
            "description": s.get("description", ""),
            "evidence": s.get("evidence", ""),
            "recommendation": s.get("recommendation", ""),
            "audit_status": "PENDING_HUMAN_REVIEW",
            "reviewer_1_label": "",
            "reviewer_1_notes": "",
            "reviewer_2_label": "",
            "reviewer_2_notes": "",
            "adjudicated_label": "",
            "adjudication_notes": "",
        })
    return formatted


def generate_audit_samples(
    audit_dir: Path = AUDIT_DIR,
    tier1_size: int = 50,
    tier2_per_config: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate both Tier 1 and Tier 2 audit sample CSVs and manifest."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    gt_by_pr = load_ground_truth(include_candidate_cohort2=False)

    # --------------------------------------------------------
    # TIER 1: Primary System E6 (n=50)
    # --------------------------------------------------------
    logger.info("Extracting apparent false positives for Tier 1 (E6)...")
    e6_fps = extract_apparent_fps("E6", gt_by_pr)
    logger.info(f"Total apparent FPs in E6: {len(e6_fps)}")

    tier1_sampled = sample_candidates(e6_fps, n_samples=tier1_size, seed=seed)
    tier1_formatted = format_audit_records(tier1_sampled, "TIER1_E6")
    tier1_file = audit_dir / "tier1_e6_sample_50.csv"
    if tier1_formatted:
        pd.DataFrame(tier1_formatted).to_csv(tier1_file, index=False)
        logger.info(f"Saved {len(tier1_formatted)} Tier 1 audit records to {tier1_file}")
    else:
        logger.warning("No Tier 1 samples extracted (E6 predictions not available or empty).")

    # --------------------------------------------------------
    # TIER 2: Baselines E1, E2, E3, E5 (n=15 each, total up to 60)
    # --------------------------------------------------------
    tier2_configs = ["E1", "E2", "E3", "E5"]
    tier2_all_formatted = []
    for cfg in tier2_configs:
        logger.info(f"Extracting apparent false positives for Tier 2 ({cfg})...")
        cfg_fps = extract_apparent_fps(cfg, gt_by_pr)
        logger.info(f"Total apparent FPs in {cfg}: {len(cfg_fps)}")
        cfg_sampled = sample_candidates(cfg_fps, n_samples=tier2_per_config, seed=seed)
        formatted = format_audit_records(cfg_sampled, f"TIER2_{cfg}")
        tier2_all_formatted.extend(formatted)

    tier2_file = audit_dir / "tier2_baselines_sample_60.csv"
    if tier2_all_formatted:
        pd.DataFrame(tier2_all_formatted).to_csv(tier2_file, index=False)
        logger.info(f"Saved {len(tier2_all_formatted)} Tier 2 audit records to {tier2_file}")

    # Write Manifest
    manifest = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "rubric": AUDIT_RUBRIC,
        "tier1_e6_total_fps": len(e6_fps),
        "tier1_e6_sampled_count": len(tier1_formatted),
        "tier1_file": str(tier1_file.name),
        "tier2_total_sampled_count": len(tier2_all_formatted),
        "tier2_file": str(tier2_file.name),
        "audit_status": "PENDING_HUMAN_REVIEW",
        "instruction": "Reviewers must populate reviewer_1_label and reviewer_2_label independently using CAT-A, CAT-B, CAT-C.",
    }
    manifest_file = audit_dir / "manual_audit_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="PR-Sentinel Manual Audit Sampling")
    parser.add_argument("--tier1_size", type=int, default=50, help="Sample size for Tier 1 (E6)")
    parser.add_argument("--tier2_size", type=int, default=15, help="Sample size per baseline for Tier 2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility")
    args = parser.parse_args()

    manifest = generate_audit_samples(
        tier1_size=args.tier1_size,
        tier2_per_config=args.tier2_size,
        seed=args.seed,
    )
    print("\n===== MANUAL AUDIT SAMPLE MANIFEST =====\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
