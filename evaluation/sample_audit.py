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
    "CAT-A": "Hallucination / Erroneous Finding: False alarm, invalid rule firing, syntax misunderstanding, incorrect or hallucinated recommendation (Confirmed False Positive).",
    "CAT-B": "Latent Valid Defect: Genuine defect or security vulnerability missed by benchmark annotations (Reclassified True Positive).",
    "CAT-C": "Benign Quality Observation / Nitpick: Stylistic suggestion, non-blocking convention, minor refactoring, or subjective observation.",
}


def calculate_adjusted_precision(
    tp_benchmark: int,
    fp_benchmark: int,
    n_A: int,
    n_B: int,
    n_C: int,
) -> Dict[str, Any]:
    """
    Computes manual-audit estimated adjusted precision:
        P_adj = (TP + rho_B * FP_apparent) / (TP + FP_apparent)
    where rho_B = n_B / (n_A + n_B + n_C).

    NOTE: This is an estimated precision under manual audit sampling, NOT a
    mathematically guaranteed lower bound unless verified under stated assumptions.
    """
    total_samples = n_A + n_B + n_C
    rho_B = (n_B / total_samples) if total_samples > 0 else 0.0
    denom = tp_benchmark + fp_benchmark
    p_apparent = (tp_benchmark / denom) if denom > 0 else 0.0
    p_adjusted = ((tp_benchmark + (rho_B * fp_benchmark)) / denom) if denom > 0 else 0.0

    return {
        "tp_benchmark": tp_benchmark,
        "fp_benchmark": fp_benchmark,
        "n_A": n_A,
        "n_B": n_B,
        "n_C": n_C,
        "total_audit_samples": total_samples,
        "rho_B_latent_rate": round(rho_B, 4),
        "apparent_precision": round(p_apparent, 4),
        "adjusted_precision": round(p_adjusted, 4),
        "precision_interpretation": (
            "Estimated precision under manual audit sampling. Not a mathematically "
            "guaranteed lower bound."
        ),
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
    """
    Proportional stratified sampling across severity and repository/category strata.
    Guarantees exact proportional quota allocation without quadratic density distortion.
    """
    if not fps:
        return []

    if len(fps) <= n_samples:
        return list(fps)

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(fps)

    # Determine stratum key: severity + category (or pr_id if category missing)
    strat_cols = [c for c in ["severity", "category"] if c in df.columns]
    if not strat_cols:
        chosen_indices = rng.choice(len(df), size=n_samples, replace=False)
        return df.iloc[chosen_indices].to_dict(orient="records")

    # True proportional allocation per stratum
    total_N = len(df)
    strata = df.groupby(strat_cols).indices
    allocated_samples = []

    # Calculate integer quotas
    stratum_quotas = {}
    remainders = {}
    for s_key, indices in strata.items():
        prop = len(indices) * n_samples / total_N
        base_quota = int(np.floor(prop))
        stratum_quotas[s_key] = base_quota
        remainders[s_key] = prop - base_quota

    # Distribute remaining slots by largest remainder
    slots_left = n_samples - sum(stratum_quotas.values())
    sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    for i in range(slots_left):
        stratum_quotas[sorted_remainders[i % len(sorted_remainders)][0]] += 1

    for s_key, indices in strata.items():
        quota = min(stratum_quotas[s_key], len(indices))
        if quota > 0:
            chosen = rng.choice(indices, size=quota, replace=False)
            allocated_samples.extend(chosen)

    # Safety: if slight rounding mismatch, fill or trim
    if len(allocated_samples) < n_samples:
        remaining_pool = list(set(range(len(df))) - set(allocated_samples))
        needed = n_samples - len(allocated_samples)
        extra = rng.choice(remaining_pool, size=min(needed, len(remaining_pool)), replace=False)
        allocated_samples.extend(extra)
    elif len(allocated_samples) > n_samples:
        allocated_samples = list(rng.choice(allocated_samples, size=n_samples, replace=False))

    return df.iloc[allocated_samples].to_dict(orient="records")


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

    tier1_adequacy = "SUFFICIENT" if len(e6_fps) >= tier1_size else f"INSUFFICIENT (M={len(e6_fps)} < {tier1_size})"
    tier1_sampled = sample_candidates(e6_fps, n_samples=tier1_size, seed=seed)
    tier1_formatted = format_audit_records(tier1_sampled, "TIER1_E6")
    tier1_file = audit_dir / "tier1_e6_sample_50.csv"
    if tier1_formatted:
        pd.DataFrame(tier1_formatted).to_csv(tier1_file, index=False)
        logger.info(f"Saved {len(tier1_formatted)} Tier 1 audit records to {tier1_file} (Adequacy: {tier1_adequacy})")
    else:
        logger.warning(f"No Tier 1 samples extracted ({tier1_adequacy}).")

    # --------------------------------------------------------
    # TIER 2: Baselines E1, E2, E3, E5 (n=15 each, total up to 60)
    # --------------------------------------------------------
    tier2_configs = ["E1", "E2", "E3", "E5"]
    tier2_all_formatted = []
    tier2_adequacy = {}
    for cfg in tier2_configs:
        logger.info(f"Extracting apparent false positives for Tier 2 ({cfg})...")
        cfg_fps = extract_apparent_fps(cfg, gt_by_pr)
        logger.info(f"Total apparent FPs in {cfg}: {len(cfg_fps)}")
        tier2_adequacy[cfg] = "SUFFICIENT" if len(cfg_fps) >= tier2_per_config else f"INSUFFICIENT (M={len(cfg_fps)} < {tier2_per_config})"
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
        "tier1_sample_adequacy": tier1_adequacy,
        "tier1_file": str(tier1_file.name),
        "tier2_total_sampled_count": len(tier2_all_formatted),
        "tier2_sample_adequacy": tier2_adequacy,
        "tier2_file": str(tier2_file.name),
        "audit_status": "PENDING_HUMAN_REVIEW",
        "instruction": "Two independent reviewers must populate reviewer_1_label and reviewer_2_label independently using CAT-A, CAT-B, CAT-C.",
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
