"""
PR-Sentinel: Research Evaluation - Bipartite Matching Engine
Implements Maximum Weight Bipartite Matching (via scipy Hungarian algorithm)
for 1:1 attribution between model predictions and ground-truth issues under
ratified Protocols M1 (Strict), M2 (Tolerant), and M3 (Semantic).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.schemas import (
    FindingCategory,
    FindingGranularity,
    GroundTruthIssue,
    ModelFinding,
)


class MatchingProtocol(str, Enum):
    """Evaluation matching protocols ratified in Protocol v1.0 and Amendment v1.1."""
    M1_STRICT = "M1"        # Primary: exact file, |Δline| <= 3
    M2_TOLERANT = "M2"      # Sensitivity: exact file, |Δline| <= 10
    M3_SEMANTIC = "M3"      # Sensitivity: exact file, category match, token Jaccard >= 0.3


@dataclass
class MatchPair:
    """Record of an assigned prediction-to-ground-truth pair."""
    prediction_id: str
    ground_truth_id: str
    prediction_file: str
    ground_truth_file: str
    prediction_line: int
    ground_truth_line: int
    line_distance: int
    cost: float
    category_match: bool
    jaccard_similarity: float
    protocol: str


@dataclass
class MatchingResult:
    """Complete bipartite matching result for a PR or evaluation cohort."""
    protocol: MatchingProtocol
    pr_id: Optional[str]
    matched_pairs: List[MatchPair] = field(default_factory=list)
    unmatched_predictions: List[ModelFinding] = field(default_factory=list)
    unmatched_ground_truth: List[GroundTruthIssue] = field(default_factory=list)
    
    # Core Evaluation Metrics
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gt_recall: float = 0.0
    benchmark_apparent_precision: float = 0.0
    benchmark_apparent_f1: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol.value,
            "pr_id": self.pr_id,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "gt_recall": round(self.gt_recall, 4),
            "benchmark_apparent_precision": round(self.benchmark_apparent_precision, 4),
            "benchmark_apparent_f1": round(self.benchmark_apparent_f1, 4),
            "matched_pairs_count": len(self.matched_pairs),
            "unmatched_predictions_count": len(self.unmatched_predictions),
            "unmatched_ground_truth_count": len(self.unmatched_ground_truth),
        }


def normalize_file_path(path: str) -> str:
    """Normalize file path for cross-platform comparison."""
    if not path:
        return ""
    p = path.strip().replace("\\", "/").lower()
    p = re.sub(r"^[ab]/", "", p)  # Strip git a/ or b/ prefixes
    return p.lstrip("/")


def tokenize_text(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9_]{2,}", text.lower()))


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Compute token-level Jaccard similarity between two texts."""
    set1 = tokenize_text(text1)
    set2 = tokenize_text(text2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0


def compute_line_distance(
    pred_start: int,
    pred_end: int,
    gt_line: int,
    granularity: Any = FindingGranularity.STATEMENT,
) -> int:
    """
    Compute line distance taking into account localization granularity.
    - If STATEMENT: |pred_start - gt_line|
    - If HUNK_SPAN: minimum distance to span [pred_start, pred_end]
      (i.e. 0 if gt_line falls inside span, else min(|gt - start|, |gt - end|))
    """
    gran_val = granularity.value if hasattr(granularity, "value") else str(granularity)
    if gran_val == FindingGranularity.HUNK_SPAN.value:
        if pred_start <= gt_line <= pred_end:
            return 0
        return min(abs(gt_line - pred_start), abs(gt_line - pred_end))
    
    # Default statement line-specific
    return abs(pred_start - gt_line)


class BipartiteMatchingEngine:
    """
    Global optimal bipartite matching engine between ModelFinding predictions
    and GroundTruthIssue annotations.
    """

    INF_COST: float = 1e6
    MATCH_THRESHOLD: float = 1e5

    def __init__(self, primary_protocol: MatchingProtocol = MatchingProtocol.M1_STRICT):
        self.primary_protocol = primary_protocol

    def match(
        self,
        predictions: List[ModelFinding],
        ground_truth: List[GroundTruthIssue],
        protocol: Optional[MatchingProtocol] = None,
        pr_id: Optional[str] = None,
    ) -> MatchingResult:
        """
        Execute Hungarian algorithm to assign predictions to ground truth 1:1.
        """
        proto = protocol or self.primary_protocol
        num_preds = len(predictions)
        num_gts = len(ground_truth)

        # Edge cases: no predictions or no ground truth
        if num_preds == 0 or num_gts == 0:
            tp = 0
            fp = num_preds
            fn = num_gts
            recall = 0.0
            precision = 0.0
            f1 = 0.0

            return MatchingResult(
                protocol=proto,
                pr_id=pr_id,
                matched_pairs=[],
                unmatched_predictions=list(predictions),
                unmatched_ground_truth=list(ground_truth),
                tp=tp,
                fp=fp,
                fn=fn,
                gt_recall=recall,
                benchmark_apparent_precision=precision,
                benchmark_apparent_f1=f1,
            )

        # Construct Cost Matrix of shape (num_preds, num_gts)
        cost_matrix = np.full((num_preds, num_gts), self.INF_COST, dtype=float)
        pair_details: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for i, pred in enumerate(predictions):
            pred_file_norm = normalize_file_path(pred.file_path)

            for j, gt in enumerate(ground_truth):
                gt_file_norm = normalize_file_path(gt.file_path)

                # Requirement 1: File path must match
                if pred_file_norm != gt_file_norm:
                    continue

                if not gt.file_path or gt.start_line is None:
                    continue

                line_dist = compute_line_distance(
                    pred.line_start,
                    pred.line_end or pred.line_start,
                    gt.start_line,
                    pred.granularity,
                )

                # Protocol-specific feasibility and cost formulation
                is_feasible = False
                cost = self.INF_COST

                pred_text = f"{pred.rule_id or ''} {pred.description}"
                gt_text = f"{gt.rule_name or ''} {gt.problematic_code or ''}"
                jaccard = compute_token_jaccard(pred_text, gt_text)
                
                pred_cat = pred.category.value if hasattr(pred.category, "value") else str(pred.category)
                gt_cat = gt.standardized_category.value if hasattr(gt.standardized_category, "value") else str(gt.standardized_category)
                cat_match = (pred_cat.upper() == gt_cat.upper())

                if proto == MatchingProtocol.M1_STRICT:
                    # Strict line tolerance: |Δline| <= 3
                    if line_dist <= 3:
                        is_feasible = True
                        # Cost: primarily line distance, secondary tie-breaker by (1 - jaccard)
                        cost = float(line_dist) + (1.0 - jaccard) * 0.1

                elif proto == MatchingProtocol.M2_TOLERANT:
                    # Tolerant line tolerance: |Δline| <= 10
                    if line_dist <= 10:
                        is_feasible = True
                        cost = float(line_dist) + (1.0 - jaccard) * 0.1

                elif proto == MatchingProtocol.M3_SEMANTIC:
                    # Semantic: category match AND token Jaccard >= 0.3
                    # (Permits cross-hunk semantic matches within same file)
                    if cat_match and jaccard >= 0.30:
                        is_feasible = True
                        # Cost: lower cost for higher jaccard + mild distance penalty
                        cost = (1.0 - jaccard) * 10.0 + min(float(line_dist) * 0.05, 5.0)

                if is_feasible:
                    cost_matrix[i, j] = cost
                    pair_details[(i, j)] = {
                        "line_distance": line_dist,
                        "cost": cost,
                        "category_match": cat_match,
                        "jaccard": jaccard,
                    }

        # Solve via Hungarian Algorithm (Linear Sum Assignment)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[MatchPair] = []
        matched_pred_indices: Set[int] = set()
        matched_gt_indices: Set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.MATCH_THRESHOLD:
                matched_pred_indices.add(r)
                matched_gt_indices.add(c)
                details = pair_details.get((r, c), {
                    "line_distance": 0,
                    "cost": cost_matrix[r, c],
                    "category_match": False,
                    "jaccard": 0.0,
                })
                pred = predictions[r]
                gt = ground_truth[c]
                matched_pairs.append(
                    MatchPair(
                        prediction_id=pred.finding_id,
                        ground_truth_id=gt.issue_id,
                        prediction_file=pred.file_path,
                        ground_truth_file=gt.file_path or "",
                        prediction_line=pred.line_start,
                        ground_truth_line=gt.start_line or 0,
                        line_distance=details["line_distance"],
                        cost=details["cost"],
                        category_match=details["category_match"],
                        jaccard_similarity=details["jaccard"],
                        protocol=proto.value,
                    )
                )

        unmatched_preds = [pred for i, pred in enumerate(predictions) if i not in matched_pred_indices]
        unmatched_gts = [gt for j, gt in enumerate(ground_truth) if j not in matched_gt_indices]

        tp = len(matched_pairs)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)

        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return MatchingResult(
            protocol=proto,
            pr_id=pr_id,
            matched_pairs=matched_pairs,
            unmatched_predictions=unmatched_preds,
            unmatched_ground_truth=unmatched_gts,
            tp=tp,
            fp=fp,
            fn=fn,
            gt_recall=recall,
            benchmark_apparent_precision=precision,
            benchmark_apparent_f1=f1,
        )
