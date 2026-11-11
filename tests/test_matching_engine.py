"""
Unit tests for evaluation/matching_engine.py:
Verifies Hungarian 1:1 bipartite assignment, M1/M2/M3 protocol constraints,
HUNK_SPAN line distance handling, and metric calculations.
"""

import pytest

from evaluation.matching_engine import (
    BipartiteMatchingEngine,
    MatchingProtocol,
    compute_line_distance,
    compute_token_jaccard,
    normalize_file_path,
)
from sentinel.schemas import (
    FindingCategory,
    FindingGranularity,
    GroundTruthIssue,
    ModelFinding,
    SeverityLevel,
)


def make_dummy_pred(
    finding_id: str,
    file_path: str,
    line_start: int,
    line_end: int = 0,
    category: FindingCategory = FindingCategory.SECURITY,
    granularity: FindingGranularity = FindingGranularity.STATEMENT,
    title: str = "SQL Injection Vulnerability",
    desc: str = "Unsanitized user input formatted directly into SQL query.",
) -> ModelFinding:
    return ModelFinding(
        pr_id="test_pr_1",
        source="MultiAgent",
        category=category.value,
        severity=SeverityLevel.HIGH.value,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end or line_start,
        rule_id=title,
        description=desc,
        evidence="query = 'SELECT * FROM users WHERE id=' + uid",
        granularity=granularity.value,
        finding_id=finding_id,
    )


def make_dummy_gt(
    record_id: str,
    file_path: str,
    line_number: int,
    category: FindingCategory = FindingCategory.SECURITY,
    title: str = "SQL Injection in User Query",
    desc: str = "Unsanitized user input executes arbitrary SQL statements.",
) -> GroundTruthIssue:
    return GroundTruthIssue(
        issue_id=record_id,
        repo="test/repo",
        pr_number=1,
        pr_url="https://github.com/test/repo/pull/1",
        cohort="COHORT_1_CONFIRMED",
        verifiable=True,
        file_path=file_path,
        start_line=line_number,
        end_line=line_number,
        rule_name=title,
        problematic_code=desc,
        standardized_category=category.value,
    )


def test_normalize_file_path():
    assert normalize_file_path("a/src/models/user.py") == "src/models/user.py"
    assert normalize_file_path("b/src\\models\\user.py") == "src/models/user.py"
    assert normalize_file_path("src/models/user.py") == "src/models/user.py"


def test_compute_token_jaccard():
    t1 = "SQL injection vulnerability in query execution"
    t2 = "Potential SQL injection in database query"
    score = compute_token_jaccard(t1, t2)
    assert 0.3 <= score <= 1.0

    t3 = "Completely unrelated memory leak warning"
    assert compute_token_jaccard(t1, t3) == 0.0


def test_compute_line_distance():
    # Statement specific
    assert compute_line_distance(10, 10, 10, FindingGranularity.STATEMENT) == 0
    assert compute_line_distance(10, 10, 13, FindingGranularity.STATEMENT) == 3
    assert compute_line_distance(10, 10, 14, FindingGranularity.STATEMENT) == 4

    # Hunk span [20, 30]
    assert compute_line_distance(20, 30, 25, FindingGranularity.HUNK_SPAN) == 0
    assert compute_line_distance(20, 30, 20, FindingGranularity.HUNK_SPAN) == 0
    assert compute_line_distance(20, 30, 30, FindingGranularity.HUNK_SPAN) == 0
    assert compute_line_distance(20, 30, 33, FindingGranularity.HUNK_SPAN) == 3
    assert compute_line_distance(20, 30, 35, FindingGranularity.HUNK_SPAN) == 5


def test_m1_strict_matching():
    engine = BipartiteMatchingEngine(MatchingProtocol.M1_STRICT)

    # 1. Exact match
    p1 = make_dummy_pred("p1", "app.py", 10)
    g1 = make_dummy_gt("g1", "app.py", 10)
    res = engine.match([p1], [g1], MatchingProtocol.M1_STRICT)
    assert res.tp == 1
    assert res.fp == 0
    assert res.fn == 0
    assert res.gt_recall == 1.0
    assert res.benchmark_apparent_precision == 1.0

    # 2. Within tolerance (|10 - 13| = 3 <= 3)
    p2 = make_dummy_pred("p2", "app.py", 13)
    res = engine.match([p2], [g1], MatchingProtocol.M1_STRICT)
    assert res.tp == 1
    assert res.fp == 0

    # 3. Beyond tolerance (|10 - 14| = 4 > 3)
    p3 = make_dummy_pred("p3", "app.py", 14)
    res = engine.match([p3], [g1], MatchingProtocol.M1_STRICT)
    assert res.tp == 0
    assert res.fp == 1
    assert res.fn == 1


def test_m2_tolerant_matching():
    engine = BipartiteMatchingEngine(MatchingProtocol.M2_TOLERANT)

    # Line distance = 7 (exceeds M1 <= 3, satisfies M2 <= 10)
    p = make_dummy_pred("p1", "app.py", 17)
    g = make_dummy_gt("g1", "app.py", 10)

    res_m1 = engine.match([p], [g], MatchingProtocol.M1_STRICT)
    assert res_m1.tp == 0

    res_m2 = engine.match([p], [g], MatchingProtocol.M2_TOLERANT)
    assert res_m2.tp == 1
    assert res_m2.fp == 0
    assert res_m2.matched_pairs[0].line_distance == 7


def test_bipartite_one_to_one_attribution():
    """
    Two predictions (p1, p2) close to one ground truth (g1).
    Hungarian matching must award g1 to p1 (closer), leaving p2 as FP.
    """
    engine = BipartiteMatchingEngine(MatchingProtocol.M1_STRICT)

    p1 = make_dummy_pred("p1", "app.py", 11)  # dist 1
    p2 = make_dummy_pred("p2", "app.py", 12)  # dist 2
    g1 = make_dummy_gt("g1", "app.py", 10)

    res = engine.match([p1, p2], [g1], MatchingProtocol.M1_STRICT)
    assert res.tp == 1
    assert res.fp == 1
    assert res.fn == 0
    assert len(res.matched_pairs) == 1
    assert res.matched_pairs[0].prediction_id == "p1"
    assert res.unmatched_predictions[0].finding_id == "p2"


def test_m3_semantic_matching():
    """
    Predictions with large line distance (> 10) can match under M3 if
    file matches, category matches, and Jaccard similarity >= 0.30.
    """
    engine = BipartiteMatchingEngine(MatchingProtocol.M3_SEMANTIC)

    p = make_dummy_pred(
        "p1", "app.py", 50,
        category=FindingCategory.SECURITY,
        title="SQL Injection Vulnerability",
        desc="User input in database query string allows SQL injection attacks."
    )
    g = make_dummy_gt(
        "g1", "app.py", 10,
        category=FindingCategory.SECURITY,
        title="SQL Injection Flaw",
        desc="User parameters directly concatenated into SQL query string."
    )

    # Under M1 and M2: fails due to line distance 40
    res_m1 = engine.match([p], [g], MatchingProtocol.M1_STRICT)
    assert res_m1.tp == 0

    res_m2 = engine.match([p], [g], MatchingProtocol.M2_TOLERANT)
    assert res_m2.tp == 0

    # Under M3: succeeds due to semantic and category overlap
    res_m3 = engine.match([p], [g], MatchingProtocol.M3_SEMANTIC)
    assert res_m3.tp == 1
    assert res_m3.matched_pairs[0].category_match is True
    assert res_m3.matched_pairs[0].jaccard_similarity >= 0.30


def test_hunk_span_granularity_matching():
    """
    CodeBERT E3 produces HUNK_SPAN predictions [20, 35].
    A GT issue at line 28 is inside the span (distance 0) -> matches M1.
    """
    engine = BipartiteMatchingEngine(MatchingProtocol.M1_STRICT)

    p_hunk = make_dummy_pred(
        "p_e3", "model.py", line_start=20, line_end=35,
        granularity=FindingGranularity.HUNK_SPAN
    )
    gt_inside = make_dummy_gt("gt1", "model.py", line_number=28)

    res = engine.match([p_hunk], [gt_inside], MatchingProtocol.M1_STRICT)
    assert res.tp == 1
    assert res.matched_pairs[0].line_distance == 0
