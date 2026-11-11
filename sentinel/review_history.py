"""
PR-Sentinel: Review History and Developer Feedback Persistence Layer
Manages SQLite database for historical code reviews, risk metrics,
synthesized reports, and developer feedback tracking.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# DATABASE LOCATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "review_history.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ============================================================
# INITIALIZE DATABASE & RUN MIGRATIONS
# ============================================================

def init_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS review_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            review_date TEXT NOT NULL,
            critical_issues INTEGER DEFAULT 0,
            security_issues INTEGER DEFAULT 0,
            code_quality_issues INTEGER DEFAULT 0,
            test_issues INTEGER DEFAULT 0,
            critical_severity INTEGER DEFAULT 0,
            high_severity INTEGER DEFAULT 0,
            medium_severity INTEGER DEFAULT 0,
            low_severity INTEGER DEFAULT 0,
            review_text TEXT,
            findings TEXT,
            risk_score INTEGER DEFAULT 0,
            risk_level TEXT DEFAULT 'LOW RISK',
            synthesized_json TEXT DEFAULT '{}'
        )
        """
    )

    # Migrations: add columns if table was created in an earlier stage
    cursor.execute("PRAGMA table_info(review_history)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    if "risk_score" not in existing_cols:
        cursor.execute("ALTER TABLE review_history ADD COLUMN risk_score INTEGER DEFAULT 0")
    if "risk_level" not in existing_cols:
        cursor.execute("ALTER TABLE review_history ADD COLUMN risk_level TEXT DEFAULT 'LOW RISK'")
    if "synthesized_json" not in existing_cols:
        cursor.execute("ALTER TABLE review_history ADD COLUMN synthesized_json TEXT DEFAULT '{}'")

    # Developer feedback table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS developer_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER,
            finding_index INTEGER,
            feedback_type TEXT NOT NULL,
            comment TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(review_id) REFERENCES review_history(id)
        )
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# SAVE REVIEW
# ============================================================

def save_review(
    filename: str,
    summary: Dict[str, Any],
    review_text: str,
    risk_score: int = 0,
    risk_level: str = "LOW RISK",
    synthesized_report: Optional[Dict[str, Any]] = None,
) -> int:
    severity = summary.get("severity", {})
    findings = summary.get("findings", [])

    if synthesized_report and not risk_score:
        risk_score = synthesized_report.get("risk_assessment", {}).get("score", 0)
        risk_level = synthesized_report.get("risk_assessment", {}).get("risk_level", "LOW RISK")

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO review_history (
            filename,
            review_date,
            critical_issues,
            security_issues,
            code_quality_issues,
            test_issues,
            critical_severity,
            high_severity,
            medium_severity,
            low_severity,
            review_text,
            findings,
            risk_score,
            risk_level,
            synthesized_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary.get("critical_issues", 0),
            summary.get("security_issues", 0),
            summary.get("code_quality_issues", 0),
            summary.get("test_issues", 0),
            severity.get("critical", 0),
            severity.get("high", 0),
            severity.get("medium", 0),
            severity.get("low", 0),
            review_text,
            json.dumps(findings),
            risk_score,
            risk_level,
            json.dumps(synthesized_report or {}),
        ),
    )

    connection.commit()
    review_id = cursor.lastrowid
    connection.close()
    return review_id


# ============================================================
# SAVE & GET DEVELOPER FEEDBACK
# ============================================================

def save_feedback(
    review_id: int,
    finding_index: int,
    feedback_type: str,
    comment: str = ""
) -> int:
    """Save developer feedback (useful, not_useful, false_positive, correct_finding)."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO developer_feedback (
            review_id,
            finding_index,
            feedback_type,
            comment,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            review_id,
            finding_index,
            feedback_type,
            comment,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )

    connection.commit()
    fb_id = cursor.lastrowid
    connection.close()
    return fb_id


def get_feedback_for_review(review_id: int) -> List[Dict[str, Any]]:
    """Retrieve all developer feedback records for a given review."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, review_id, finding_index, feedback_type, comment, created_at
        FROM developer_feedback
        WHERE review_id = ?
        ORDER BY created_at DESC
        """,
        (review_id,),
    )
    rows = cursor.fetchall()
    connection.close()
    return [dict(r) for r in rows]


# ============================================================
# GET ALL REVIEWS
# ============================================================

def get_all_reviews() -> List[Dict[str, Any]]:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            review_date,
            critical_issues,
            security_issues,
            code_quality_issues,
            test_issues,
            critical_severity,
            high_severity,
            medium_severity,
            low_severity,
            risk_score,
            risk_level
        FROM review_history
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    connection.close()

    reviews = []
    for row in rows:
        r = dict(row)
        reviews.append({
            "id": r["id"],
            "filename": r["filename"],
            "review_date": r["review_date"],
            "critical_issues": r.get("critical_issues", 0),
            "security_issues": r.get("security_issues", 0),
            "code_quality_issues": r.get("code_quality_issues", 0),
            "test_issues": r.get("test_issues", 0),
            "critical_severity": r.get("critical_severity", 0),
            "high_severity": r.get("high_severity", 0),
            "medium_severity": r.get("medium_severity", 0),
            "low_severity": r.get("low_severity", 0),
            "risk_score": r.get("risk_score", 0),
            "risk_level": r.get("risk_level", "LOW RISK"),
        })

    return reviews


# ============================================================
# GET REVIEW DETAIL
# ============================================================

def get_review(review_id: int) -> Optional[Dict[str, Any]]:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM review_history
        WHERE id = ?
        """,
        (review_id,),
    )

    row = cursor.fetchone()
    connection.close()

    if row is None:
        return None

    r = dict(row)
    findings = []
    if r.get("findings"):
        try:
            findings = json.loads(r["findings"])
        except Exception:
            findings = []

    synthesized_data = {}
    if r.get("synthesized_json"):
        try:
            synthesized_data = json.loads(r["synthesized_json"])
        except Exception:
            synthesized_data = {}

    feedback = get_feedback_for_review(review_id)

    return {
        "id": r["id"],
        "filename": r["filename"],
        "review_date": r["review_date"],
        "critical_issues": r.get("critical_issues", 0),
        "security_issues": r.get("security_issues", 0),
        "code_quality_issues": r.get("code_quality_issues", 0),
        "test_issues": r.get("test_issues", 0),
        "critical_severity": r.get("critical_severity", 0),
        "high_severity": r.get("high_severity", 0),
        "medium_severity": r.get("medium_severity", 0),
        "low_severity": r.get("low_severity", 0),
        "risk_score": r.get("risk_score", 0),
        "risk_level": r.get("risk_level", "LOW RISK"),
        "review_text": r.get("review_text", ""),
        "findings": findings,
        "synthesized_data": synthesized_data,
        "developer_feedback": feedback,
    }


def delete_review(review_id: int) -> bool:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM developer_feedback WHERE review_id = ?", (review_id,))
    cursor.execute("DELETE FROM review_history WHERE id = ?", (review_id,))
    deleted = cursor.rowcount > 0
    connection.commit()
    connection.close()
    return deleted
