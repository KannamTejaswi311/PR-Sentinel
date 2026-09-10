import sqlite3
import json
from datetime import datetime
from pathlib import Path


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
# INITIALIZE DATABASE
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

            findings TEXT
        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# SAVE REVIEW
# ============================================================

def save_review(
    filename,
    summary,
    review_text
):

    severity = summary.get(
        "severity",
        {}
    )

    findings = summary.get(
        "findings",
        []
    )

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
            findings
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            summary.get(
                "critical_issues",
                0
            ),

            summary.get(
                "security_issues",
                0
            ),

            summary.get(
                "code_quality_issues",
                0
            ),

            summary.get(
                "test_issues",
                0
            ),

            severity.get(
                "critical",
                0
            ),

            severity.get(
                "high",
                0
            ),

            severity.get(
                "medium",
                0
            ),

            severity.get(
                "low",
                0
            ),

            review_text,

            json.dumps(
                findings
            ),
        )
    )

    connection.commit()

    review_id = cursor.lastrowid

    connection.close()

    return review_id


# ============================================================
# GET ALL REVIEWS
# ============================================================

def get_all_reviews():

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
            low_severity
        FROM review_history
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# GET SINGLE REVIEW
# ============================================================

def get_review(review_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM review_history
        WHERE id = ?
        """,
        (review_id,)
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    review = dict(row)

    try:
        review["findings"] = json.loads(
            review["findings"] or "[]"
        )
    except json.JSONDecodeError:
        review["findings"] = []

    return review


# ============================================================
# DELETE REVIEW
# ============================================================

def delete_review(review_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM review_history
        WHERE id = ?
        """,
        (review_id,)
    )

    deleted = cursor.rowcount > 0

    connection.commit()

    connection.close()

    return deleted