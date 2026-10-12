"""SQLite storage for submitted reports.

Works with a fresh database and with databases created by the two earlier
prototypes:
  * notebook 03 app: reports(id, report_text, language_mode NOT NULL, predicted_category, submitted_at)
  * notebook 01 app: reports(id, report_text, predicted_category, probability NOT NULL, submitted_at)
Missing columns are added in place; nothing is deleted.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_text TEXT NOT NULL,
    predicted_category TEXT NOT NULL,
    confidence REAL,
    top_predictions TEXT,
    detected_language TEXT,
    detection_method TEXT,
    language_mode TEXT,
    model_used TEXT,
    submitted_at TEXT NOT NULL
)
"""

NEW_COLUMNS = {
    "confidence": "REAL",
    "top_predictions": "TEXT",
    "detected_language": "TEXT",
    "detection_method": "TEXT",
    "language_mode": "TEXT",
    "model_used": "TEXT",
}


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path or config.DB_PATH))


def _columns(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute("PRAGMA table_info(reports)").fetchall()
    return {r[1]: {"notnull": bool(r[3]), "default": r[4]} for r in rows}


def init_db(db_path: Path | str | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(SCHEMA)
        existing = _columns(conn)
        for name, sql_type in NEW_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE reports ADD COLUMN {name} {sql_type}")
        existing = _columns(conn)
        # Legacy notebook-01 databases stored confidence as "probability"
        if "probability" in existing:
            conn.execute("UPDATE reports SET confidence = probability "
                         "WHERE confidence IS NULL AND probability IS NOT NULL")
        # Legacy notebook-03 rows: language_mode is the best info we have
        conn.execute("UPDATE reports SET detected_language = language_mode, detection_method = 'manual' "
                     "WHERE detected_language IS NULL AND language_mode IS NOT NULL")


def save_report(result, db_path: Path | str | None = None) -> int:
    """Store a ClassificationResult and return its new id."""
    values = {
        "report_text": result.text,
        "predicted_category": result.prediction.category,
        "confidence": round(float(result.prediction.confidence), 6),
        "top_predictions": json.dumps([[l, round(p, 6)] for l, p in result.prediction.top]),
        "detected_language": result.detected_language,
        "detection_method": result.detection_method,
        "language_mode": result.language_used,
        "model_used": result.model_used,
        "probability": round(float(result.prediction.confidence), 6),  # legacy column
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
    }
    with closing(_connect(db_path)) as conn, conn:
        columns = [c for c in _columns(conn) if c in values]
        placeholders = ", ".join("?" for _ in columns)
        cursor = conn.execute(
            f"INSERT INTO reports ({', '.join(columns)}) VALUES ({placeholders})",
            [values[c] for c in columns],
        )
        return int(cursor.lastrowid)


def fetch_reports(db_path: Path | str | None = None) -> pd.DataFrame:
    wanted = ["id", "report_text", "predicted_category", "confidence", "top_predictions",
              "detected_language", "detection_method", "language_mode", "model_used", "submitted_at"]
    with closing(_connect(db_path)) as conn:
        available = [c for c in wanted if c in _columns(conn)]
        frame = pd.read_sql_query(
            f"SELECT {', '.join(available)} FROM reports ORDER BY id DESC", conn)
    for c in wanted:
        if c not in frame.columns:
            frame[c] = None
    frame["category_name"] = frame["predicted_category"].map(config.readable_label)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    return frame[wanted + ["category_name"]]


def delete_report(report_id: int, db_path: Path | str | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM reports WHERE id = ?", (int(report_id),))
