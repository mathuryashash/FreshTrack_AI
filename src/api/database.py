"""
SQLite-backed prediction store.
Uses SQLite for zero-config local/dev use.
Swap DATABASE_URL to postgresql://... for production.
"""
import os
import uuid
import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path

DATABASE_URL = os.environ.get("DATABASE_URL", "data/freshtrack.db")

if not DATABASE_URL.endswith(".db"):
    raise ValueError(f"DATABASE_URL must point to a .db file, got: {DATABASE_URL}")


_local = threading.local()


def _connect() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        Path(DATABASE_URL).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DATABASE_URL, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.conn = conn
    return _local.conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                freshness       TEXT,
                freshness_conf  REAL,
                quality         TEXT,
                shelf_life_days REAL,
                inference_ms    REAL,
                model_version   TEXT DEFAULT '1.0.0'
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id                  TEXT PRIMARY KEY,
                timestamp           TEXT NOT NULL,
                prediction_id       TEXT,
                predicted_freshness TEXT,
                correct_freshness   TEXT,
                notes               TEXT CHECK(length(notes) <= 500),
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_pred_ts  ON predictions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_feed_pid ON feedback(prediction_id);
        """)


def log_prediction(
    freshness: str,
    freshness_conf: float,
    quality: str,
    shelf_life_days: float,
    inference_ms: float,
) -> str:
    """Insert a prediction row and return its UUID."""
    pred_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, timestamp, freshness, freshness_conf, quality, shelf_life_days, inference_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                datetime.utcnow().isoformat(),
                freshness,
                freshness_conf,
                quality,
                shelf_life_days,
                inference_ms,
            ),
        )
    return pred_id


def log_feedback(
    prediction_id: str,
    predicted_freshness: str,
    correct_freshness: str,
    notes: str,
) -> str:
    feed_id = str(uuid.uuid4())
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"prediction_id '{prediction_id}' does not exist")
        conn.execute(
            """INSERT INTO feedback
               (id, timestamp, prediction_id, predicted_freshness, correct_freshness, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                feed_id,
                datetime.utcnow().isoformat(),
                prediction_id,
                predicted_freshness,
                correct_freshness,
                notes,
            ),
        )
    return feed_id


def get_recent_predictions(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(freshness_conf) FROM predictions"
        ).fetchone()[0] or 0.0
        dist = conn.execute(
            "SELECT freshness, COUNT(*) as cnt FROM predictions GROUP BY freshness"
        ).fetchall()
    return {
        "total_predictions": total,
        "avg_confidence": round(avg_conf, 4),
        "freshness_distribution": {r["freshness"]: r["cnt"] for r in dist},
    }
