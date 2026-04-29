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


def get_connection() -> sqlite3.Connection:
    """Public helper to get a database connection."""
    return _connect()


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
                entropy_score  REAL DEFAULT 0.0,
                model_version   TEXT DEFAULT '1.0.0'
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id                  TEXT PRIMARY KEY,
                timestamp           TEXT NOT NULL,
                prediction_id       TEXT,
                predicted_freshness TEXT,
                correct_freshness   TEXT,
                notes               TEXT CHECK(length(notes) <= 500),
                user_flagged        INTEGER DEFAULT 0,
                created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            );

            CREATE TABLE IF NOT EXISTS fruit_types (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT UNIQUE NOT NULL,
                default_shelf_life REAL DEFAULT 7.0,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_pred_ts  ON predictions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pred_entropy ON predictions(entropy_score DESC);
            CREATE INDEX IF NOT EXISTS idx_feed_pid ON feedback(prediction_id);
            CREATE INDEX IF NOT EXISTS idx_feed_flagged ON feedback(user_flagged);
        """)

        # Add entropy_score column if missing (for existing databases)
        try:
            conn.execute(
                "ALTER TABLE predictions ADD COLUMN entropy_score REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Seed default fruit types if empty
        cursor = conn.execute("SELECT COUNT(*) FROM fruit_types").fetchone()[0]
        if cursor == 0:
            fruit_types = [
                ("apple", 7.0),
                ("banana", 5.0),
                ("orange", 10.0),
                ("mango", 5.0),
                ("strawberry", 3.0),
                ("grape", 7.0),
                ("tomato", 7.0),
                ("pineapple", 7.0),
                ("watermelon", 7.0),
                ("default", 7.0),
            ]
            conn.executemany(
                "INSERT INTO fruit_types (name, default_shelf_life) VALUES (?, ?)",
                fruit_types,
            )


def log_prediction(
    freshness: str,
    freshness_conf: float,
    quality: str,
    shelf_life_days: float,
    inference_ms: float,
    entropy_score: float = 0.0,
) -> str:
    """Insert a prediction row and return its UUID."""
    pred_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, timestamp, freshness, freshness_conf, quality, shelf_life_days, inference_ms, entropy_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                datetime.utcnow().isoformat(),
                freshness,
                freshness_conf,
                quality,
                shelf_life_days,
                inference_ms,
                entropy_score,
            ),
        )
    return pred_id


def log_feedback(
    prediction_id: str,
    predicted_freshness: str,
    correct_freshness: str,
    notes: str,
) -> str:
    """Insert user feedback with proper transaction handling."""
    feed_id = str(uuid.uuid4())
    conn = _connect()

    try:
        # Check if prediction exists
        existing = conn.execute(
            "SELECT id FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"prediction_id '{prediction_id}' does not exist")

        # Insert feedback
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
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Feedback insert failed: {e}") from e

    return feed_id


def get_recent_predictions(
    limit: int = 20, offset: int = 0, freshness_filter: str = None
) -> list[dict]:
    """Get recent predictions with pagination and optional freshness filter."""
    with _connect() as conn:
        if freshness_filter:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE freshness = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (freshness_filter, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        avg_conf = (
            conn.execute("SELECT AVG(freshness_conf) FROM predictions").fetchone()[0]
            or 0.0
        )
        dist = conn.execute(
            "SELECT freshness, COUNT(*) as cnt FROM predictions GROUP BY freshness"
        ).fetchall()
    return {
        "total_predictions": total,
        "avg_confidence": round(avg_conf, 4),
        "freshness_distribution": {r["freshness"]: r["cnt"] for r in dist},
    }


def get_uncertain_predictions(
    limit: int = 100, entropy_threshold: float = 1.5
) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM predictions 
               WHERE entropy_score > ? 
               ORDER BY entropy_score DESC 
               LIMIT ?""",
            (entropy_threshold, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_corrected_predictions(limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT p.*, f.correct_freshness, f.notes, f.created_at as feedback_time
               FROM predictions p
               JOIN feedback f ON p.id = f.prediction_id
               WHERE f.correct_freshness != p.freshness
               ORDER BY f.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_fruit_types() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM fruit_types").fetchall()
    return [dict(r) for r in rows]


def get_fruit_type_id(fruit_name: str) -> int:
    """Get fruit type ID by name, returns default if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM fruit_types WHERE name = ?",
            (fruit_name.lower(),),
        ).fetchone()
        if row:
            return int(row["id"])
        default = conn.execute(
            "SELECT id FROM fruit_types WHERE name = 'default'"
        ).fetchone()
        return int(default["id"]) if default else 0
