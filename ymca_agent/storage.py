"""SQLite storage for the local blood-smear agent MVP."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    user_id TEXT,
    original_image_path TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    status TEXT NOT NULL DEFAULT 'pipeline_pending',
    pipeline_version TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cells (
    cell_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    detection_id TEXT,
    bbox_xyxy_original TEXT NOT NULL,
    yolo_class_id INTEGER,
    yolo_class_name TEXT,
    downstream_eligible INTEGER NOT NULL DEFAULT 1,
    yolo_confidence REAL,
    overlap_score REAL DEFAULT 0,
    roi_image_path TEXT,
    roi_xyxy_original TEXT,
    mask_path TEXT,
    clean_patch_path TEXT,
    segmentation_status TEXT,
    segmentation_quality REAL,
    model_label TEXT,
    top_probability REAL,
    top2_label TEXT,
    top2_probability REAL,
    probability_margin REAL,
    entropy REAL,
    probabilities_json TEXT,
    classifier_checkpoint TEXT,
    label_map_version TEXT,
    preprocess_version TEXT,
    review_status TEXT NOT NULL DEFAULT 'unreviewed',
    review_label TEXT,
    review_note TEXT,
    reviewer_id TEXT,
    reviewed_at TEXT,
    is_current INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cells_case ON cells(case_id);
CREATE INDEX IF NOT EXISTS idx_cells_label ON cells(model_label);
CREATE INDEX IF NOT EXISTS idx_cells_review_status ON cells(review_status);

CREATE TABLE IF NOT EXISTS review_events (
    review_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id TEXT NOT NULL REFERENCES cells(cell_id),
    previous_review_status TEXT,
    previous_review_label TEXT,
    new_review_status TEXT NOT NULL,
    new_review_label TEXT,
    note TEXT,
    reviewer_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    case_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_state (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id),
    active_case_id TEXT,
    active_image_id TEXT,
    last_referenced_cell_id TEXT,
    last_report_id INTEGER,
    pending_action TEXT,
    state_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: str | Path = "ymca_agent.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path = "ymca_agent.db") -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        _migrate_existing_schema(conn)


def _migrate_existing_schema(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(cells)").fetchall()}
    migrations = {
        "yolo_class_id": "ALTER TABLE cells ADD COLUMN yolo_class_id INTEGER",
        "yolo_class_name": "ALTER TABLE cells ADD COLUMN yolo_class_name TEXT",
        "downstream_eligible": "ALTER TABLE cells ADD COLUMN downstream_eligible INTEGER NOT NULL DEFAULT 1",
    }
    for column, statement in migrations.items():
        if column not in columns:
            conn.execute(statement)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cells_yolo_class ON cells(yolo_class_name)")
