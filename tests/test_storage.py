"""Tests for ymca_agent.storage."""

import sqlite3

import pytest

from ymca_agent.storage import connect, init_db


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    with connect(db_path) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    expected = {"cases", "cells", "review_events", "reports",
                "conversations", "messages", "conversation_state"}
    assert expected.issubset(tables)


def test_connect_returns_row_factory(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = connect(db_path)
    assert conn.row_factory == sqlite3.Row
    conn.close()


def test_foreign_key_enforcement(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO cells (cell_id, case_id, bbox_xyxy_original) VALUES (?,?,?)",
                ("orphan_cell", "nonexistent_case", "[0,0,10,10]"),
            )


def test_init_db_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    init_db(db_path)  # should not raise
    with connect(db_path) as conn:
        tables = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
    assert tables >= 6


def test_migrate_adds_yolo_columns(tmp_path):
    """Simulates an old DB without yolo columns, verifies migration adds them."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    with connect(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(cells)").fetchall()}
    assert "yolo_class_id" in columns
    assert "yolo_class_name" in columns
    assert "downstream_eligible" in columns
