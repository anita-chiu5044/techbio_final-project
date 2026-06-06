"""Shared test fixtures for YMCA agent tests."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ymca_agent.storage import connect
from ymca_agent.tools import AgentTools


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def tools(db_path):
    return AgentTools(db_path)


@pytest.fixture
def seeded_case(db_path):
    """A case with 3 cells: 1 high-confidence NGS, 1 ambiguous MYB, 1 rare PMB."""
    tools = AgentTools(db_path)
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, status) VALUES (?,?,?,?)",
            ("case_seed", "user_seed", "slide_001.tiff", "completed"),
        )
        cells = [
            ("cell_ngs", "NGS", 0.95, "LYT", 0.03, False),
            ("cell_myb", "MYB", 0.55, "PMO", 0.40, True),
            ("cell_pmb", "PMB", 0.80, "MYO", 0.10, True),
        ]
        for cell_id, label, top_prob, top2, top2_prob, rare in cells:
            conn.execute(
                """INSERT INTO cells (
                    cell_id, case_id, bbox_xyxy_original, yolo_confidence, overlap_score,
                    segmentation_status, segmentation_quality, model_label, top_probability,
                    top2_label, top2_probability, probability_margin, probabilities_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cell_id, "case_seed", json.dumps([0, 0, 50, 50]),
                    0.95, 0.0, "ok", 0.90,
                    label, top_prob, top2, top2_prob,
                    top_prob - top2_prob,
                    json.dumps({label: top_prob, top2: top2_prob}),
                ),
            )
    return tools


@pytest.fixture
def guidelines_dir(tmp_path):
    """Minimal reporting guidelines directory with test content."""
    gdir = tmp_path / "guidelines"
    gdir.mkdir()
    (gdir / "report_template.md").write_text("# Report\nMorphology review session.\n")
    (gdir / "allowed_phrases.md").write_text("- morphology-level screening\n- research draft\n")
    (gdir / "prohibited_claims.md").write_text("- confirmed AML\n- definitive diagnosis\n")
    (gdir / "cell_abbreviation_canonical_map.md").write_text(
        "| Canonical | Full term | Synonyms | Notes |\n"
        "| --- | --- | --- | --- |\n"
        "| NGS | Neutrophil segmented | Seg | mature |\n"
    )
    (gdir / "review_triggers.yaml").write_text("triggers:\n  - rare_class\n")
    (gdir / "critical_flags.yaml").write_text("flags:\n  - apl_suspect\n")
    (gdir / "qc_review_template.md").write_text("# QC\nCheck cells.\n")
    (gdir / "source_notes.md").write_text("Source: AML dataset\n")
    return gdir
