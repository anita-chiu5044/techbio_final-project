"""Integration tests for the YMCA agent pipeline (DB-level, no GPU)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ymca_agent.storage import connect
from ymca_agent.tools import AgentTools


def _make_yolo_detection(det_id: str, case_id: str, confidence: float = 0.95) -> dict:
    return {
        "detection_id": det_id,
        "case_id": case_id,
        "image_id": "image_001",
        "source_image_path": "/data/slide_001.tiff",
        "class_label": "WBC",
        "class_id": 0,
        "confidence": confidence,
        "bbox_xyxy_original": [100, 100, 200, 200],
        "image_width": 1024,
        "image_height": 1024,
    }


def _make_classifier_result(label: str, prob: float) -> dict:
    return {
        "image": "patch.png",
        "top1_class": label,
        "top1_prob": prob,
        "predictions": [
            {"rank": 1, "class": label, "probability": prob},
            {"rank": 2, "class": "LYT", "probability": 1.0 - prob},
        ],
    }


def _make_medsam_record(status: str = "OK") -> dict:
    return {
        "image": "patch.tiff",
        "category": "granulocyte_mature",
        "cell_type": "NGS",
        "status": status,
        "num_detections": "1",
        "fail_reason": "",
        "mask_path": "/data/mask.png",
    }


def test_multi_cell_e2e_pipeline(tmp_path):
    """Simulate 3 WBC detections from one image through full pipeline."""
    db_path = tmp_path / "test.db"
    tools = AgentTools(db_path)
    case_id = "case_e2e"

    # 1. Create case
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, status) VALUES (?,?,?,?)",
            (case_id, "user_e2e", "slide_001.tiff", "completed"),
        )

    # 2. Import 3 YOLO detections
    for i, det_id in enumerate(["det_000001", "det_000002", "det_000003"]):
        tools.import_yolo_detection(case_id, _make_yolo_detection(det_id, case_id))

    # 3. Apply MedSAM results
    for det_id in ["det_000001", "det_000002", "det_000003"]:
        tools.apply_medsam_result(det_id, _make_medsam_record("OK"))

    # 4. Apply classifier results: NGS(high), MYB(low), LYT(medium)
    tools.apply_classifier_result("det_000001", _make_classifier_result("NGS", 0.95), classifier_checkpoint="ckpt_v1")
    tools.apply_classifier_result("det_000002", _make_classifier_result("MYB", 0.55), classifier_checkpoint="ckpt_v1")
    tools.apply_classifier_result("det_000003", _make_classifier_result("LYT", 0.85), classifier_checkpoint="ckpt_v1")

    # 5. Verify summary
    summary = tools.summarize_case(case_id)
    assert summary["total_cells"] == 3
    # MYB should be in review queue (rare class + low confidence)
    assert summary["review_needed_count"] >= 1

    # 6. Accept NGS (high confidence, common class)
    tools.update_cell_review("det_000001", review_status="accepted_model_label")

    # 7. Correct MYB to PMO
    tools.update_cell_review("det_000002", review_label="PMO", review_status="corrected")

    # 8. Exclude det_000003
    tools.update_cell_review("det_000003", review_status="excluded")

    # 9. Verify final summary
    final_summary = tools.summarize_case(case_id)
    assert final_summary["hard_counts"].get("NGS", 0) == 1
    assert final_summary["hard_counts"].get("PMO", 0) == 1
    assert "LYT" not in final_summary["hard_counts"]  # excluded
    assert final_summary["excluded_count"] == 1
    assert final_summary["review_needed_count"] == 0

    # 10. Generate report
    report = tools.generate_case_report(case_id)
    assert report["safety"]["safe"] is True
    assert "NGS" in report["content"]

    # 11. Verify audit trail
    with connect(db_path) as conn:
        events = conn.execute(
            "SELECT * FROM review_events WHERE cell_id IN (?,?,?)",
            ("det_000001", "det_000002", "det_000003"),
        ).fetchall()
    assert len(events) == 3  # one event per review action


def test_cell_id_uniqueness_multi_detection(tmp_path):
    """Two WBC detections from same image must get distinct cell_ids."""
    db_path = tmp_path / "test.db"
    tools = AgentTools(db_path)
    case_id = "case_multi"

    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, status) VALUES (?,?,?,?)",
            (case_id, "user_multi", "slide_001.tiff", "completed"),
        )

    det1 = _make_yolo_detection("det_001", case_id, confidence=0.95)
    det2 = _make_yolo_detection("det_002", case_id, confidence=0.88)
    # Same image_id but different detection_ids
    assert det1["image_id"] == det2["image_id"]

    tools.import_yolo_detection(case_id, det1)
    tools.import_yolo_detection(case_id, det2)

    cells = tools.list_cells(case_id)
    cell_ids = [c["cell_id"] for c in cells]
    assert len(cell_ids) == 2
    assert cell_ids[0] != cell_ids[1]


def test_disease_warnings_in_summary(tmp_path):
    """Summary should include disease_warnings when blast ratio is high."""
    db_path = tmp_path / "test.db"
    tools = AgentTools(db_path)
    case_id = "case_blast"

    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, status) VALUES (?,?,?,?)",
            (case_id, "user_blast", "slide.tiff", "completed"),
        )

    # Import 5 cells: 3 MYB (blast-like) + 2 NGS
    for i in range(5):
        det_id = f"det_b{i:03d}"
        tools.import_yolo_detection(case_id, _make_yolo_detection(det_id, case_id))
        label = "MYB" if i < 3 else "NGS"
        prob = 0.90
        tools.apply_classifier_result(det_id, _make_classifier_result(label, prob), classifier_checkpoint="v1")

    # Accept all cells for hard counts (MYB is rare, so accept via corrected to itself)
    for i in range(5):
        det_id = f"det_b{i:03d}"
        label = "MYB" if i < 3 else "NGS"
        tools.update_cell_review(det_id, review_label=label, review_status="corrected")

    summary = tools.summarize_case(case_id)
    # 3/5 = 60% blast-like, should trigger AML warning
    assert len(summary["disease_warnings"]) > 0
    assert any("AML" in w for w in summary["disease_warnings"])
