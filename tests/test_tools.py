import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ymca_agent.storage import connect
from ymca_agent.tools import AgentTools


def seed_one_case(db_path: Path) -> AgentTools:
    tools = AgentTools(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("case_test", "user_test", "image.png", 100, 100, "completed"),
        )
        conn.execute(
            """
            INSERT INTO cells (
                cell_id, case_id, bbox_xyxy_original, yolo_confidence, overlap_score,
                segmentation_status, segmentation_quality, model_label, top_probability,
                top2_label, top2_probability, probability_margin, probabilities_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cell_ambiguous",
                "case_test",
                json.dumps([10, 10, 40, 40]),
                0.9,
                0.0,
                "ok",
                0.9,
                "Immature",
                0.46,
                "Lymphocyte",
                0.43,
                0.03,
                json.dumps({"Immature": 0.46, "Lymphocyte": 0.43, "Others": 0.11}),
            ),
        )
    return tools


def test_corrected_cell_leaves_uncertain_queue(tmp_path):
    tools = seed_one_case(tmp_path / "test.db")
    assert [c["cell_id"] for c in tools.list_uncertain_cells("case_test", label="Immature")] == ["cell_ambiguous"]
    tools.update_cell_review("cell_ambiguous", review_label="LYT", review_status="corrected")
    assert tools.list_uncertain_cells("case_test", label="LYT") == []


def test_accepted_model_label_counts_despite_qc_reason(tmp_path):
    tools = seed_one_case(tmp_path / "test.db")
    assert tools.summarize_case("case_test")["review_needed_count"] == 1
    tools.update_cell_review("cell_ambiguous", review_status="accepted_model_label")
    summary = tools.summarize_case("case_test")
    assert summary["review_needed_count"] == 0
    assert summary["hard_counts"] == {"Immature": 1}



def test_conversation_cannot_bind_other_users_case(tmp_path):
    db_path = tmp_path / "test.db"
    tools = AgentTools(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("case_user_b", "user_b", "image_b.png", 100, 100, "completed"),
        )
    tools.start_conversation("conv_user_a", user_id="user_a")
    try:
        tools.set_active_case("conv_user_a", "case_user_b", user_id="user_a")
    except PermissionError as exc:
        assert "does not belong" in str(exc)
    else:
        raise AssertionError("expected PermissionError")


def test_conversation_user_mismatch_rejected(tmp_path):
    tools = AgentTools(tmp_path / "test.db")
    tools.start_conversation("conv", user_id="user_a")
    try:
        tools.start_conversation("conv", user_id="user_b")
    except PermissionError as exc:
        assert "does not belong" in str(exc)
    else:
        raise AssertionError("expected PermissionError")


def _seed_case_with_cell(db_path: Path, cell_id: str, model_label: str, top_probability: float, downstream_eligible: int = 1) -> AgentTools:
    tools = AgentTools(db_path)
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status) VALUES (?, ?, ?, ?, ?, ?)",
            ("case_x", "user_x", "img.png", 100, 100, "completed"),
        )
        conn.execute(
            """
            INSERT INTO cells (
                cell_id, case_id, bbox_xyxy_original, yolo_confidence, overlap_score,
                segmentation_status, segmentation_quality, model_label, top_probability,
                top2_label, top2_probability, probability_margin, probabilities_json,
                downstream_eligible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cell_id, "case_x",
                json.dumps([0, 0, 20, 20]),
                0.95, 0.0, "ok", 0.95,
                model_label, top_probability,
                "Lymphocyte", top_probability - 0.01,
                0.01,
                json.dumps({model_label: top_probability, "Lymphocyte": top_probability - 0.01}),
                downstream_eligible,
            ),
        )
    return tools


def test_rare_class_always_queued(tmp_path):
    """A cell with an apl_suspect label and high confidence must still be queued for review."""
    tools = _seed_case_with_cell(tmp_path / "test.db", "cell_apl", "apl_suspect", 0.99)
    summary = tools.summarize_case("case_x")
    assert summary["review_needed_count"] > 0, "rare class cell should be in review queue"
    assert "apl_suspect" not in summary["hard_counts"], "rare class cell should not be in hard counts"


def test_apply_classifier_twice_raises(tmp_path):
    """Calling apply_classifier_result twice on the same cell must raise RuntimeError."""
    tools = AgentTools(tmp_path / "test.db")
    with connect(tmp_path / "test.db") as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status) VALUES (?,?,?,?,?,?)",
            ("case_cls", "u", "img.png", 100, 100, "completed"),
        )
        conn.execute(
            "INSERT INTO cells (cell_id, case_id, bbox_xyxy_original) VALUES (?,?,?)",
            ("cell_cls", "case_cls", json.dumps([0, 0, 10, 10])),
        )
    classifier_result = {
        "image": "img.png",
        "top1_class": "Lymphocyte",
        "top1_prob": 0.90,
        "predictions": [
            {"rank": 1, "class": "Lymphocyte", "probability": 0.90},
            {"rank": 2, "class": "Others", "probability": 0.10},
        ],
    }
    tools.apply_classifier_result("cell_cls", classifier_result, classifier_checkpoint="ckpt_v1")
    try:
        tools.apply_classifier_result("cell_cls", classifier_result, classifier_checkpoint="ckpt_v1")
    except RuntimeError as exc:
        assert "cannot overwrite" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on second apply_classifier_result")


def test_update_cell_review_accepted_without_model_label_raises(tmp_path):
    """Accepting model label when model_label is None must raise ValueError."""
    tools = AgentTools(tmp_path / "test.db")
    with connect(tmp_path / "test.db") as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status) VALUES (?,?,?,?,?,?)",
            ("case_nolabel", "u", "img.png", 100, 100, "completed"),
        )
        conn.execute(
            "INSERT INTO cells (cell_id, case_id, bbox_xyxy_original) VALUES (?,?,?)",
            ("cell_nolabel", "case_nolabel", json.dumps([0, 0, 10, 10])),
        )
    try:
        tools.update_cell_review("cell_nolabel", review_status="accepted_model_label")
    except ValueError as exc:
        assert "model_label is not set" in str(exc)
    else:
        raise AssertionError("expected ValueError when accepting null model_label")


def test_blast_like_ratio_includes_apl_suspect(tmp_path):
    """blast_like_ratio must include apl_suspect cells after review acceptance."""
    tools = _seed_case_with_cell(tmp_path / "test.db", "cell_blast", "apl_suspect", 0.99)
    # Must override rare_class blocking: use corrected review to accept with explicit label
    tools.update_cell_review("cell_blast", review_label="apl_suspect", review_status="corrected")
    summary = tools.summarize_case("case_x")
    assert summary["blast_like_ratio"] is not None and summary["blast_like_ratio"] > 0, (
        "apl_suspect should contribute to blast_like_ratio"
    )


def test_classifier_not_run_queues_cell(tmp_path):
    """A downstream-eligible cell without classifier output must have classifier_not_run review reason."""
    from ymca_agent.qc import review_reasons
    reasons = review_reasons(
        yolo_confidence=0.95,
        segmentation_quality=0.95,
        segmentation_status="ok",
        top_probability=None,
        top2_probability=None,
        probabilities=None,
        overlap_score=0.0,
        downstream_eligible=True,
    )
    assert "classifier_not_run" in reasons


def test_deactivate_cell(tmp_path):
    tools = _seed_case_with_cell(tmp_path / "test.db", "cell_deact", "LYT", 0.95)
    result = tools.deactivate_cell("cell_deact", note="test deactivation")
    assert result["is_current"] == 0
    try:
        tools.get_cell("cell_deact")
    except KeyError:
        pass  # expected: is_current=0 means not found
    else:
        raise AssertionError("deactivated cell should not be returned by get_cell")


def test_deactivate_cell_not_found(tmp_path):
    tools = AgentTools(tmp_path / "test.db")
    try:
        tools.deactivate_cell("nonexistent")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for nonexistent cell")


def test_list_cells_filter_by_label(tmp_path):
    tools = seed_one_case(tmp_path / "test.db")
    cells = tools.list_cells("case_test", label="Immature")
    assert len(cells) == 1
    assert cells[0]["model_label"] == "Immature"


def test_list_cells_filter_by_review_status(tmp_path):
    tools = seed_one_case(tmp_path / "test.db")
    tools.update_cell_review("cell_ambiguous", review_status="accepted_model_label")
    cells = tools.list_cells("case_test", review_status="accepted_model_label")
    assert len(cells) == 1
    assert cells[0]["cell_id"] == "cell_ambiguous"


def test_record_message_valid_roles(tmp_path):
    tools = AgentTools(tmp_path / "test.db")
    tools.start_conversation("conv_msg")
    for role in ["user", "agent", "tool", "system"]:
        result = tools.record_message("conv_msg", role, f"test {role}")
        assert result["role"] == role
        assert result["message_id"] is not None


def test_record_message_invalid_role(tmp_path):
    tools = AgentTools(tmp_path / "test.db")
    tools.start_conversation("conv_msg2")
    try:
        tools.record_message("conv_msg2", "admin", "bad role")
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid role")


def test_summarize_case_empty(tmp_path):
    tools = AgentTools(tmp_path / "test.db")
    with connect(tmp_path / "test.db") as conn:
        conn.execute(
            "INSERT INTO cases (case_id, user_id, original_image_path, status) VALUES (?,?,?,?)",
            ("case_empty", "u", "img.png", "completed"),
        )
    summary = tools.summarize_case("case_empty")
    assert summary["total_cells"] == 0
    assert summary["hard_counts"] == {}
    assert summary["review_needed_count"] == 0
    assert summary["disease_warnings"] == []


def test_summarize_case_has_disease_warnings(tmp_path):
    """Summary includes disease_warnings field."""
    tools = seed_one_case(tmp_path / "test.db")
    tools.update_cell_review("cell_ambiguous", review_status="accepted_model_label")
    summary = tools.summarize_case("case_test")
    assert "disease_warnings" in summary
    assert isinstance(summary["disease_warnings"], list)


def test_update_cell_review_invalid_label(tmp_path):
    """Correcting to an unknown label must raise ValueError."""
    tools = seed_one_case(tmp_path / "test.db")
    try:
        tools.update_cell_review("cell_ambiguous", review_label="INVALID_LABEL", review_status="corrected")
    except ValueError as exc:
        assert "unknown review_label" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown review_label")


def test_get_cell_not_found(tmp_path):
    tools = AgentTools(tmp_path / "test.db")
    try:
        tools.get_cell("nonexistent_cell")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
