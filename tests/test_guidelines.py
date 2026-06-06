from pathlib import Path

from ymca_agent.guidelines import load_reporting_guidelines, validate_report_safety
from ymca_agent.storage import connect
from ymca_agent.tools import AgentTools


def _seed_case(db_path: Path, guidelines_dir: Path) -> AgentTools:
    tools = AgentTools(db_path, guidelines_dir=guidelines_dir)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("case_guideline", "user_test", "image.png", 128, 128, "completed"),
        )
        conn.execute(
            """
            INSERT INTO cells (
                cell_id, case_id, bbox_xyxy_original, yolo_confidence, overlap_score,
                segmentation_status, segmentation_quality, model_label, top_probability,
                top2_label, top2_probability, probability_margin, probabilities_json, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cell_review",
                "case_guideline",
                "[1, 2, 30, 40]",
                0.91,
                0.0,
                "ok",
                0.92,
                "Immature",
                0.52,
                "Lymphocyte",
                0.40,
                0.12,
                "{\"Immature\": 0.52, \"Lymphocyte\": 0.40, \"Others\": 0.08}",
                "queued_for_review",
            ),
        )
        conn.execute(
            """
            INSERT INTO cells (
                cell_id, case_id, bbox_xyxy_original, yolo_confidence, overlap_score,
                segmentation_status, segmentation_quality, model_label, top_probability,
                top2_label, top2_probability, probability_margin, probabilities_json, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cell_accepted",
                "case_guideline",
                "[40, 5, 70, 35]",
                0.97,
                0.0,
                "ok",
                0.95,
                "Lymphocyte",
                0.88,
                "Others",
                0.08,
                0.80,
                "{\"Lymphocyte\": 0.88, \"Others\": 0.08, \"Immature\": 0.04}",
                "accepted_model_label",
            ),
        )
    return tools


def test_load_reporting_guidelines(tmp_path):
    guidelines_dir = tmp_path / "reporting_guidelines"
    guidelines_dir.mkdir()
    (guidelines_dir / "allowed_phrases.md").write_text("- compatible with\n- recommend confirmatory review/testing\n")
    (guidelines_dir / "prohibited_claims.md").write_text("- confirmed aml\n- complete wbc differential\n")
    (guidelines_dir / "critical_flags.yaml").write_text("- possible apl requires urgent expert review\n")
    guidelines = load_reporting_guidelines(guidelines_dir)
    assert guidelines.allowed_phrases == ["compatible with", "recommend confirmatory review/testing"]
    assert guidelines.prohibited_claims == ["confirmed aml", "complete wbc differential"]
    assert guidelines.critical_flags == ["possible apl requires urgent expert review"]


def test_validate_report_safety_detects_prohibited_claims(tmp_path):
    guidelines_dir = tmp_path / "reporting_guidelines"
    guidelines_dir.mkdir()
    (guidelines_dir / "prohibited_claims.md").write_text("- confirmed aml\n")
    guidelines = load_reporting_guidelines(guidelines_dir)
    safety = validate_report_safety("This is confirmed AML.", guidelines)
    assert safety == {"safe": False, "violations": ["confirmed aml"]}


def test_generate_case_report_uses_morphology_boundary(tmp_path):
    guidelines_dir = tmp_path / "reporting_guidelines"
    guidelines_dir.mkdir()
    (guidelines_dir / "allowed_phrases.md").write_text("- morphology-level suggestion only\n")
    (guidelines_dir / "prohibited_claims.md").write_text("- confirmed aml\n- complete wbc differential\n")
    (guidelines_dir / "critical_flags.yaml").write_text("- possible apl requires urgent expert review\n")
    tools = _seed_case(tmp_path / "test.db", guidelines_dir)
    report = tools.generate_case_report("case_guideline")
    assert report["safety"]["safe"] is True
    assert "morphology-review session" in report["content"]
    assert "Boundary: this MVP does not report complete WBC differential" in report["content"]
    assert "Differential:" not in report["content"]
    assert "Blast-like ratio:" not in report["content"]


def test_validate_report_safety_allows_negated_boundary_language(tmp_path):
    guidelines_dir = tmp_path / "reporting_guidelines"
    guidelines_dir.mkdir()
    (guidelines_dir / "prohibited_claims.md").write_text("- complete wbc differential\n")
    guidelines = load_reporting_guidelines(guidelines_dir)
    safety = validate_report_safety("This MVP does not report complete WBC differential.", guidelines)
    assert safety == {"safe": True, "violations": []}


def test_negation_window_beyond_40_chars(tmp_path):
    """Negation >40 chars before a prohibited phrase must still be detected after window fix."""
    guidelines_dir = tmp_path / "reporting_guidelines"
    guidelines_dir.mkdir()
    (guidelines_dir / "prohibited_claims.md").write_text("- confirmed aml\n")
    guidelines = load_reporting_guidelines(guidelines_dir)
    # "not" is 62 chars before "confirmed aml" — old 40-char window would miss it
    report = "The system does not, based on morphology screening alone, diagnose confirmed aml in any patient."
    safety = validate_report_safety(report, guidelines)
    assert safety["safe"] is True, (
        "negation >40 chars away should still be detected: " + str(safety["violations"])
    )
