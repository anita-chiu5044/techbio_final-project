import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ymca_agent.model_contracts import (
    classifier_result_to_cell_fields,
    medsam_summary_to_cell_fields,
    validate_classifier_result,
    validate_medsam_summary_record,
    validate_yolo_detection,
    yolo_detection_to_cell_fields,
)
from ymca_agent.storage import connect
from ymca_agent.tools import AgentTools


def yolo_record(class_label="WBC"):
    return {
        "detection_id": "det_000001",
        "case_id": "case_test",
        "image_id": "image_001",
        "source_image_path": "image.png",
        "source_image_relative_path": "case/image.png",
        "bbox_xyxy_original": [10.0, 12.0, 40.0, 48.0],
        "bbox_xyxy_clipped": [10, 12, 40, 48],
        "confidence": 0.91,
        "class_id": 1,
        "class_label": class_label,
        "image_width": 100,
        "image_height": 100,
        "detector_checkpoint": "techbio_final-project/best.pt",
        "patch_path": "patches/case/image_det_000001.png",
        "box_index_in_image": 1,
    }


def classifier_record():
    return {
        "image": "patches/case/image_det_000001.png",
        "top1_class": "apl_suspect",
        "top1_prob": 0.863,
        "predictions": [
            {"rank": 1, "class": "apl_suspect", "probability": 0.863},
            {"rank": 2, "class": "other_immature", "probability": 0.0695},
            {"rank": 3, "class": "myelocyte", "probability": 0.0526},
        ],
    }


def medsam_record(status="OK", num_detections=1, mask_path="out/granulocyte/MYB/MYB_0001_mask.png"):
    return {
        "image": "MYB_0001.tiff",
        "category": "granulocyte_immature",
        "cell_type": "MYB",
        "status": status,
        "num_detections": num_detections,
        "fail_reason": "" if status == "OK" else "no mask",
        "mask_path": mask_path,
    }


def test_validate_yolo_detection_accepts_manifest_record():
    result = validate_yolo_detection(yolo_record())
    assert result["class_label"] == "WBC"
    assert result["downstream_eligible"] is True


def test_yolo_non_wbc_is_not_downstream_eligible():
    fields = yolo_detection_to_cell_fields(yolo_record(class_label="RBC"))
    assert fields["yolo_class_name"] == "RBC"
    assert fields["downstream_eligible"] == 0


def test_validate_yolo_detection_rejects_unknown_class():
    bad = yolo_record(class_label="Blast")
    try:
        validate_yolo_detection(bad)
    except ValueError as exc:
        assert "unsupported YOLO class_label" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_classifier_result_accepts_inference_json():
    result = validate_classifier_result(classifier_record())
    assert result["top1_class"] == "apl_suspect"
    assert result["predictions"][1]["class"] == "other_immature"


def test_classifier_result_to_cell_fields_computes_margin():
    fields = classifier_result_to_cell_fields(
        classifier_record(),
        classifier_checkpoint="artifacts/checkpoints/convnet/best_flat_convnext.pth",
    )
    probabilities = json.loads(fields["probabilities_json"])
    assert fields["model_label"] == "apl_suspect"
    assert fields["top2_label"] == "other_immature"
    assert fields["probability_margin"] == 0.7935
    assert probabilities["apl_suspect"] == 0.863


def test_agent_tools_import_yolo_and_apply_classifier_result(tmp_path):
    db_path = tmp_path / "test.db"
    tools = AgentTools(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("case_test", "user_test", "image.png", 100, 100, "completed"),
        )

    cell = tools.import_yolo_detection("case_test", yolo_record())
    assert cell["cell_id"] == "det_000001"
    assert cell["yolo_class_name"] == "WBC"
    assert cell["downstream_eligible"] == 1

    update = tools.apply_classifier_result(
        "det_000001",
        classifier_record(),
        classifier_checkpoint="artifacts/checkpoints/convnet/best_flat_convnext.pth",
    )
    after = update["after"]
    assert after["model_label"] == "apl_suspect"
    assert after["top_probability"] == 0.863
    assert after["top2_label"] == "other_immature"


def test_validate_medsam_summary_record_accepts_ok_row():
    result = validate_medsam_summary_record(medsam_record())
    assert result["status"] == "OK"
    assert result["num_detections"] == 1


def test_medsam_summary_to_cell_fields_marks_no_detection_failed():
    fields = medsam_summary_to_cell_fields(medsam_record(status="NO_DETECTION", num_detections=0, mask_path=""))
    assert fields["segmentation_status"] == "failed"
    assert fields["segmentation_quality"] == 0.0
    assert fields["clean_patch_path"] is None


def test_agent_tools_apply_medsam_result(tmp_path):
    db_path = tmp_path / "test.db"
    tools = AgentTools(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (case_id, user_id, original_image_path, image_width, image_height, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("case_medsam", "user_test", "image.png", 100, 100, "completed"),
        )
    tools.import_yolo_detection("case_medsam", yolo_record())
    update = tools.apply_medsam_result("det_000001", medsam_record())
    after = update["after"]
    assert after["segmentation_status"] == "ok"
    assert after["segmentation_quality"] == 1.0
    assert after["mask_path"] == "out/granulocyte/MYB/MYB_0001_mask.png"
