"""Contracts for external YOLO and ConvNet module outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

YOLO_CLASSES = {"RBC", "WBC", "Platelets"}
WBC_CLASS_NAME = "WBC"
MEDSAM_STATUSES = {"OK", "FAIL", "SKIPPED", "NO_DETECTION"}


def _require(record: Mapping[str, Any], key: str) -> Any:
    if key not in record:
        raise ValueError(f"missing required key: {key}")
    return record[key]


def _as_float(value: Any, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _as_probability(value: Any, key: str) -> float:
    number = _as_float(value, key)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return number


def _as_bbox(value: Any, key: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{key} must be a 4-item sequence")
    bbox = [_as_float(v, key) for v in value]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{key} must have x2 > x1 and y2 > y1")
    return bbox


def validate_yolo_detection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record from export_yolo_detection_manifest.py."""
    detection_id = str(_require(record, "detection_id"))
    class_label = str(_require(record, "class_label"))
    if class_label not in YOLO_CLASSES:
        raise ValueError(f"unsupported YOLO class_label: {class_label}")
    bbox = _as_bbox(_require(record, "bbox_xyxy_original"), "bbox_xyxy_original")
    confidence = _as_probability(_require(record, "confidence"), "confidence")
    class_id = int(_require(record, "class_id"))
    image_width = int(_require(record, "image_width"))
    image_height = int(_require(record, "image_height"))
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    return {
        "detection_id": detection_id,
        "case_id": record.get("case_id"),
        "image_id": record.get("image_id"),
        "source_image_path": str(_require(record, "source_image_path")),
        "bbox_xyxy_original": bbox,
        "confidence": confidence,
        "class_id": class_id,
        "class_label": class_label,
        "image_width": image_width,
        "image_height": image_height,
        "detector_checkpoint": str(record.get("detector_checkpoint", "")),
        "patch_path": record.get("patch_path"),
        "downstream_eligible": class_label == WBC_CLASS_NAME,
    }


def _abs_path(p: str | None) -> str | None:
    """Resolve a path to absolute if it exists, else return as-is (or None)."""
    if not p:
        return None
    resolved = Path(p).resolve()
    return str(resolved)


def yolo_detection_to_cell_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    detection = validate_yolo_detection(record)
    return {
        "detection_id": detection["detection_id"],
        "bbox_xyxy_original": json.dumps(detection["bbox_xyxy_original"]),
        "yolo_class_id": detection["class_id"],
        "yolo_class_name": detection["class_label"],
        "downstream_eligible": 1 if detection["downstream_eligible"] else 0,
        "yolo_confidence": detection["confidence"],
        "clean_patch_path": _abs_path(detection["patch_path"]),
        "roi_image_path": _abs_path(detection["source_image_path"]),
    }


def validate_classifier_result(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one JSON result from classifier_inference.py."""
    image = str(_require(record, "image"))
    top1_class = str(_require(record, "top1_class"))
    top1_prob = _as_probability(_require(record, "top1_prob"), "top1_prob")
    predictions = _require(record, "predictions")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise ValueError("predictions must be a sequence")
    if not predictions:
        raise ValueError("predictions must not be empty")

    normalized_predictions = []
    for idx, pred in enumerate(predictions, start=1):
        if not isinstance(pred, Mapping):
            raise ValueError("each prediction must be a mapping")
        rank = int(_require(pred, "rank"))
        if rank != idx:
            raise ValueError("prediction ranks must be contiguous and start at 1")
        label = str(_require(pred, "class"))
        probability = _as_probability(_require(pred, "probability"), "probability")
        normalized_predictions.append({"rank": rank, "class": label, "probability": probability})

    first = normalized_predictions[0]
    if first["class"] != top1_class:
        raise ValueError("top1_class must match predictions[0].class")
    if abs(first["probability"] - top1_prob) > 1e-4:
        raise ValueError("top1_prob must match predictions[0].probability")

    return {
        "image": image,
        "top1_class": top1_class,
        "top1_prob": top1_prob,
        "predictions": normalized_predictions,
    }


def classifier_result_to_cell_fields(
    record: Mapping[str, Any],
    *,
    classifier_checkpoint: str,
    label_map_version: str = "classifier_flat16_v1",
    preprocess_version: str = "convnet_224_imagenet_v1",
) -> dict[str, Any]:
    result = validate_classifier_result(record)
    predictions = result["predictions"]
    top2 = predictions[1] if len(predictions) > 1 else None
    top2_probability = None if top2 is None else top2["probability"]
    probability_margin = None if top2_probability is None else round(result["top1_prob"] - top2_probability, 6)
    probabilities = {pred["class"]: pred["probability"] for pred in predictions}
    return {
        "clean_patch_path": _abs_path(result["image"]),
        "model_label": result["top1_class"],
        "top_probability": result["top1_prob"],
        "top2_label": None if top2 is None else top2["class"],
        "top2_probability": top2_probability,
        "probability_margin": probability_margin,
        "probabilities_json": json.dumps(probabilities, sort_keys=True),
        "classifier_checkpoint": classifier_checkpoint,
        "label_map_version": label_map_version,
        "preprocess_version": preprocess_version,
    }


def validate_medsam_summary_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one row from MedSAM3 tiff_wbc_inference.py inference_summary.csv."""
    image = str(_require(record, "image"))
    category = str(record.get("category", ""))
    cell_type = str(record.get("cell_type", ""))
    status = str(_require(record, "status")).upper()
    if status not in MEDSAM_STATUSES:
        raise ValueError(f"unsupported MedSAM status: {status}")
    num_detections = int(_require(record, "num_detections"))
    if num_detections < 0:
        raise ValueError("num_detections must be non-negative")
    mask_path = str(record.get("mask_path", ""))
    fail_reason = str(record.get("fail_reason", ""))
    if status == "OK" and not mask_path:
        raise ValueError("mask_path is required when status is OK")
    if status == "OK" and num_detections <= 0:
        raise ValueError("num_detections must be positive when status is OK")
    # For SKIPPED, keep mask_path if it points to an existing file (pre-existing mask).
    # For all other non-OK statuses, clear it.
    if status not in {"OK", "SKIPPED"}:
        mask_path = ""
    elif status == "SKIPPED" and mask_path and not Path(mask_path).exists():
        mask_path = ""
    return {
        "image": image,
        "category": category,
        "cell_type": cell_type,
        "status": status,
        "num_detections": num_detections,
        "fail_reason": fail_reason,
        "mask_path": mask_path,
    }


def medsam_summary_to_cell_fields(
    record: Mapping[str, Any],
    *,
    preprocess_version: str = "medsam3_lisc_wbc_v1",
) -> dict[str, Any]:
    result = validate_medsam_summary_record(record)
    status = result["status"]
    if status == "OK":
        segmentation_status = "ok"
        segmentation_quality = 1.0
    elif status == "SKIPPED":
        segmentation_status = "skipped_existing"
        segmentation_quality = None
    elif status == "NO_DETECTION":
        segmentation_status = "failed"
        segmentation_quality = 0.0
    else:
        segmentation_status = "failed"
        segmentation_quality = 0.0
    abs_mask = _abs_path(result["mask_path"]) if result["mask_path"] else None
    return {
        "mask_path": abs_mask,
        "clean_patch_path": abs_mask,
        "segmentation_status": segmentation_status,
        "segmentation_quality": segmentation_quality,
        "preprocess_version": preprocess_version,
        "medsam_status": status,
        "medsam_num_detections": result["num_detections"],
        "medsam_fail_reason": result["fail_reason"],
        "medsam_category": result["category"],
        "medsam_cell_type": result["cell_type"],
    }
