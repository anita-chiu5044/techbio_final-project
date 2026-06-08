"""Quality-control helpers for cell-level uncertainty and review routing."""

from __future__ import annotations

import math
from collections.abc import Mapping

RARE_CLASSES = frozenset({
    "monoblast", "myeloblast", "apl_suspect", "other_immature",
    "early_pre_b", "pre_b", "pro_b", "hematogone",
})

BLAST_LIKE_LABELS = frozenset({
    "monoblast", "myeloblast", "apl_suspect", "other_immature",
})


def probability_margin(top_probability: float | None, top2_probability: float | None) -> float | None:
    if top_probability is None or top2_probability is None:
        return None
    return max(0.0, float(top_probability) - float(top2_probability))


def entropy(probabilities: Mapping[str, float] | None) -> float | None:
    if not probabilities:
        return None
    total = sum(max(0.0, float(v)) for v in probabilities.values())
    if total <= 0:
        return None
    value = 0.0
    for raw in probabilities.values():
        p = max(0.0, float(raw)) / total
        if p > 0:
            value -= p * math.log(p)
    return value


def normalized_entropy(probabilities: Mapping[str, float] | None) -> float:
    value = entropy(probabilities)
    if value is None or not probabilities or len(probabilities) <= 1:
        return 0.0
    return min(1.0, value / math.log(len(probabilities)))


def review_reasons(
    *,
    yolo_confidence: float | None,
    segmentation_quality: float | None,
    segmentation_status: str | None,
    top_probability: float | None,
    top2_probability: float | None,
    probabilities: Mapping[str, float] | None,
    overlap_score: float | None,
    rare_class: bool = False,
    downstream_eligible: bool = True,
    yolo_low_threshold: float = 0.50,
    segmentation_low_threshold: float = 0.65,
    top_probability_threshold: float = 0.70,
    margin_threshold: float = 0.15,
    overlap_threshold: float = 0.50,
    entropy_threshold: float = 0.75,
) -> list[str]:
    reasons: list[str] = []
    if top_probability is None and downstream_eligible:
        reasons.append("classifier_not_run")
        return reasons  # no further QC meaningful without classifier output
    if yolo_confidence is not None and yolo_confidence < yolo_low_threshold:
        reasons.append("low_yolo_confidence")
    if segmentation_status and segmentation_status not in {"ok", "accepted"}:
        reasons.append("segmentation_suspicious")
    if segmentation_quality is not None and segmentation_quality < segmentation_low_threshold:
        reasons.append("low_segmentation_quality")
    if top_probability is not None and top_probability < top_probability_threshold:
        reasons.append("low_classifier_probability")
    margin = probability_margin(top_probability, top2_probability)
    if margin is not None and margin < margin_threshold:
        reasons.append("small_top1_top2_margin")
    if normalized_entropy(probabilities) > entropy_threshold:
        reasons.append("high_entropy")
    if overlap_score is not None and overlap_score > overlap_threshold:
        reasons.append("high_bbox_overlap")
    if rare_class:
        reasons.append("rare_or_immature_class")
    return reasons


def uncertainty_score(
    *,
    yolo_confidence: float | None,
    segmentation_quality: float | None,
    top_probability: float | None,
    top2_probability: float | None,
    probabilities: Mapping[str, float] | None,
    overlap_score: float | None,
) -> float:
    """Return a review-priority score in [0, 1-ish]. Higher means review earlier."""
    top = 0.0 if top_probability is None else max(0.0, min(1.0, float(top_probability)))
    margin_raw = probability_margin(top_probability, top2_probability)
    margin = 0.0 if margin_raw is None else max(0.0, min(1.0, margin_raw))
    yolo = 1.0 if yolo_confidence is None else max(0.0, min(1.0, float(yolo_confidence)))
    seg = 1.0 if segmentation_quality is None else max(0.0, min(1.0, float(segmentation_quality)))
    overlap = 0.0 if overlap_score is None else max(0.0, min(1.0, float(overlap_score)))
    score = (
        0.35 * (1.0 - top)
        + 0.25 * (1.0 - margin)
        + 0.15 * normalized_entropy(probabilities)
        + 0.10 * (1.0 - yolo)
        + 0.10 * (1.0 - seg)
        + 0.05 * overlap
    )
    return round(max(0.0, min(1.0, score)), 4)
