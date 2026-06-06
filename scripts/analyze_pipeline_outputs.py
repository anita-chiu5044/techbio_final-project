"""
Analyze YOLO and MedSAM pipeline outputs.

Outputs:
  - pipeline_output_analysis.md
  - pipeline_output_analysis.json

Example:
  python scripts/analyze_pipeline_outputs.py \
    --yolo-dir /home/yucheng/Desktop/techbio_pipeline_output/yolo \
    --medsam-dir /home/yucheng/Desktop/techbio_pipeline_output/medsam_output \
    --out-dir docs/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from PIL import Image

try:
    import numpy as np
except ImportError:  # pragma: no cover - slow fallback is acceptable for tiny checks.
    np = None


DEFAULT_YOLO_DIR = Path("/home/yucheng/Desktop/techbio_pipeline_output/yolo")
DEFAULT_MEDSAM_DIR = Path("/home/yucheng/Desktop/techbio_pipeline_output/medsam_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO and MedSAM pipeline outputs.")
    parser.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO_DIR)
    parser.add_argument("--medsam-dir", type=Path, default=DEFAULT_MEDSAM_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/analysis"))
    parser.add_argument("--mask-sample-limit", type=int, default=0,
                        help="0 means analyze all OK masks; set a smaller number for smoke checks.")
    parser.add_argument("--overlap-threshold", type=float, default=0.50)
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ["min", "p05", "p25", "median", "p75", "p95", "max", "mean"]}
    xs = sorted(values)

    def q(frac: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = frac * (len(xs) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {
        "min": round(xs[0], 6),
        "p05": round(q(0.05), 6),
        "p25": round(q(0.25), 6),
        "median": round(q(0.50), 6),
        "p75": round(q(0.75), 6),
        "p95": round(q(0.95), 6),
        "max": round(xs[-1], 6),
        "mean": round(mean(xs), 6),
    }


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at {path}:{line_no}") from exc


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def analyze_yolo(yolo_dir: Path, overlap_threshold: float) -> dict:
    summary_path = yolo_dir / "summary.json"
    images_path = yolo_dir / "images.jsonl"
    detections_path = yolo_dir / "detections.jsonl"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    images = list(read_jsonl(images_path))
    detections_by_class: Counter[str] = Counter()
    conf_by_class: dict[str, list[float]] = defaultdict(list)
    wbc_boxes_by_image: dict[str, list[list[float]]] = defaultdict(list)
    wbc_conf_by_image: dict[str, list[float]] = defaultdict(list)
    bbox_widths: dict[str, list[float]] = defaultdict(list)
    bbox_heights: dict[str, list[float]] = defaultdict(list)
    bbox_area_frac: dict[str, list[float]] = defaultdict(list)

    detection_rows = 0
    for det in read_jsonl(detections_path):
        detection_rows += 1
        label = det.get("class_label", "")
        conf = float(det.get("confidence", 0.0))
        bbox = [float(x) for x in det.get("bbox_xyxy_original", [0, 0, 0, 0])]
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        img_area = max(1.0, float(det.get("image_width", 1)) * float(det.get("image_height", 1)))
        key = det.get("source_image_path") or det.get("image_id", "")

        detections_by_class[label] += 1
        conf_by_class[label].append(conf)
        bbox_widths[label].append(width)
        bbox_heights[label].append(height)
        bbox_area_frac[label].append(width * height / img_area)
        if label == "WBC":
            wbc_boxes_by_image[key].append(bbox)
            wbc_conf_by_image[key].append(conf)

    image_count = len(images)
    wbc_counts = [len(wbc_boxes_by_image.get(img.get("source_image_path", ""), [])) for img in images]
    zero_wbc = sum(1 for x in wbc_counts if x == 0)
    low_wbc_conf = sum(1 for vals in wbc_conf_by_image.values() for v in vals if v < 0.5)
    total_wbc = detections_by_class.get("WBC", 0)

    max_ious = []
    high_overlap_images = 0
    high_overlap_pairs = 0
    for boxes in wbc_boxes_by_image.values():
        local_max = 0.0
        local_high = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = box_iou(boxes[i], boxes[j])
                local_max = max(local_max, iou)
                if iou >= overlap_threshold:
                    local_high += 1
        if boxes:
            max_ious.append(local_max)
        if local_high:
            high_overlap_images += 1
            high_overlap_pairs += local_high

    return {
        "summary_json": summary,
        "image_count": image_count,
        "detection_rows": detection_rows,
        "detections_by_class": dict(detections_by_class),
        "confidence_by_class": {k: quantiles(v) for k, v in conf_by_class.items()},
        "bbox_width_by_class": {k: quantiles(v) for k, v in bbox_widths.items()},
        "bbox_height_by_class": {k: quantiles(v) for k, v in bbox_heights.items()},
        "bbox_area_fraction_by_class": {k: quantiles(v) for k, v in bbox_area_frac.items()},
        "wbc_per_image": quantiles([float(x) for x in wbc_counts]),
        "zero_wbc_images": zero_wbc,
        "wbc_low_conf_lt_0_5": low_wbc_conf,
        "wbc_low_conf_fraction": round(low_wbc_conf / total_wbc, 6) if total_wbc else None,
        "wbc_top1_candidate_images": image_count - zero_wbc,
        "wbc_overlap": {
            "threshold": overlap_threshold,
            "max_iou_by_image": quantiles(max_ious),
            "images_with_high_overlap": high_overlap_images,
            "high_overlap_pairs": high_overlap_pairs,
        },
    }


def corrected_label_from_mask_path(row: dict) -> tuple[str, str]:
    mask_path = row.get("mask_path", "")
    if mask_path:
        p = Path(mask_path)
        if p.parent.name:
            category = p.parent.parent.name or row.get("category", "")
            return category, p.parent.name
    return row.get("category", ""), row.get("cell_type", "")


def mask_stats(mask_path: Path) -> dict[str, float | int | bool]:
    with Image.open(mask_path) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        if np is not None:
            arr = np.asarray(rgba)
            alpha = arr[:, :, 3] > 0
            not_white = (arr[:, :, :3] < 250).any(axis=2)
            fg = alpha & not_white
            foreground = int(fg.sum())
            if foreground:
                edge_pixels = int(fg[0, :].sum() + fg[-1, :].sum() + fg[:, 0].sum() + fg[:, -1].sum())
                ys, xs = np.where(fg)
                bbox_w = int(xs.max() - xs.min() + 1)
                bbox_h = int(ys.max() - ys.min() + 1)
            else:
                edge_pixels = 0
                bbox_w = bbox_h = 0
        else:
            pix = rgba.load()
            foreground = edge_pixels = 0
            xs_seen, ys_seen = [], []
            for y in range(height):
                for x in range(width):
                    r, g, b, a = pix[x, y]
                    is_fg = a > 0 and (r < 250 or g < 250 or b < 250)
                    if is_fg:
                        foreground += 1
                        xs_seen.append(x)
                        ys_seen.append(y)
                        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                            edge_pixels += 1
            bbox_w = max(xs_seen) - min(xs_seen) + 1 if xs_seen else 0
            bbox_h = max(ys_seen) - min(ys_seen) + 1 if ys_seen else 0

    total = max(1, width * height)
    area_ratio = foreground / total
    edge_ratio = edge_pixels / foreground if foreground else 0.0
    bbox_coverage = (bbox_w * bbox_h / total) if total else 0.0
    return {
        "width": width,
        "height": height,
        "foreground_pixels": foreground,
        "area_ratio": area_ratio,
        "edge_ratio": edge_ratio,
        "bbox_coverage": bbox_coverage,
        "blank": foreground == 0,
    }


def analyze_medsam(medsam_dir: Path, mask_sample_limit: int) -> dict:
    summary_path = medsam_dir / "inference_summary.csv"
    rows = []
    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category, cell_type = corrected_label_from_mask_path(row)
            row["corrected_category"] = category
            row["corrected_cell_type"] = cell_type
            rows.append(row)

    status_counts = Counter(row["status"].upper() for row in rows)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    no_detection_examples: dict[str, list[str]] = defaultdict(list)
    cell_type_mismatch = 0
    multi_detection_by_class: Counter[str] = Counter()
    num_det_by_class: dict[str, list[float]] = defaultdict(list)

    coverage_by_class: dict[str, list[float]] = defaultdict(list)
    edge_by_class: dict[str, list[float]] = defaultdict(list)
    bbox_coverage_by_class: dict[str, list[float]] = defaultdict(list)
    suspicious_by_class: Counter[str] = Counter()
    mask_errors: list[dict[str, str]] = []
    checked_masks = 0

    for row in rows:
        status = row["status"].upper()
        category = row["corrected_category"]
        cell_type = row["corrected_cell_type"]
        if row.get("cell_type") != cell_type:
            cell_type_mismatch += 1
        by_class[cell_type][status] += 1
        by_category[category][status] += 1
        try:
            n_det = int(row.get("num_detections") or 0)
        except ValueError:
            n_det = 0
        num_det_by_class[cell_type].append(float(n_det))
        if n_det > 1:
            multi_detection_by_class[cell_type] += 1
        if status == "NO_DETECTION" and len(no_detection_examples[cell_type]) < 8:
            no_detection_examples[cell_type].append(row.get("image", ""))
        if status != "OK":
            continue
        if mask_sample_limit and checked_masks >= mask_sample_limit:
            continue
        mask_path = Path(row.get("mask_path", ""))
        checked_masks += 1
        try:
            stats = mask_stats(mask_path)
        except Exception as exc:  # noqa: BLE001 - report file-level failures.
            mask_errors.append({"mask_path": str(mask_path), "error": str(exc)})
            continue
        area = float(stats["area_ratio"])
        edge = float(stats["edge_ratio"])
        bbox_cov = float(stats["bbox_coverage"])
        coverage_by_class[cell_type].append(area)
        edge_by_class[cell_type].append(edge)
        bbox_coverage_by_class[cell_type].append(bbox_cov)
        if stats["blank"] or area < 0.02 or area > 0.90 or edge > 0.15:
            suspicious_by_class[cell_type] += 1

    return {
        "total_rows": len(rows),
        "status_counts": dict(status_counts),
        "ok_fraction": round(status_counts.get("OK", 0) / len(rows), 6) if rows else None,
        "by_class_status": {k: dict(v) for k, v in sorted(by_class.items())},
        "by_category_status": {k: dict(v) for k, v in sorted(by_category.items())},
        "cell_type_mismatch_rows_corrected_from_mask_path": cell_type_mismatch,
        "no_detection_examples_by_class": dict(no_detection_examples),
        "num_detections_by_class": {k: quantiles(v) for k, v in sorted(num_det_by_class.items())},
        "multi_detection_ok_rows_by_class": dict(sorted(multi_detection_by_class.items())),
        "mask_coverage_area_ratio_by_class": {k: quantiles(v) for k, v in sorted(coverage_by_class.items())},
        "mask_edge_ratio_by_class": {k: quantiles(v) for k, v in sorted(edge_by_class.items())},
        "mask_bbox_coverage_by_class": {k: quantiles(v) for k, v in sorted(bbox_coverage_by_class.items())},
        "suspicious_mask_count_by_class": dict(sorted(suspicious_by_class.items())),
        "checked_masks": checked_masks,
        "mask_errors": mask_errors[:20],
    }


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.2f}%" if d else "n/a"


def render_markdown(yolo: dict, medsam: dict) -> str:
    y_counts = yolo["detections_by_class"]
    m_status = medsam["status_counts"]
    total_m = medsam["total_rows"]

    yolo_rows = []
    for label, count in sorted(y_counts.items()):
        conf = yolo["confidence_by_class"].get(label, {})
        area = yolo["bbox_area_fraction_by_class"].get(label, {})
        yolo_rows.append([
            label, count, conf.get("median"), conf.get("p05"), conf.get("p95"), area.get("median"),
        ])

    medsam_rows = []
    for label, counts in medsam["by_class_status"].items():
        total = sum(counts.values())
        cov = medsam["mask_coverage_area_ratio_by_class"].get(label, {})
        edge = medsam["mask_edge_ratio_by_class"].get(label, {})
        medsam_rows.append([
            label,
            total,
            counts.get("OK", 0),
            counts.get("NO_DETECTION", 0),
            counts.get("FAIL", 0),
            pct(counts.get("OK", 0), total),
            cov.get("median"),
            cov.get("p05"),
            cov.get("p95"),
            edge.get("p95"),
            medsam["suspicious_mask_count_by_class"].get(label, 0),
        ])

    no_det_rows = []
    for label, examples in sorted(medsam["no_detection_examples_by_class"].items()):
        no_det_rows.append([label, len(examples), ", ".join(examples[:5])])

    return f"""# Pipeline Output Analysis

Generated from local YOLO and MedSAM outputs.

## Executive Summary

- YOLO processed `{yolo['image_count']}` images and produced `{yolo['detection_rows']}` detections.
- YOLO WBC detections: `{y_counts.get('WBC', 0)}`. Images with at least one WBC candidate: `{yolo['wbc_top1_candidate_images']}`. Images with zero WBC candidate: `{yolo['zero_wbc_images']}`.
- WBC detections below confidence 0.5: `{yolo['wbc_low_conf_lt_0_5']}` (`{yolo['wbc_low_conf_fraction']}`).
- MedSAM rows: `{total_m}`. OK: `{m_status.get('OK', 0)}` (`{medsam['ok_fraction']}`). NO_DETECTION: `{m_status.get('NO_DETECTION', 0)}`. FAIL: `{m_status.get('FAIL', 0)}`.
- MedSAM label correction: `{medsam['cell_type_mismatch_rows_corrected_from_mask_path']}` rows had CSV `cell_type` corrected from `mask_path` parent folder. This matters for LYA files named `WBC-Malignant-...`.
- Mask coverage statistics below use foreground area / patch area. Extremely tiny, huge, or edge-touching masks should be reviewed before trusting classifier training.

## YOLO Analysis

### Detection Counts And Confidence

{md_table(['class', 'count', 'conf_median', 'conf_p05', 'conf_p95', 'bbox_area_frac_median'], yolo_rows)}

### WBC Candidate Distribution

```json
{json.dumps({'wbc_per_image': yolo['wbc_per_image'], 'zero_wbc_images': yolo['zero_wbc_images'], 'overlap': yolo['wbc_overlap']}, indent=2)}
```

Interpretation:

- The current retraining shortcut used top-1 WBC per source image because the source dataset is supposed to be single-cell crops.
- For real clinical chat sessions, keep all WBC detections and route high-overlap candidates to review rather than forcing single-cell classification.
- Low-confidence WBC detections should stay in the review/QC path and should not be treated as reliable morphology evidence.

## MedSAM Analysis

### Status By Corrected Class

{md_table(['class', 'total', 'OK', 'NO_DETECTION', 'FAIL', 'OK_rate', 'coverage_median', 'coverage_p05', 'coverage_p95', 'edge_p95', 'suspicious_masks'], medsam_rows)}

### NO_DETECTION Examples

{md_table(['class', 'example_count_listed', 'examples'], no_det_rows) if no_det_rows else 'No NO_DETECTION rows.'}

### Why Some MedSAM Rows Failed Or Had No Detection

Observed from this run:

- There were `{m_status.get('FAIL', 0)}` hard FAIL rows, so no Python/runtime exception class dominates this run.
- There were `{m_status.get('NO_DETECTION', 0)}` NO_DETECTION rows. These mean the model ran but found zero target objects under the prompt/threshold settings.
- NO_DETECTION is likely caused by one or more of: weak/atypical WBC appearance after YOLO ROI, overly strict MedSAM threshold, cells near crop boundary, background/alpha artifacts, or source labels whose visual content is not a clean WBC.
- The NO_DETECTION rows should be visually sampled before excluding a class or changing thresholds globally.

## Training Implications

- Use corrected labels from folder structure or `mask_path`, not filename prefixes.
- Do not train on fake `WBC` class from LYA filenames.
- Consider excluding or downweighting suspicious masks after visual review.
- Report MedSAM status by class together with classifier metrics; otherwise classifier failures may hide upstream segmentation failures.

## Recommended Next Checks

1. Visually inspect NO_DETECTION examples, especially lymphoid/LYA examples.
2. Inspect the highest and lowest coverage masks by class.
3. Add optional mask-QC filtering to classifier training.
4. Keep YOLO low-confidence and high-overlap candidates review-required in the agent DB.
"""


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    yolo = analyze_yolo(args.yolo_dir, args.overlap_threshold)
    medsam = analyze_medsam(args.medsam_dir, args.mask_sample_limit)
    payload = {"yolo": yolo, "medsam": medsam}
    (args.out_dir / "pipeline_output_analysis.json").write_text(json.dumps(payload, indent=2))
    (args.out_dir / "pipeline_output_analysis.md").write_text(render_markdown(yolo, medsam))
    print(f"Wrote {args.out_dir / 'pipeline_output_analysis.md'}")
    print(f"Wrote {args.out_dir / 'pipeline_output_analysis.json'}")


if __name__ == "__main__":
    main()
