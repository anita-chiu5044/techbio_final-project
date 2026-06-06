"""
Convert YOLO WBC detections → context-padded square TIFF patches for MedSAM.

Reads detections.jsonl produced by export_yolo_detection_manifest.py,
can keep either all WBC detections or the top-1 highest-confidence WBC detection per source image,
crops a context-padded square ROI (white-padding at image boundaries),
and saves as TIFF under:

    OUTPUT_ROOT/{category}/{cell_type}/{image_id}.tiff

Usage:
    python yolo_to_medsam_patches.py \
        --detections /path/to/detections.jsonl \
        --output-root /path/to/medsam_input \
        [--context-scale 1.3] \
        [--wbc-label WBC] \
        [--selection all|top1] \
        [--mapping-csv /path/to/cell_map.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Allow importing roi.py from the ymca_agent package
sys.path.insert(0, str(Path(__file__).parent.parent / "ymca_agent"))
from roi import build_roi_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True,
                        help="Path to detections.jsonl from export_yolo_detection_manifest.py")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root directory for MedSAM TIFF patches")
    parser.add_argument("--context-scale", type=float, default=1.3,
                        help="Padding multiplier around YOLO bbox (default 1.3)")
    parser.add_argument("--wbc-label", default="WBC",
                        help="Class label string for white blood cells (default 'WBC')")
    parser.add_argument("--selection", choices=["all", "top1"], default="top1",
                        help="Keep all WBC detections or only top-1 WBC per source image. "
                             "Use all for real user sessions; top1 is for legacy single-cell datasets.")
    parser.add_argument("--mapping-csv", type=Path, default=None,
                        help="Optional CSV mapping MedSAM output mask image path to agent cell_id.")
    parser.add_argument("--exclude-ids", type=Path, default=None,
                        help="JSON file with {gated_ids: [...]} — detection IDs to skip (YOLO gating).")
    return parser.parse_args()


def rgba_to_rgb_white(img: Image.Image) -> Image.Image:
    """Composite RGBA image onto white background. Returns RGB image."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def crop_with_padding(img: Image.Image, roi_geom) -> Image.Image:
    """Crop the visible ROI region and add white padding to reach requested square size."""
    vx1, vy1, vx2, vy2 = roi_geom.roi_xyxy_original
    crop = rgba_to_rgb_white(img).crop((vx1, vy1, vx2, vy2))

    pad = roi_geom.padding_pixels
    if not roi_geom.padding_applied:
        return crop

    side = roi_geom.roi_side_requested_px
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(crop, (pad.left, pad.top))
    return padded


def load_wbc_records(detections_path: Path, wbc_label: str, selection: str) -> list[dict]:
    """Read detections.jsonl and keep WBC detections according to selection mode."""
    records: list[dict] = []
    best: dict[str, dict] = {}
    with detections_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["class_label"] != wbc_label:
                continue
            if selection == "all":
                records.append(rec)
                continue
            key = rec["source_image_path"]
            if key not in best or rec["confidence"] > best[key]["confidence"]:
                best[key] = rec
    return records if selection == "all" else list(best.values())


def output_labels_for_record(rec: dict) -> tuple[str, str]:
    """Return output folder labels for MedSAM input.

    Classified training data uses category/cell_type/source_image. Real user uploads
    may be flat, so fall back to session/WBC.
    """
    rel_parts = Path(rec.get("source_image_relative_path", "")).parts
    if len(rel_parts) >= 3:
        return rel_parts[0], rel_parts[1]
    if len(rel_parts) >= 2:
        return rel_parts[0], "WBC"
    return "session", "WBC"


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Load gated (excluded) detection IDs
    exclude_ids: set[str] = set()
    if args.exclude_ids and args.exclude_ids.exists():
        gated_data = json.loads(args.exclude_ids.read_text())
        exclude_ids = set(gated_data.get("gated_ids", []))
        print(f"Excluding {len(exclude_ids)} gated detection(s) (YOLO low confidence)")

    print(f"Loading detections and selecting WBC records (selection={args.selection})...")
    records = load_wbc_records(args.detections, args.wbc_label, args.selection)
    if exclude_ids:
        records = [r for r in records if r.get("detection_id") not in exclude_ids]
    print(f"Selected {len(records)} WBC detection(s)")

    saved = errors = 0
    mapping_rows: list[dict[str, str]] = []

    for line_no, rec in enumerate(records, 1):
            category, cell_type = output_labels_for_record(rec)

            try:
                roi_geom = build_roi_geometry(
                    bbox_xyxy_original=rec["bbox_xyxy_original"],
                    image_width=rec["image_width"],
                    image_height=rec["image_height"],
                    context_scale=args.context_scale,
                )
            except ValueError as exc:
                print(f"[WARN] {rec['detection_id']}: roi error: {exc}")
                errors += 1
                continue

            try:
                with Image.open(rec["source_image_path"]) as img:
                    patch = crop_with_padding(img, roi_geom)
            except Exception as exc:
                print(f"[WARN] {rec['detection_id']}: open/crop error: {exc}")
                errors += 1
                continue

            out_dir = args.output_root / category / cell_type
            out_dir.mkdir(parents=True, exist_ok=True)
            out_stem = f"{rec['image_id']}_{rec['detection_id']}"
            out_path = out_dir / f"{out_stem}.tiff"
            patch.save(out_path, format="TIFF")
            saved += 1

            if args.mapping_csv:
                expected_mask = out_path.with_suffix("").name + "_mask.png"
                mapping_rows.append({
                    "image": str(Path(category) / cell_type / expected_mask),
                    "cell_id": rec["detection_id"],
                    "medsam_input": str(out_path),
                    "source_image_path": rec["source_image_path"],
                    "detection_id": rec["detection_id"],
                })

            if saved % 1000 == 0:
                print(f"Saved {saved} patches...")

    if args.mapping_csv:
        import csv
        args.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.mapping_csv.open("w", newline="") as handle:
            fieldnames = ["image", "cell_id", "medsam_input", "source_image_path", "detection_id"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mapping_rows)
        print(f"Mapping CSV: {args.mapping_csv}")

    print(f"\nDone. saved={saved}  errors={errors}")
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()
