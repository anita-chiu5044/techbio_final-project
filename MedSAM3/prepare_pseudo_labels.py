"""
Convert manually selected (TIFF, RGBA-mask PNG) pairs → COCO JSON
for fine-tuning MedSAM3 with full segmentation supervision.

Background
----------
After running tiff_wbc_inference.py with --masked-output --fill-holes,
manually inspect the results and note which images have good masks
(nucleus + cytoplasm both covered). List those image stems in a text
file, then run this script to build a COCO dataset ready for training.

Input layout (mirrors tiff_wbc_inference.py output):
    data_root/
        erythroid/EBO/EBO_0001.tiff
        granulocyte_mature/NGS/NGS_0042.tiff
        ...
    mask_dir/
        erythroid/EBO/EBO_0001_mask.png   ← RGBA PNG from --masked-output
        granulocyte_mature/NGS/NGS_0042_mask.png
        ...

Output:
    output_dir/
        train/
            _annotations.coco.json
            EBO_0001.tiff   (symlink or copy)
            ...
        valid/
            _annotations.coco.json
            ...

Usage
-----
# Process all masks found in mask_dir:
    python prepare_pseudo_labels.py \\
        --data-root ~/nas2/.../nas_wbc_crops_bccd_400_white \\
        --mask-dir  ~/Desktop/wbc_tiff_results_fillholes \\
        --output-dir data/pseudo_erythroid_gran

# Process only a curated list (one stem per line, e.g. "EBO_0001"):
    python prepare_pseudo_labels.py \\
        --data-root ~/nas2/.../nas_wbc_crops_bccd_400_white \\
        --mask-dir  ~/Desktop/wbc_tiff_results_fillholes \\
        --output-dir data/pseudo_erythroid_gran \\
        --selected good_samples.txt \\
        --categories erythroid granulocyte_mature
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


WBC_CATEGORY = {"id": 1, "name": "white blood cell"}
MIN_CONTOUR_AREA = 100   # pixels² — discard tiny noise contours


def mask_png_to_polygon(mask_path: Path):
    """
    Read an RGBA mask PNG (alpha=255 inside cell, 0 outside) and return:
        segmentation  [[x1,y1,x2,y2,...]]   COCO polygon of the largest contour
        bbox          [x, y, w, h]           COCO bounding box
        area          float

    Returns None if no valid contour is found.
    """
    rgba = np.array(Image.open(mask_path).convert("RGBA"))
    binary = (rgba[:, :, 3] > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep the largest contour (the cell body)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return None

    # Flatten contour → COCO polygon [x1, y1, x2, y2, ...]
    segmentation = [contour.flatten().tolist()]

    x, y, w, h = cv2.boundingRect(contour)
    bbox = [int(x), int(y), int(w), int(h)]
    area = float(cv2.contourArea(contour))

    return segmentation, bbox, area


def find_tiff(data_root: Path, relative_mask_path: Path) -> Path | None:
    """
    Given the relative path of a mask PNG inside mask_dir, reconstruct the
    path to the original TIFF inside data_root.

    Example:
        relative_mask_path = erythroid/EBO/EBO_0001_mask.png
        → data_root/erythroid/EBO/EBO_0001.tiff
    """
    # Strip _mask suffix and change extension
    parts = relative_mask_path.parts       # ('erythroid', 'EBO', 'EBO_0001_mask.png')
    stem = parts[-1].replace("_mask.png", "")
    tiff_path = data_root.joinpath(*parts[:-1]) / (stem + ".tiff")
    return tiff_path if tiff_path.exists() else None


def build_coco_skeleton():
    return {
        "info": {"description": "MedSAM3 pseudo-label dataset"},
        "images": [],
        "annotations": [],
        "categories": [WBC_CATEGORY],
    }


def collect_records(mask_dir: Path, data_root: Path,
                    selected_stems: set | None,
                    categories: list | None) -> list:
    """Walk mask_dir and collect valid (tiff_path, mask_path) pairs."""
    records = []
    for mask_path in sorted(mask_dir.rglob("*_mask.png")):
        rel = mask_path.relative_to(mask_dir)

        # Category filter (first directory level under mask_dir)
        if categories and rel.parts[0] not in categories:
            continue

        # Selected-stems filter
        stem = mask_path.name.replace("_mask.png", "")
        if selected_stems is not None and stem not in selected_stems:
            continue

        tiff_path = find_tiff(data_root, rel)
        if tiff_path is None:
            print(f"  [WARN] No TIFF found for {rel} — skipped")
            continue

        records.append({"tiff": tiff_path, "mask": mask_path, "stem": stem})

    return records


def write_split(records: list, split_dir: Path, category_name: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    coco = build_coco_skeleton()
    coco["categories"] = [{"id": 1, "name": category_name}]

    image_id = 0
    ann_id = 0
    skipped = 0

    for rec in records:
        result = mask_png_to_polygon(rec["mask"])
        if result is None:
            print(f"  [WARN] No valid contour in {rec['mask'].name} — skipped")
            skipped += 1
            continue

        segmentation, bbox, area = result

        # Get image size from TIFF
        with Image.open(rec["tiff"]) as img:
            img_w, img_h = img.size

        # Copy TIFF into split directory
        dst = split_dir / rec["tiff"].name
        if not dst.exists():
            shutil.copy2(rec["tiff"], dst)

        coco["images"].append({
            "id": image_id,
            "file_name": rec["tiff"].name,
            "width": img_w,
            "height": img_h,
        })

        coco["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })

        image_id += 1
        ann_id += 1

    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  {split_dir.name:6s}: {image_id} images, {ann_id} annotations"
          + (f", {skipped} skipped" if skipped else ""))
    print(f"           → {ann_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert selected pseudo-label (TIFF + mask PNG) pairs to COCO JSON"
    )
    parser.add_argument("--data-root",   required=True,
                        help="Root of original TIFF files")
    parser.add_argument("--mask-dir",    required=True,
                        help="Inference output dir containing *_mask.png files")
    parser.add_argument("--output-dir",  required=True,
                        help="Output COCO dataset directory")
    parser.add_argument("--selected",    default=None,
                        help="Text file with one image stem per line to include "
                             "(e.g. 'EBO_0001'). Omit to include all masks found.")
    parser.add_argument("--categories",  nargs="+", default=None,
                        help="Limit to specific category folders "
                             "(e.g. erythroid granulocyte_mature)")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Fraction of samples used for train (default: 0.8)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--category-name", default="white blood cell",
                        help="COCO category name (default: 'white blood cell')")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    mask_dir  = Path(args.mask_dir).expanduser()
    output    = Path(args.output_dir).expanduser()

    if not data_root.exists():
        raise SystemExit(f"[ERROR] data-root not found: {data_root}")
    if not mask_dir.exists():
        raise SystemExit(f"[ERROR] mask-dir not found: {mask_dir}")

    # Load optional selected-stems list
    selected_stems = None
    if args.selected:
        with open(args.selected) as f:
            selected_stems = {line.strip() for line in f if line.strip()}
        print(f"Selected stems loaded: {len(selected_stems)}")

    print(f"Scanning {mask_dir} ...")
    records = collect_records(mask_dir, data_root, selected_stems, args.categories)

    if not records:
        raise SystemExit("[ERROR] No valid pairs found.")

    print(f"Found {len(records)} pairs")
    for cat in sorted({r['tiff'].parent.parent.name for r in records}):
        n = sum(1 for r in records if r['tiff'].parent.parent.name == cat)
        print(f"  {cat}: {n}")

    # Train / valid split
    random.seed(args.seed)
    shuffled = records[:]
    random.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * args.split_ratio))
    train_recs = shuffled[:n_train]
    valid_recs = shuffled[n_train:]

    print(f"\nSplit: {len(train_recs)} train / {len(valid_recs)} valid\n")

    write_split(train_recs, output / "train", args.category_name)
    if valid_recs:
        write_split(valid_recs, output / "valid", args.category_name)
    else:
        print("  valid : 0 images (all samples went to train)")

    print(f"\nDone! Now run:")
    print(f"  python train_sam3_lora_native.py --config configs/full_lora_config.yaml")
    print(f"  (update data_dir in config to '{output}')")


if __name__ == "__main__":
    main()
