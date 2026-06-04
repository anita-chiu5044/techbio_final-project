"""
Convert LISC Dataset (YOLO format) → COCO format for MedSAM3 fine-tuning
=========================================================================
LISC input structure:
    LISC/
        train/images/*.bmp   train/labels/*.txt
        valid/images/*.bmp   valid/labels/*.txt

YOLO label format (per line):
    class_id  cx  cy  w  h   (all normalized to [0, 1])

Output structure expected by train_sam3_lora_native.py (COCOSegmentDataset):
    {output_dir}/
        train/
            _annotations.coco.json
            Baso_10.bmp  ...          <- images copied here
        valid/
            _annotations.coco.json
            ...

All 5 LISC WBC classes are collapsed into a single "white blood cell" category
to match the inference prompt used in tiff_wbc_inference.py.
No segmentation masks are generated (bbox-only fine-tuning).

Usage:
    python prepare_lisc_for_medsam3.py
    python prepare_lisc_for_medsam3.py --lisc-dir LISC --output-dir data/lisc_medsam3
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


WBC_CATEGORY = {"id": 1, "name": "white blood cell"}


def convert_split(lisc_split_dir: Path, output_split_dir: Path, split: str) -> None:
    output_split_dir.mkdir(parents=True, exist_ok=True)
    images_dir = lisc_split_dir / "images"
    labels_dir = lisc_split_dir / "labels"

    bmp_files = sorted(images_dir.glob("*.bmp"))
    if not bmp_files:
        print(f"  [WARN] No .bmp files in {images_dir}")
        return

    coco = {
        "info": {"description": "LISC WBC dataset converted for MedSAM3 fine-tuning"},
        "images": [],
        "annotations": [],
        "categories": [WBC_CATEGORY],
    }

    image_id = 0
    ann_id = 0
    skipped = 0

    for bmp_path in bmp_files:
        label_path = labels_dir / (bmp_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        with Image.open(bmp_path) as img:
            img_w, img_h = img.size

        # Copy image into flat output directory
        dst = output_split_dir / bmp_path.name
        if not dst.exists():
            shutil.copy2(bmp_path, dst)

        coco["images"].append({
            "id": image_id,
            "file_name": bmp_path.name,
            "width": img_w,
            "height": img_h,
        })

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # class_id ignored — all WBC types map to "white blood cell"
                cx, cy, bw, bh = map(float, parts[1:5])

                # Normalized YOLO → absolute COCO [x, y, w, h]
                abs_w = bw * img_w
                abs_h = bh * img_h
                abs_x = cx * img_w - abs_w / 2
                abs_y = cy * img_h - abs_h / 2

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [round(abs_x, 2), round(abs_y, 2),
                             round(abs_w, 2), round(abs_h, 2)],
                    "area": round(abs_w * abs_h, 2),
                    "iscrowd": 0,
                    # No "segmentation" field → segment=None in COCOSegmentDataset
                    # Only bbox + classification losses will fire during training
                })
                ann_id += 1

        image_id += 1

    ann_path = output_split_dir / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  {split:6s}: {image_id} images, {ann_id} boxes"
          + (f", {skipped} skipped (no label)" if skipped else ""))
    print(f"           → {ann_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LISC YOLO labels to COCO JSON for MedSAM3"
    )
    parser.add_argument("--lisc-dir",   default="LISC",
                        help="Path to LISC dataset root (default: LISC)")
    parser.add_argument("--output-dir", default="data/lisc_medsam3",
                        help="Output directory (default: data/lisc_medsam3)")
    args = parser.parse_args()

    lisc = Path(args.lisc_dir).expanduser()
    out  = Path(args.output_dir).expanduser()

    if not lisc.exists():
        raise SystemExit(f"[ERROR] LISC directory not found: {lisc}")

    print(f"Converting LISC → COCO format")
    print(f"  Source : {lisc}")
    print(f"  Output : {out}")
    print(f"  Prompt : 'white blood cell' (all 5 WBC types unified)\n")

    for split in ["train", "valid"]:
        src = lisc / split
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue
        convert_split(src, out / split, split)

    print(f"\nDone! Now run:")
    print(f"  python train_sam3_lora_native.py --config configs/lisc_lora_config.yaml")


if __name__ == "__main__":
    main()
