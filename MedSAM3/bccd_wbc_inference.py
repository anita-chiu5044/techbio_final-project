"""
BCCD White Blood Cell Batch Inference with MedSAM3
====================================================
This script:
1. Reads BCCD Supervisely-format JSON annotations
2. Extracts WBC bounding boxes from each image
3. Runs MedSAM3 inference for each WBC bounding box
4. Saves output masks and a summary CSV

Usage:
    python bccd_wbc_inference.py \
        --img-dir ~/Desktop/archive/train/img \
        --ann-dir ~/Desktop/archive/train/ann \
        --config ~/Desktop/MedSAM3/configs/full_lora_config.yaml \
        --output-dir ~/Desktop/wbc_results \
        --threshold 0.5 \
        --nms-iou 0.5
"""

import os
import json
import argparse
import subprocess
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Batch WBC inference with MedSAM3")
    parser.add_argument("--img-dir",    required=True,  help="Path to image folder (*.jpeg)")
    parser.add_argument("--ann-dir",    required=True,  help="Path to annotation folder (*.jpeg.json)")
    parser.add_argument("--config",     required=True,  help="Path to MedSAM3 config YAML")
    parser.add_argument("--medsam3-dir",default=os.path.expanduser("~/Desktop/MedSAM3"),
                        help="Root directory of MedSAM3 repo (default: ~/Desktop/MedSAM3)")
    parser.add_argument("--output-dir", required=True,  help="Directory to save output masks")
    parser.add_argument("--prompt",     default="white blood cell",
                        help="Text prompt for MedSAM3 (default: 'white blood cell')")
    parser.add_argument("--threshold",  type=float, default=0.5, help="Score threshold")
    parser.add_argument("--nms-iou",    type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--max-images", type=int,   default=None,
                        help="Limit number of images to process (for quick testing)")
    return parser.parse_args()


def extract_wbc_boxes(ann_path):
    """
    Parse a Supervisely JSON annotation file and return all WBC bounding boxes.
    Returns list of dicts: {x1, y1, x2, y2}
    """
    with open(ann_path, "r") as f:
        ann = json.load(f)

    boxes = []
    for obj in ann.get("objects", []):
        if obj.get("classTitle", "").upper() == "WBC":
            exterior = obj["points"]["exterior"]
            if len(exterior) >= 2:
                x1 = int(min(exterior[0][0], exterior[1][0]))
                y1 = int(min(exterior[0][1], exterior[1][1]))
                x2 = int(max(exterior[0][0], exterior[1][0]))
                y2 = int(max(exterior[0][1], exterior[1][1]))
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def run_medsam3_inference(medsam3_dir, config, image_path, output_path,
                           prompt, threshold, nms_iou):
    """
    Call MedSAM3's infer_sam.py as a subprocess.
    Returns (success: bool, stderr: str)
    """
    infer_script = os.path.join(medsam3_dir, "infer_sam.py")
    cmd = [
        "python3", infer_script,
        "--config",    config,
        "--image",     str(image_path),
        "--prompt",    prompt,
        "--threshold", str(threshold),
        "--nms-iou",   str(nms_iou),
        "--output",    str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def main():
    args = parse_args()

    img_dir    = Path(args.img_dir).expanduser()
    ann_dir    = Path(args.ann_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all .jpeg images
    image_files = sorted(img_dir.glob("*.jpeg"))
    if args.max_images:
        image_files = image_files[: args.max_images]

    print(f"Found {len(image_files)} images in {img_dir}")
    print(f"Prompt        : '{args.prompt}'")
    print(f"Threshold     : {args.threshold}")
    print(f"NMS IoU       : {args.nms_iou}")
    print(f"Output dir    : {output_dir}")
    print("-" * 60)

    summary_rows = []
    total_wbc    = 0
    success_cnt  = 0

    for img_path in image_files:
        stem    = img_path.stem                          # e.g. BloodImage_00090
        ann_path = ann_dir / (img_path.name + ".json")  # BloodImage_00090.jpeg.json

        if not ann_path.exists():
            print(f"[SKIP] No annotation found for {img_path.name}")
            continue

        wbc_boxes = extract_wbc_boxes(ann_path)
        if not wbc_boxes:
            print(f"[SKIP] No WBC found in {img_path.name}")
            continue

        print(f"[INFO] {img_path.name} — {len(wbc_boxes)} WBC(s)")
        total_wbc += len(wbc_boxes)

        # Run inference once per image (MedSAM3 detects all instances with the text prompt)
        out_mask_path = output_dir / f"{stem}_mask.png"
        ok, err = run_medsam3_inference(
            medsam3_dir  = args.medsam3_dir,
            config       = args.config,
            image_path   = img_path,
            output_path  = out_mask_path,
            prompt       = args.prompt,
            threshold    = args.threshold,
            nms_iou      = args.nms_iou,
        )

        status = "OK" if ok else "FAIL"
        if ok:
            success_cnt += 1
            print(f"  ✓  Saved mask → {out_mask_path.name}")
        else:
            print(f"  ✗  Inference failed: {err.strip()[-200:]}")

        summary_rows.append({
            "image":     img_path.name,
            "wbc_count": len(wbc_boxes),
            "status":    status,
            "mask_path": str(out_mask_path) if ok else "",
        })

    # Save summary CSV
    csv_path = output_dir / "inference_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "wbc_count", "status", "mask_path"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 60)
    print(f"Done! {success_cnt}/{len(summary_rows)} images succeeded.")
    print(f"Total WBC annotations : {total_wbc}")
    print(f"Summary CSV saved     : {csv_path}")


if __name__ == "__main__":
    main()