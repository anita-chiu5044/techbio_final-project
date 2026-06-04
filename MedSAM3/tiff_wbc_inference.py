"""
AML Cytomorphology TIFF Batch Inference with MedSAM3
=====================================================
Processes pre-cropped WBC TIFF images (400x400 RGBA) organised as:
    {data_root}/{category}/{cell_type}/[ALL/]{name}.tiff

For each image MedSAM3 is called once via infer_sam.py and the output
mask is saved under output_dir mirroring the same sub-folder structure.

Usage:
    python tiff_wbc_inference.py \
        --data-root "~/nas2/anita/TechBio/classified/PKG - AML-Cytomorphology_LMU/nas_wbc_crops_bccd_400_white" \
        --config    ~/Desktop/MedSAM3/configs/full_lora_config.yaml \
        --output-dir ~/Desktop/wbc_tiff_results \
        --prompt "white blood cell" \
        --threshold 0.5 \
        --nms-iou 0.5

Optional filters:
    --categories granulocyte_mature lymphoid   (process only these categories)
    --cell-types NGS LYT                       (process only these cell-type codes)
    --max-images 100                           (cap total images for a quick test)
    --skip-existing                            (skip images whose mask already exists)

Defaults (on by default, disable with --no-* flags):
    --suppress-rbc      replace RBC-like (pink/red) pixels with white before inference
    --wbc-mode          keep only the single most-central, largest-area detection per image
    --min-mask-area-frac 0.02  discard masks smaller than 2% of image area in wbc-mode
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch MedSAM3 inference on AML Cytomorphology TIFF crops"
    )
    parser.add_argument("--data-root",   required=True,
                        help="Root dir containing category sub-folders with .tiff files")
    parser.add_argument("--config",      required=True,
                        help="Path to MedSAM3 config YAML")
    parser.add_argument("--output-dir",  required=True,
                        help="Directory to save output masks (structure mirrors data-root)")
    parser.add_argument("--medsam3-dir", default=os.path.expanduser("~/Desktop/MedSAM3"),
                        help="MedSAM3 repo root (default: ~/Desktop/MedSAM3)")
    parser.add_argument("--prompt",      default="white blood cell",
                        help="Text prompt (default: 'white blood cell')")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--nms-iou",     type=float, default=0.5)
    parser.add_argument("--max-images",  type=int,   default=None,
                        help="Stop after processing this many images (for testing)")
    parser.add_argument("--max-per-type", type=int, default=None,
                        help="Limit to this many images per cell-type (e.g. 5 for a quick sanity check)")
    parser.add_argument("--categories",  nargs="+", default=None,
                        help="Limit to specific category folders (e.g. granulocyte_mature lymphoid)")
    parser.add_argument("--cell-types",  nargs="+", default=None,
                        help="Limit to specific 3-letter cell-type codes (e.g. NGS LYT BAS)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images whose output mask file already exists")
    parser.add_argument("--suppress-rbc", action="store_true", default=True,
                        help="Replace RBC-like (pink/red) pixels with white before inference (default: True)")
    parser.add_argument("--no-suppress-rbc", dest="suppress_rbc", action="store_false",
                        help="Disable RBC suppression pre-processing")
    parser.add_argument("--wbc-mode", action="store_true", default=True,
                        help="Keep only the single most-central, largest-area detection per image "
                             "(default: True, designed for pre-cropped WBC images)")
    parser.add_argument("--no-wbc-mode", dest="wbc_mode", action="store_false",
                        help="Disable WBC mode (return all detections above threshold)")
    parser.add_argument("--min-mask-area-frac", type=float, default=0.02,
                        help="Minimum mask area as fraction of image pixels; smaller masks are "
                             "discarded in wbc-mode (default: 0.02)")
    parser.add_argument("--fallback-threshold", type=float, default=0.2,
                        help="Retry at this lower threshold if 0 detections survive the main "
                             "threshold (default: 0.2). Helps EBO/MYO/LYA categories.")
    parser.add_argument("--no-fallback-threshold", dest="fallback_threshold",
                        action="store_const", const=None,
                        help="Disable the fallback threshold retry")
    parser.add_argument("--masked-output", action="store_true",
                        help="Save original image pixels inside the mask (transparent background) "
                             "instead of a visualization overlay")
    parser.add_argument("--fill-holes", action="store_true",
                        help="Fill interior holes in predicted masks (binary_fill_holes). "
                             "Helps with ring/crescent predictions on high nucleus-cytoplasm "
                             "contrast cells like erythroblasts and granulocytes.")
    parser.add_argument("--erythroid-categories", nargs="*",
                        default=["erythroid"],
                        help="Category names whose cells have erythroid cytoplasm; "
                             "RBC suppression is skipped for these to avoid destroying the "
                             "target cell itself (default: erythroid). "
                             "Pass with no arguments (--erythroid-categories) to apply "
                             "RBC suppression to erythroid cells too.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# RBC suppression pre-processing
# ---------------------------------------------------------------------------

def suppress_rbc(image_path: Path) -> Path:
    """
    Load a TIFF, replace RBC-like pixels (pink/red in HSV) with white,
    save to a temp PNG, and return its path.

    RBC heuristic (OpenCV H range 0-179):
      - Red/pink hue:  H < 18  or  H > 148
      - Meaningful saturation: S > 25  (lowered from 40 to catch pale RBCs)
      - Not very dark (avoids nuking purple WBC nucleus): V > 60
    The caller is responsible for deleting the returned temp file.
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    rbc_mask = ((h < 18) | (h > 148)) & (s > 25) & (v > 60)
    arr[rbc_mask] = [255, 255, 255]

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    PILImage.fromarray(arr).save(tmp.name)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def collect_tiff_files(data_root: Path, categories=None, cell_types=None):
    """
    Walk data_root and collect all .tiff files.

    Returns list of dicts:
        {path, category, cell_type, rel_path}

    cell_type is inferred from the first 3 uppercase characters of the filename
    (e.g. "NGS_0001.tiff" → "NGS").
    """
    records = []
    for tiff_path in sorted(data_root.rglob("*.tiff")):
        # Category is always the first level below data_root
        rel = tiff_path.relative_to(data_root)
        parts = rel.parts          # e.g. ('granulocyte_mature', 'NGS', 'NGS_0001.tiff')
                                   #   or ('lymphoid', 'LYA', 'ALL', 'LYA_0001.tiff')
        category  = parts[0]
        cell_type = tiff_path.stem[:3].upper()   # first 3 chars of filename

        if categories and category not in categories:
            continue
        if cell_types and cell_type not in cell_types:
            continue

        records.append({
            "path":      tiff_path,
            "category":  category,
            "cell_type": cell_type,
            "rel_path":  rel,           # relative path for output mirroring
        })

    return records


# ---------------------------------------------------------------------------
# Inference call
# ---------------------------------------------------------------------------

def run_medsam3_inference(medsam3_dir, config, image_path, output_path,
                          prompt, threshold, nms_iou,
                          wbc_mode=True, min_mask_area_frac=0.02,
                          fallback_threshold=None, masked_output=False,
                          fill_holes=False):
    """
    Call infer_sam.py as a subprocess.
    Returns (success: bool, stderr: str)
    """
    infer_script = os.path.join(medsam3_dir, "infer_sam.py")
    cmd = [
        sys.executable, infer_script,
        "--config",    str(config),
        "--image",     str(image_path),
        "--prompt",    prompt,
        "--threshold", str(threshold),
        "--nms-iou",   str(nms_iou),
        "--output",    str(output_path),
        "--min-mask-area-frac", str(min_mask_area_frac),
    ]
    if wbc_mode:
        cmd.append("--wbc-mode")
    if fallback_threshold is not None:
        cmd += ["--fallback-threshold", str(fallback_threshold)]
    if fill_holes:
        cmd.append("--fill-holes")
    if masked_output:
        cmd.append("--masked-output")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=medsam3_dir)
    return result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_root  = Path(args.data_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    config     = Path(args.config).expanduser()
    medsam3_dir = Path(args.medsam3_dir).expanduser()

    if not data_root.exists():
        sys.exit(f"[ERROR] data-root not found: {data_root}")
    if not config.exists():
        sys.exit(f"[ERROR] config not found: {config}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    records = collect_tiff_files(data_root, args.categories, args.cell_types)
    if not records:
        sys.exit("[ERROR] No .tiff files found with the given filters.")

    if args.max_per_type:
        from collections import defaultdict
        seen = defaultdict(int)
        filtered = []
        for r in records:
            if seen[r["cell_type"]] < args.max_per_type:
                filtered.append(r)
                seen[r["cell_type"]] += 1
        records = filtered

    if args.max_images:
        records = records[: args.max_images]

    total = len(records)
    print(f"Found {total} .tiff images to process")
    print(f"Prompt     : '{args.prompt}'")
    print(f"Threshold  : {args.threshold}")
    print(f"NMS IoU    : {args.nms_iou}")
    print(f"Suppress RBC: {args.suppress_rbc} (skipped for: {args.erythroid_categories})")
    print(f"WBC mode   : {args.wbc_mode}")
    print(f"Min mask area: {args.min_mask_area_frac:.1%}")
    print(f"Fallback threshold: {args.fallback_threshold}")
    print(f"Output dir : {output_dir}")
    print("-" * 60)

    # Count by category for info
    from collections import Counter
    cat_counts = Counter(r["category"] for r in records)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt} images")
    print("-" * 60)

    summary_rows = []
    success_cnt  = 0
    skip_cnt     = 0

    for i, rec in enumerate(records, 1):
        img_path  = rec["path"]
        category  = rec["category"]
        cell_type = rec["cell_type"]

        # Mirror sub-folder structure: replace .tiff → _mask.png
        rel_mask = rec["rel_path"].with_suffix("").name + "_mask.png"
        mask_dir  = output_dir / Path(*rec["rel_path"].parts[:-1])
        mask_dir.mkdir(parents=True, exist_ok=True)
        out_mask_path = mask_dir / rel_mask

        # Skip if already done
        if args.skip_existing and out_mask_path.exists():
            skip_cnt += 1
            summary_rows.append({
                "image":     img_path.name,
                "category":  category,
                "cell_type": cell_type,
                "status":    "SKIPPED",
                "mask_path": str(out_mask_path),
            })
            continue

        print(f"[{i:>6}/{total}] {category}/{cell_type}/{img_path.name}", end="  ", flush=True)

        # Skip RBC suppression for erythroid categories: the target cells
        # themselves have pink/erythroid cytoplasm and would be destroyed.
        is_erythroid = category in (args.erythroid_categories or [])
        do_suppress = args.suppress_rbc and not is_erythroid

        tmp_path = None
        if do_suppress:
            tmp_path = suppress_rbc(img_path)
            infer_input = tmp_path
        else:
            infer_input = img_path

        try:
            ok, err = run_medsam3_inference(
                medsam3_dir        = str(medsam3_dir),
                config             = str(config),
                image_path         = str(infer_input),
                output_path        = str(out_mask_path),
                prompt             = args.prompt,
                threshold          = args.threshold,
                nms_iou            = args.nms_iou,
                wbc_mode           = args.wbc_mode,
                min_mask_area_frac = args.min_mask_area_frac,
                fallback_threshold = args.fallback_threshold,
                masked_output      = args.masked_output,
                fill_holes         = args.fill_holes,
            )
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

        if ok:
            success_cnt += 1
            print(f"OK -> {out_mask_path.name}")
        else:
            print(f"FAIL: {err.strip()[-200:]}")

        summary_rows.append({
            "image":     img_path.name,
            "category":  category,
            "cell_type": cell_type,
            "status":    "OK" if ok else "FAIL",
            "mask_path": str(out_mask_path) if ok else "",
        })

    # Save summary CSV
    csv_path = output_dir / "inference_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "category", "cell_type", "status", "mask_path"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    processed = total - skip_cnt
    print("-" * 60)
    print(f"Done!")
    print(f"  Processed  : {processed}  (success: {success_cnt}, fail: {processed - success_cnt})")
    print(f"  Skipped    : {skip_cnt}")
    print(f"  Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
