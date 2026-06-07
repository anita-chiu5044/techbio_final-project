"""
Preprocess task_combine dataset through YOLO → top-1 → MedSAM to produce
properly segmented cell images for classifier training.

Steps:
  1. Create a remapped input directory where images sit under
     {class_name}/{class_name}/ so that yolo_to_medsam_patches.py assigns
     the correct cell_type label (= class name), which train.py reads from
     mask_path.parent.name.
  2. Run YOLO (export_yolo_detection_manifest.py) on the remapped structure.
  3. Run yolo_to_medsam_patches.py --selection top1 to get one patch per image.
  4. Run MedSAM (tiff_wbc_inference.py) on the patches.
  5. Collect inference_summary.csv (only OK rows) as the final training CSV.

Usage:
    cd techbio_final-project/checkpoints_classifier
    python preprocess_task_combine.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]          # techbio_final-project/
DATA_ROOT = Path(
    "/mnt2/anita/TechBio/classified/PKG - AML-Cytomorphology_LMU/for_fang/task_combine"
)
WORK_DIR = Path(
    "/home/yucheng/Desktop/techbio/artifacts/task_combine_preprocess"
)
REMAP_DIR      = WORK_DIR / "remapped_input"    # {class}/{class}/*.img
YOLO_OUT       = WORK_DIR / "yolo_output"
MEDSAM_INPUT   = WORK_DIR / "medsam_input"
MEDSAM_OUTPUT  = WORK_DIR / "medsam_output"
CELL_MAP_CSV   = WORK_DIR / "cell_map.csv"
FINAL_CSV      = Path(__file__).parent / "task_combine_medsam_summary.csv"

YOLO_PYTHON     = "/home/yucheng/miniconda3/envs/AICUP/bin/python"
MEDSAM_PYTHON   = "/home/yucheng/miniconda3/envs/techbio/bin/python"
YOLO_MODEL      = str(REPO_ROOT / "best.pt")
MEDSAM_CONFIG   = str(REPO_ROOT / "MedSAM3" / "configs" / "lisc_lora_config.yaml")
MEDSAM3_DIR     = str(REPO_ROOT / "MedSAM3")

YOLO_CONF = "0.25"
YOLO_IMGSZ = "640"
YOLO_DEVICE = "0"
YOLO_BATCH = "32"
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd: list, dry_run: bool, cwd: Path | None = None) -> int:
    cmd_s = " ".join(str(c) for c in cmd)
    print(f"\n$ {cmd_s}")
    if dry_run:
        return 0
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd)
    return proc.returncode


def step1_remap(dry_run: bool) -> None:
    """Create symlinked {class}/{class}/ directory structure."""
    print("\n=== Step 1: Create remapped directory structure ===")
    for cls_dir in sorted(DATA_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        target = REMAP_DIR / cls_dir.name / cls_dir.name
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
        images = (
            sorted(cls_dir.glob("*.jpg"))
            + sorted(cls_dir.glob("*.png"))
            + sorted(cls_dir.glob("*.tiff"))
            + sorted(cls_dir.glob("*.tif"))
        )
        linked = 0
        for img in images:
            link = target / img.name
            if not dry_run and not link.exists():
                link.symlink_to(img.resolve())
            linked += 1
        print(f"  {cls_dir.name}: {linked} images → {target}")


def step2_yolo(dry_run: bool) -> int:
    """Run YOLO on the remapped directory."""
    print("\n=== Step 2: YOLO detection ===")
    cmd = [
        YOLO_PYTHON,
        REPO_ROOT / "export_yolo_detection_manifest.py",
        "--dataset-root", REMAP_DIR,
        "--output-root", YOLO_OUT,
        "--model-path", YOLO_MODEL,
        "--conf", YOLO_CONF,
        "--imgsz", YOLO_IMGSZ,
        "--device", YOLO_DEVICE,
        "--batch-size", YOLO_BATCH,
        "--case-id-mode", "none",
        "--save-patches",
        "--per-image-json",
    ]
    return run(cmd, dry_run)


def step3_roi(dry_run: bool) -> int:
    """Run yolo_to_medsam_patches.py with top-1 selection."""
    print("\n=== Step 3: ROI extraction (top-1 per image) ===")
    detections = YOLO_OUT / "detections.jsonl"
    cmd = [
        sys.executable,
        REPO_ROOT / "yolo_to_medsam_patches.py",
        "--detections", detections,
        "--output-root", MEDSAM_INPUT,
        "--context-scale", "1.3",
        "--selection", "top1",
        "--mapping-csv", CELL_MAP_CSV,
    ]
    return run(cmd, dry_run)


def step4_medsam(dry_run: bool) -> int:
    """Run MedSAM on the ROI patches."""
    print("\n=== Step 4: MedSAM segmentation ===")
    cmd = [
        MEDSAM_PYTHON,
        REPO_ROOT / "MedSAM3" / "tiff_wbc_inference.py",
        "--data-root", MEDSAM_INPUT,
        "--config", MEDSAM_CONFIG,
        "--output-dir", MEDSAM_OUTPUT,
        "--medsam3-dir", MEDSAM3_DIR,
        "--threshold", "0.5",
        "--nms-iou", "0.5",
        "--masked-output",
        "--fill-holes",
        "--erythroid-categories",
        "--skip-existing",
    ]
    return run(cmd, dry_run, cwd=Path(MEDSAM3_DIR))


def step5_collect_csv(dry_run: bool) -> None:
    """Read inference_summary.csv and write final training CSV."""
    print("\n=== Step 5: Collect MedSAM outputs into training CSV ===")
    summary_csv = MEDSAM_OUTPUT / "inference_summary.csv"
    if dry_run:
        print(f"  [dry-run] would read {summary_csv} → {FINAL_CSV}")
        return
    if not summary_csv.exists():
        print(f"  ERROR: {summary_csv} not found. MedSAM may have failed.")
        return

    rows_ok: list[dict] = []
    rows_fail = 0
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"].upper() == "OK" and row.get("mask_path"):
                mask_path = Path(row["mask_path"])
                # cell_type from mask_path.parent.name — this is the class name
                cell_type = mask_path.parent.name
                rows_ok.append({
                    "status": "OK",
                    "mask_path": str(mask_path),
                    "cell_type": cell_type,
                })
            else:
                rows_fail += 1

    with FINAL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["status", "mask_path", "cell_type"])
        w.writeheader()
        w.writerows(rows_ok)

    from collections import Counter
    counts = Counter(r["cell_type"] for r in rows_ok)
    print(f"\nFinal training CSV: {FINAL_CSV}")
    print(f"  OK rows: {len(rows_ok)}  |  Failed/skipped: {rows_fail}")
    print("  Class counts:")
    for cls, n in sorted(counts.items()):
        print(f"    {cls:<25}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-yolo", action="store_true",
                        help="Skip YOLO step (reuse existing output)")
    parser.add_argument("--skip-medsam", action="store_true",
                        help="Skip MedSAM step (reuse existing output)")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    step1_remap(args.dry_run)

    if not args.skip_yolo:
        rc = step2_yolo(args.dry_run)
        if rc != 0:
            print(f"\nERROR: YOLO failed (exit {rc}). Stopping.")
            sys.exit(rc)

    rc = step3_roi(args.dry_run)
    if rc != 0:
        print(f"\nERROR: ROI extraction failed (exit {rc}). Stopping.")
        sys.exit(rc)

    if not args.skip_medsam:
        rc = step4_medsam(args.dry_run)
        if rc != 0:
            print(f"\nERROR: MedSAM failed (exit {rc}). Stopping.")
            sys.exit(rc)

    step5_collect_csv(args.dry_run)
    print("\nPreprocessing done. Run classifier training with:")
    print(f"  python run_all_configs.py")
    print(f"  (after updating SUMMARY_CSV to {FINAL_CSV})")


if __name__ == "__main__":
    main()
