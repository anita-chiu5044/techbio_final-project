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
    """Read inference_summary.csv and write final training CSV with YOLO confidence."""
    print("\n=== Step 5: Collect MedSAM outputs into training CSV ===")
    summary_csv = MEDSAM_OUTPUT / "inference_summary.csv"
    if dry_run:
        print(f"  [dry-run] would read {summary_csv} → {FINAL_CSV}")
        return
    if not summary_csv.exists():
        print(f"  ERROR: {summary_csv} not found. MedSAM may have failed.")
        return

    # Build detection_id → (confidence, class_label) from YOLO detections
    import json as _json
    det_info: dict[str, tuple[float, str]] = {}
    jsonl = YOLO_OUT / "detections.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                d = _json.loads(line)
                det_info[d["detection_id"]] = (d.get("confidence", 0.0), d.get("class_label", ""))

    # Build mask stem → detection_id from cell_map
    # cell_map image column: "class/class/STEM_mask.png"
    stem_to_det: dict[str, str] = {}
    if CELL_MAP_CSV.exists():
        with CELL_MAP_CSV.open() as f:
            for row in csv.DictReader(f):
                stem = Path(row["image"]).stem  # e.g. "PMB_0001_det_000006_mask" → strip _mask below
                det_id = row.get("detection_id") or row.get("cell_id", "")
                stem_to_det[stem] = det_id

    def _get_conf(image_name: str) -> tuple[float, str]:
        """image_name is the tiff patch name, e.g. PMB_0001_det_000006.tiff"""
        stem = Path(image_name).stem  # PMB_0001_det_000006
        mask_stem = stem + "_mask"    # PMB_0001_det_000006_mask
        det_id = stem_to_det.get(mask_stem, "")
        if det_id and det_id in det_info:
            return det_info[det_id]
        # fallback: extract det_id from stem suffix
        if "_det_" in stem:
            det_id = "det_" + stem.split("_det_")[-1]
            if det_id in det_info:
                return det_info[det_id]
        return (0.0, "")

    rows_ok: list[dict] = []
    rows_fail = 0
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"].upper() == "OK" and row.get("mask_path"):
                mask_path = Path(row["mask_path"])
                cell_type = mask_path.parent.name
                conf, cls_label = _get_conf(row["image"])
                rows_ok.append({
                    "status": "OK",
                    "mask_path": str(mask_path),
                    "cell_type": cell_type,
                    "yolo_confidence": round(conf, 6),
                    "yolo_class_label": cls_label,
                })
            else:
                rows_fail += 1

    with FINAL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["status", "mask_path", "cell_type", "yolo_confidence", "yolo_class_label"])
        w.writeheader()
        w.writerows(rows_ok)

    # ── Per-class analysis ────────────────────────────────────────────────────
    import json as _json2
    import statistics as _stats
    from collections import Counter, defaultdict

    # Also load ALL summary rows (including failed) for MedSAM quality per class
    all_rows_by_class: dict[str, list[str]] = defaultdict(list)
    ok_rows_by_class:  dict[str, list[str]] = defaultdict(list)
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            cls = row.get("cell_type", "unknown")
            all_rows_by_class[cls].append(row["status"])
            if row["status"].upper() == "OK":
                ok_rows_by_class[cls].append(row["status"])

    conf_by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows_ok:
        conf_by_class[r["cell_type"]].append(r["yolo_confidence"])

    analysis: list[dict] = []
    all_classes = sorted(set(list(all_rows_by_class) + list(conf_by_class)))
    print(f"\nFinal training CSV: {FINAL_CSV}")
    print(f"  OK rows: {len(rows_ok)}  |  Failed/no-detection: {rows_fail}")
    print(f"\n{'Class':<25} {'N_ok':>5} {'N_total':>7} {'MedSAM_ok%':>10} {'conf_mean':>9} {'conf_min':>8} {'conf_max':>8} {'conf_med':>8}")
    print("-" * 90)
    for cls in all_classes:
        confs = conf_by_class.get(cls, [])
        n_ok = len(ok_rows_by_class.get(cls, []))
        n_total = len(all_rows_by_class.get(cls, []))
        medsam_pct = 100 * n_ok / n_total if n_total else 0
        c_mean = _stats.mean(confs) if confs else 0.0
        c_min  = min(confs) if confs else 0.0
        c_max  = max(confs) if confs else 0.0
        c_med  = _stats.median(confs) if confs else 0.0
        print(f"  {cls:<23} {n_ok:>5} {n_total:>7} {medsam_pct:>9.1f}% {c_mean:>9.3f} {c_min:>8.3f} {c_max:>8.3f} {c_med:>8.3f}")
        analysis.append({
            "class": cls, "n_ok": n_ok, "n_total": n_total,
            "medsam_ok_pct": round(medsam_pct, 2),
            "conf_mean": round(c_mean, 4), "conf_min": round(c_min, 4),
            "conf_max": round(c_max, 4), "conf_median": round(c_med, 4),
        })

    analysis_path = FINAL_CSV.parent / "task_combine_class_analysis.json"
    analysis_path.write_text(_json2.dumps({"total_ok": len(rows_ok), "classes": analysis}, indent=2))
    print(f"\nPer-class analysis saved: {analysis_path}")


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
