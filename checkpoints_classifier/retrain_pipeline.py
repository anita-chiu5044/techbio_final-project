"""
Robust ConvNet retraining orchestrator.

This replaces the fragile shell-only flow with a Python entry point that:
  1. optionally waits for the MedSAM process to finish,
  2. validates and summarizes inference_summary.csv,
  3. writes class/status and mask-QC reports,
  4. runs selected train.py configs sequentially,
  5. generates compare_runs.py output.

Example:
  python retrain_pipeline.py \
      --medsam-pid 307432 \
      --summary-csv /home/yucheng/Desktop/techbio_pipeline_output/medsam_output/inference_summary.csv \
      --runs-dir /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

DEFAULT_SUMMARY_CSV = Path(
    "/home/yucheng/Desktop/techbio_pipeline_output/medsam_output/inference_summary.csv"
)
DEFAULT_RUNS_DIR = Path("/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs")
DEFAULT_CONFIGS = ["ce_uniform", "focal_wrs", "focal_wrs_stage2", "focal_wrs_effv2"]
IMBALANCE_CONFIGS = [
    "ce_capped2000",
    "cb_ce_capped2000",
    "balanced_softmax_capped2000",
    "focal_capped2000",
]
REQUIRED_COLUMNS = {"image", "category", "cell_type", "status", "num_detections", "mask_path"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifier retraining experiments safely.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--medsam-pid", type=int, default=None)
    parser.add_argument("--wait-interval", type=int, default=60)
    parser.add_argument("--conda-env", default="techbio")
    parser.add_argument("--python-executable", default=None,
                        help="Use this Python executable instead of conda run -n ENV python.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--stage1-epochs-override", type=int, default=None,
                        help="Forwarded to train.py for short smoke runs.")
    parser.add_argument("--stage2-epochs-override", type=int, default=None,
                        help="Forwarded to train.py for short smoke runs.")
    parser.add_argument("--force", action="store_true",
                        help="Rerun a config even if metrics.json already exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-ok-fraction", type=float, default=0.50,
                        help="Fail before training if OK rows are below this fraction.")
    parser.add_argument("--mask-qc-limit", type=int, default=0,
                        help="0 means QC all OK masks; useful to lower for quick smoke checks.")
    return parser.parse_args()


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int, interval: int) -> None:
    print(f"Waiting for MedSAM PID {pid} ...", flush=True)
    while is_pid_running(pid):
        print(f"  PID {pid} still running; sleeping {interval}s", flush=True)
        time.sleep(interval)
    print(f"MedSAM PID {pid} finished.", flush=True)


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"summary CSV not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary CSV missing columns: {sorted(missing)}")
        return list(reader)


def summarize_rows(rows: Iterable[dict[str, str]], output_dir: Path) -> dict[str, object]:
    status_counts: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    ok_by_class: Counter[str] = Counter()

    total = 0
    for row in rows:
        total += 1
        status = row.get("status", "").upper()
        category = row.get("category", "")
        mask_path = Path(row.get("mask_path", ""))
        if row.get("mask_path") and mask_path.parent.name:
            cell_type = mask_path.parent.name
            if mask_path.parent.parent.name:
                category = mask_path.parent.parent.name
        else:
            cell_type = row.get("cell_type", "")
        status_counts[status] += 1
        by_class[cell_type][status] += 1
        by_category[category][status] += 1
        if status == "OK":
            ok_by_class[cell_type] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    class_csv = output_dir / "medsam_status_by_class.csv"
    all_statuses = sorted(status_counts)
    with class_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_type", "total", *all_statuses])
        for cell_type in sorted(by_class):
            counts = by_class[cell_type]
            writer.writerow([cell_type, sum(counts.values()), *[counts[s] for s in all_statuses]])

    payload = {
        "total_rows": total,
        "status_counts": dict(status_counts),
        "ok_fraction": (status_counts.get("OK", 0) / total) if total else 0.0,
        "ok_by_class": dict(sorted(ok_by_class.items())),
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "class_status_csv": str(class_csv),
    }
    (output_dir / "preflight_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def foreground_mask_stats(mask_path: Path) -> dict[str, object]:
    with Image.open(mask_path) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        pixels = rgba.load()
        total = width * height
        foreground = 0
        edge_foreground = 0
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                is_fg = a > 0 and (r < 250 or g < 250 or b < 250)
                if is_fg:
                    foreground += 1
                    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                        edge_foreground += 1
    area_ratio = foreground / total if total else 0.0
    edge_ratio = edge_foreground / foreground if foreground else 0.0
    return {
        "width": width,
        "height": height,
        "foreground_pixels": foreground,
        "area_ratio": round(area_ratio, 6),
        "edge_ratio": round(edge_ratio, 6),
    }


def run_mask_qc(rows: list[dict[str, str]], output_dir: Path, limit: int) -> dict[str, object]:
    qc_path = output_dir / "mask_qc.csv"
    suspicious = 0
    checked = 0
    with qc_path.open("w", newline="") as handle:
        fieldnames = [
            "image", "category", "cell_type", "mask_path", "width", "height",
            "foreground_pixels", "area_ratio", "edge_ratio", "qc_flag", "qc_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("status", "").upper() != "OK":
                continue
            if limit and checked >= limit:
                break
            mask_path = Path(row.get("mask_path", ""))
            checked += 1
            reason = []
            try:
                stats = foreground_mask_stats(mask_path)
                area = float(stats["area_ratio"])
                edge = float(stats["edge_ratio"])
                if area < 0.02:
                    reason.append("tiny_foreground")
                if area > 0.90:
                    reason.append("huge_foreground")
                if edge > 0.15:
                    reason.append("edge_touching")
            except Exception as exc:  # noqa: BLE001 - QC should record bad files, not hide them.
                stats = {"width": "", "height": "", "foreground_pixels": "", "area_ratio": "", "edge_ratio": ""}
                reason.append(f"open_error:{exc}")
            if reason:
                suspicious += 1
            writer.writerow({
                "image": row.get("image", ""),
                "category": Path(row.get("mask_path", "")).parent.parent.name or row.get("category", ""),
                "cell_type": Path(row.get("mask_path", "")).parent.name or row.get("cell_type", ""),
                "mask_path": str(mask_path),
                **stats,
                "qc_flag": "suspicious" if reason else "ok",
                "qc_reason": ";".join(reason),
            })
    payload = {"checked": checked, "suspicious": suspicious, "mask_qc_csv": str(qc_path)}
    (output_dir / "mask_qc_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def python_cmd(args: argparse.Namespace) -> list[str]:
    if args.python_executable:
        return [args.python_executable]
    return ["conda", "run", "-n", args.conda_env, "python"]


def tee_subprocess(cmd: list[str], log_path: Path, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def run_training(args: argparse.Namespace, script_dir: Path) -> None:
    base_py = python_cmd(args)
    train_script = script_dir / "train.py"
    compare_script = script_dir / "compare_runs.py"
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    for config in args.configs:
        out_dir = args.runs_dir / config
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"Skipping {config}; metrics.json already exists.", flush=True)
            continue
        cmd = [
            *base_py,
            str(train_script),
            "--config", config,
            "--summary-csv", str(args.summary_csv),
            "--output-dir", str(out_dir),
            "--val-fraction", str(args.val_fraction),
            "--image-size", str(args.image_size),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
        ]
        if args.no_amp:
            cmd.append("--no-amp")
        if args.stage1_epochs_override is not None:
            cmd.extend(["--stage1-epochs-override", str(args.stage1_epochs_override)])
        if args.stage2_epochs_override is not None:
            cmd.extend(["--stage2-epochs-override", str(args.stage2_epochs_override)])
        tee_subprocess(cmd, args.runs_dir / f"{config}.log", args.dry_run)

    compare_cmd = [
        *base_py,
        str(compare_script),
        "--runs-dir", str(args.runs_dir),
        "--output", str(args.runs_dir / "comparison_report.txt"),
    ]
    tee_subprocess(compare_cmd, args.runs_dir / "compare_runs.log", args.dry_run)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.medsam_pid is not None and is_pid_running(args.medsam_pid):
        wait_for_pid(args.medsam_pid, args.wait_interval)

    rows = read_summary(args.summary_csv)
    summary = summarize_rows(rows, args.runs_dir)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["ok_fraction"] < args.min_ok_fraction:
        raise RuntimeError(
            f"OK fraction {summary['ok_fraction']:.3f} is below minimum {args.min_ok_fraction:.3f}"
        )

    qc_summary = run_mask_qc(rows, args.runs_dir, args.mask_qc_limit)
    print(json.dumps(qc_summary, indent=2), flush=True)

    run_training(args, script_dir)


if __name__ == "__main__":
    main()
