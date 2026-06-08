"""
Run multiple classifier configs on task_combine dataset and log results.

Split strategy (stratified):
  - test  : 15% of total, held out completely
  - train : 72% of total  ─┐ train.py receives this combined
  - val   : 13% of total  ─┘ CSV and does its own val split

Runs each config sequentially, then writes results_summary.txt and
copies the best checkpoint to the canonical path.

Usage:
    cd techbio_final-project/checkpoints_classifier
    python run_all_configs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
SUMMARY_CSV = Path(__file__).parent / "task_combine_medsam_summary.csv"
RUNS_DIR = Path(
    "/home/yucheng/Desktop/techbio/artifacts/checkpoints/convnet/task_combine_all_configs"
)
BEST_CKPT_OUT = Path(
    "/home/yucheng/Desktop/techbio/artifacts/checkpoints/convnet/task_combine_dinobloom/best.pth"
)
PYTHON = sys.executable

# Configs to benchmark
CONFIGS = [
    "dinobloom_ce_uniform",        # DinoBloom-B + CE uniform [best config from prev run]
    "dinobloom_l_ce_uniform",      # DinoBloom-L + CE uniform [L vs B comparison]
]

TEST_FRACTION = 0.15   # held-out test set (never seen by train.py)
# train.py gets (1-TEST_FRACTION) of data and internally splits val at 15%
# → effective: train≈72%, val≈13%, test≈15% of total
INTERNAL_VAL_FRACTION = 0.15
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"].upper() == "OK"]


def stratified_split_two(rows: list[dict], test_frac: float, seed: int):
    """Split into (train_val, test) with class-stratified test holdout."""
    rng = random.Random(seed)
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r["cell_type"], []).append(r)
    train_val_rows, test_rows = [], []
    for cls, items in sorted(by_class.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_frac))
        test_rows      += shuffled[:n_test]
        train_val_rows += shuffled[n_test:]
    return train_val_rows, test_rows


def write_split_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["status", "mask_path", "cell_type"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def log_split_counts(
    train_val: list[dict], test: list[dict], val_frac: float
) -> None:
    # Approximate train/val breakdown (same logic as train.py's stratified_split)
    tv_count = Counter(r["cell_type"] for r in train_val)
    test_count = Counter(r["cell_type"] for r in test)
    classes = sorted(tv_count.keys() | test_count.keys())

    print("\n" + "=" * 68)
    print(f"{'Class':<25} {'Train~':>8} {'Val~':>8} {'Test':>8} {'Total':>8}")
    print("-" * 68)
    grand_tr, grand_v, grand_t = 0, 0, 0
    for cls in classes:
        tv = tv_count[cls]
        n_val  = max(1, int(tv * val_frac))
        n_train = tv - n_val
        t = test_count[cls]
        print(f"  {cls:<23} {n_train:>8} {n_val:>8} {t:>8} {n_train+n_val+t:>8}")
        grand_tr += n_train; grand_v += n_val; grand_t += t
    print("-" * 68)
    total = grand_tr + grand_v + grand_t
    print(f"  {'TOTAL':<23} {grand_tr:>8} {grand_v:>8} {grand_t:>8} {total:>8}")
    print("=" * 68 + "\n")


def run_config(config: str, train_val_csv: Path, dry_run: bool) -> dict:
    out_dir = RUNS_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "train.py",
        "--config", config,
        "--summary-csv", str(train_val_csv),
        "--val-fraction", str(INTERNAL_VAL_FRACTION),
        "--output-dir", str(out_dir),
    ]
    print(f"\n{'='*60}\nRunning config: {config}\n{'='*60}")
    if dry_run:
        print("  [dry-run] would run:", " ".join(cmd))
        return {"config": config, "returncode": 0, "elapsed_min": 0.0,
                "val_macro_f1": None, "val_bal_acc": None, "val_acc": None}
    t0 = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    result = {
        "config": config,
        "returncode": proc.returncode,
        "elapsed_min": round(elapsed / 60, 1),
        "val_macro_f1": metrics.get("macro_f1"),
        "val_bal_acc":  metrics.get("balanced_acc"),
        "val_acc":      metrics.get("overall_acc"),
    }
    print(f"  → macro_f1={result['val_macro_f1']}  bal_acc={result['val_bal_acc']}"
          f"  ({result['elapsed_min']} min)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running training")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = load_csv(SUMMARY_CSV)
    print(f"Total rows loaded: {len(all_rows)}")

    train_val_rows, test_rows = stratified_split_two(all_rows, TEST_FRACTION, SEED)
    log_split_counts(train_val_rows, test_rows, INTERNAL_VAL_FRACTION)

    # Write split CSVs
    train_val_csv = RUNS_DIR / "split_train_val.csv"
    test_csv      = RUNS_DIR / "split_test.csv"
    write_split_csv(train_val_rows, train_val_csv)
    write_split_csv(test_rows,      test_csv)
    print(f"Split CSVs written to {RUNS_DIR}/")

    # Run all configs
    results = []
    for cfg in CONFIGS:
        r = run_config(cfg, train_val_csv, dry_run=args.dry_run)
        results.append(r)

    # Summary table
    summary_path = RUNS_DIR / "results_summary.txt"
    lines = [
        "=" * 70,
        f"{'Config':<35} {'macro_F1':>9} {'bal_acc':>9} {'acc':>7} {'min':>6}",
        "-" * 70,
    ]
    best_f1, best_cfg = -1.0, ""
    for r in results:
        f1 = r["val_macro_f1"] or 0.0
        lines.append(
            f"  {r['config']:<33} {f1:>9.4f}"
            f" {(r['val_bal_acc'] or 0):>9.4f}"
            f" {(r['val_acc'] or 0):>7.4f}"
            f" {r['elapsed_min']:>6}"
        )
        if f1 > best_f1:
            best_f1, best_cfg = f1, r["config"]
    lines += ["=" * 70, f"Best config: {best_cfg}  (macro_F1={best_f1:.4f})"]
    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text)
    print("\n" + summary_text)

    # Copy best checkpoint to canonical path
    if not args.dry_run and best_cfg:
        best_ckpt = RUNS_DIR / best_cfg / "best.pth"
        if best_ckpt.exists():
            BEST_CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_ckpt, BEST_CKPT_OUT)
            print(f"\nBest checkpoint ({best_cfg}) copied to {BEST_CKPT_OUT}")


if __name__ == "__main__":
    main()
