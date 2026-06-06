"""
Multi-seed split variance evaluation.

Loads an existing checkpoint (no retraining) and evaluates it on N different
val splits from the same training data. Reports mean ± std of key metrics to
characterise how much results depend on the random train/val split.

This is fast (evaluation only, no training) and is the correct way to check
whether a macro_F1 number is reliable given tiny tail-class val sets.

Usage:
    python eval_seeds.py \
        --ckpt  /path/to/best.pth \
        --summary-csv /path/to/inference_summary.csv \
        --num-seeds 5 \
        --val-fraction 0.15

    # Compare multiple checkpoints:
    python eval_seeds.py \
        --ckpt  run_a/best.pth run_b/best.pth \
        --summary-csv /path/to/inference_summary.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import (
    WBCDataset,
    _class_counts,
    evaluate,
    get_val_transform,
    stratified_split,
    build_convnext,
    build_resnet,
    build_efficientnet_v2,
    build_dinobloom,
    SimpleCNN,
)
from torch.utils.data import DataLoader


TAIL_CLASSES = {"PMO", "MYB", "MOB", "PMB", "KSC", "MMZ"}


def load_checkpoint(ckpt_path: Path, device: torch.device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device)

    class_names: list[str] = ckpt["class_names"]
    n = len(class_names)
    model_name: str = ckpt.get("args", {}).get("model", "convnext")

    if model_name == "dinobloom":
        model = build_dinobloom(n, ckpt_path=None, head_type="mlp")
    elif model_name == "resnet101":
        model = build_resnet(n, variant=101, pretrained=False)
    elif model_name == "resnet50":
        model = build_resnet(n, variant=50, pretrained=False)
    elif model_name == "efficientv2":
        model = build_efficientnet_v2(n, pretrained=False)
    elif model_name == "cnn":
        model = SimpleCNN(n)
    else:
        model = build_convnext(n, pretrained=False)

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    class_freq = ckpt.get("class_freq")
    freq = np.array(class_freq, dtype=np.float32) if class_freq else None
    return model, class_names, freq


def eval_one_seed(model, full_ds, class_names, freq, val_fraction, seed,
                  batch_size, num_workers, device, use_logit_adj):
    n_classes = len(class_names)
    _, val_samples = stratified_split(full_ds.samples, val_fraction, seed)

    val_ds = WBCDataset.__new__(WBCDataset)
    val_ds.class_to_idx = full_ds.class_to_idx
    val_ds.samples = val_samples
    val_ds.transform = get_val_transform(224)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    eval_freq = freq if use_logit_adj else None
    metrics = evaluate(model, val_loader, device, class_names, freq=eval_freq)

    # Per-class val support
    val_counts = _class_counts(val_ds, n_classes)
    metrics["val_support"] = dict(zip(class_names, val_counts))
    return metrics


def aggregate(results: list[dict], class_names: list[str]) -> dict:
    keys = ["macro_f1", "balanced_acc", "overall_acc"]
    agg: dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = {"mean": round(float(np.mean(vals)), 4),
                  "std":  round(float(np.std(vals)),  4),
                  "min":  round(float(np.min(vals)),  4),
                  "max":  round(float(np.max(vals)),  4)}

    tail = {}
    for cls in class_names:
        if cls in TAIL_CLASSES:
            vals = [r.get("tail_recall", {}).get(cls, float("nan")) for r in results]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                tail[cls] = {"mean": round(float(np.mean(vals)), 3),
                             "std":  round(float(np.std(vals)),  3)}
    agg["tail_recall"] = tail

    # avg val support per class
    support: dict[str, list[int]] = {c: [] for c in class_names}
    for r in results:
        for c, n in r.get("val_support", {}).items():
            support[c].append(n)
    agg["avg_val_support"] = {c: round(float(np.mean(v)), 1) for c, v in support.items() if v}
    return agg


def print_report(ckpt_label: str, agg: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_label}")
    print(f"{'='*60}")
    print(f"{'Metric':<16} {'mean':>8} {'±std':>8} {'min':>8} {'max':>8}")
    print("-" * 52)
    for k in ["macro_f1", "balanced_acc", "overall_acc"]:
        d = agg[k]
        print(f"{k:<16} {d['mean']:>8.4f} {d['std']:>8.4f} {d['min']:>8.4f} {d['max']:>8.4f}")

    print(f"\nTail class recall (mean ± std across splits):")
    print(f"  {'Class':<6}  {'mean':>6}  {'±std':>6}  avg_val_n")
    for cls, d in sorted(agg["tail_recall"].items()):
        n = agg["avg_val_support"].get(cls, "?")
        flag = "  ✓" if d["mean"] >= 0.65 else "  ✗"
        print(f"  {cls:<6}  {d['mean']:>6.3f}  {d['std']:>6.3f}  {n}{flag}")

    print(f"\nNote: avg_val_n = avg validation samples per class across seeds.")
    print(f"      Low n (< 5) means recall is very noisy — do not over-interpret.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, type=Path,
                   help="One or more checkpoint paths to evaluate")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--num-seeds", type=int, default=5,
                   help="Number of different val splits to evaluate (default: 5)")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-logit-adj", action="store_true",
                   help="Disable logit adjustment even if checkpoint has class_freq")
    p.add_argument("--output", type=Path, default=None,
                   help="Save results JSON to this path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seeds: {args.base_seed}..{args.base_seed + args.num_seeds - 1}")

    all_results: dict[str, dict] = {}

    for ckpt_path in args.ckpt:
        print(f"\nLoading: {ckpt_path}")
        model, class_names, freq = load_checkpoint(ckpt_path, device)
        use_logit_adj = (not args.no_logit_adj) and (freq is not None)
        print(f"  classes={len(class_names)}  logit_adj={use_logit_adj}")

        # Build full dataset once (val_transform; only split indices matter)
        full_ds = WBCDataset(args.summary_csv, class_names,
                             transform=get_val_transform(224))
        print(f"  total samples: {len(full_ds)}")

        seed_metrics: list[dict] = []
        for i in range(args.num_seeds):
            seed = args.base_seed + i
            m = eval_one_seed(model, full_ds, class_names, freq,
                              args.val_fraction, seed,
                              args.batch_size, args.num_workers,
                              device, use_logit_adj)
            seed_metrics.append(m)
            print(f"  seed {seed}: macro_f1={m['macro_f1']:.4f}  "
                  f"bal_acc={m['balanced_acc']:.4f}")

        agg = aggregate(seed_metrics, class_names)
        label = str(ckpt_path)
        print_report(label, agg)
        all_results[label] = agg

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
