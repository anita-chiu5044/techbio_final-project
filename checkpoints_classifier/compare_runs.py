"""
Compare ConvNet experiment results.

Usage:
    python compare_runs.py --runs-dir /path/to/convnet_runs
"""

import argparse
import json
from pathlib import Path


def load_metrics(runs_dir: Path) -> list[dict]:
    results = []
    for metrics_file in sorted(runs_dir.glob("*/metrics.json")):
        with metrics_file.open() as fh:
            data = json.load(fh)
        config_name = metrics_file.parent.name
        final = data.get("final", {})
        results.append({
            "config":       config_name,
            "macro_f1":     final.get("macro_f1", 0),
            "balanced_acc": final.get("balanced_acc", 0),
            "overall_acc":  final.get("overall_acc", 0),
            "tail_recall":  final.get("tail_recall", {}),
            "per_class":    final.get("per_class", {}),
            "confusion_matrix": final.get("confusion_matrix", []),
        })
    return results


def print_comparison(results: list[dict]) -> None:
    if not results:
        print("No metrics.json files found.")
        return

    results_sorted = sorted(results, key=lambda r: r["macro_f1"], reverse=True)

    # ---- Summary table ----
    header = f"{'Config':<22}  {'macro_F1':>8}  {'bal_acc':>8}  {'ovr_acc':>8}"
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    print(header)
    print("-" * 60)
    for r in results_sorted:
        marker = " ← BEST" if r is results_sorted[0] else ""
        print(f"{r['config']:<22}  {r['macro_f1']:>8.4f}  "
              f"{r['balanced_acc']:>8.4f}  {r['overall_acc']:>8.4f}{marker}")

    # ---- Tail class recall ----
    tail_classes = ["PMO", "MYB", "MOB", "PMB", "KSC", "MMZ"]
    print("\n" + "-" * 60)
    print("TAIL CLASS RECALL (target ≥ 0.65)")
    print("-" * 60)
    hdr = f"{'Config':<22} " + "".join(f"{c:>6}" for c in tail_classes)
    print(hdr)
    for r in results_sorted:
        row = f"{r['config']:<22} "
        for c in tail_classes:
            val = r["tail_recall"].get(c, None)
            if val is None:
                row += f"{'N/A':>6}"
            else:
                row += f"{val:>6.3f}"
        print(row)

    # ---- Best config per-class F1 ----
    best = results_sorted[0]
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best['config']}  (macro_F1={best['macro_f1']:.4f})")
    print(f"{'='*60}")
    print(f"{'Class':<8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 40)
    for cls, vals in sorted(best["per_class"].items()):
        if isinstance(vals, dict):
            print(f"{cls:<8} {vals.get('p', 0):>10.3f} {vals.get('r', 0):>8.3f} {vals.get('f1', 0):>8.3f}")

    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--output",   type=Path, default=None,
                   help="Optional: also write results to this text file")
    args = p.parse_args()

    results = load_metrics(args.runs_dir)

    if args.output:
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_comparison(results)
        sys.stdout = old_stdout
        text = buf.getvalue()
        print(text, end="")  # single print to terminal
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Results also saved to: {args.output}")
    else:
        print_comparison(results)


if __name__ == "__main__":
    main()
