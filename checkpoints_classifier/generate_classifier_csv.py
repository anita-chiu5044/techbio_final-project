"""Generate summary_csv from task_combine folder structure for train.py.

Usage:
    cd techbio_final-project/checkpoints_classifier
    python generate_classifier_csv.py

Output: task_combine_summary.csv  (columns: status, mask_path, cell_type)
train.py uses mask_path.parent.name as the class label and Image.open(mask_path)
to load the image, so this works for raw JPG crops organized by class folder.
"""

import csv
from pathlib import Path

DATA_ROOT = Path(
    "/mnt2/anita/TechBio/classified/PKG - AML-Cytomorphology_LMU/for_fang/task_combine"
)
OUT_CSV = Path(__file__).parent / "task_combine_summary.csv"


def main() -> None:
    rows: list[dict] = []
    for cls_dir in sorted(DATA_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.tif")) + sorted(cls_dir.glob("*.tiff"))
        for img in images:
            rows.append({"status": "OK", "mask_path": str(img), "cell_type": cls_dir.name})

    if not rows:
        raise RuntimeError(f"No images found under {DATA_ROOT}")

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "mask_path", "cell_type"])
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    counts = Counter(r["cell_type"] for r in rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print("Class counts:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
