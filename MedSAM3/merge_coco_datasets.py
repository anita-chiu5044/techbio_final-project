"""
Merge two COCO datasets into one.

Usage:
    python merge_coco_datasets.py \
        --base    data/lisc_medsam3 \
        --extra   data/pseudo_labels \
        --output  data/merged

The output directory mirrors the split structure (train/, valid/).
Images are copied into the output directory.
IDs are reassigned to avoid collisions.
"""

import argparse
import json
import shutil
from pathlib import Path


def load_coco(split_dir: Path) -> dict | None:
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return None
    with open(ann_path) as f:
        return json.load(f)


def merge_split(base_dir: Path, extra_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    base  = load_coco(base_dir)
    extra = load_coco(extra_dir)

    if base is None and extra is None:
        return
    if base is None:
        base = {"images": [], "annotations": [], "categories": extra["categories"]}
    if extra is None:
        extra = {"images": [], "annotations": [], "categories": base["categories"]}

    # Copy base images
    for img in base["images"]:
        src = base_dir / img["file_name"]
        dst = out_dir  / img["file_name"]
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Copy extra images (rename on collision)
    filename_remap = {}   # old file_name → new file_name
    for img in extra["images"]:
        src = extra_dir / img["file_name"]
        dst_name = img["file_name"]
        dst = out_dir / dst_name
        # Avoid overwriting a base image with the same filename
        if dst.exists():
            dst_name = "extra_" + img["file_name"]
            dst = out_dir / dst_name
        filename_remap[img["file_name"]] = dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Reassign IDs: base keeps its IDs, extra gets shifted
    max_img_id = max((img["id"] for img in base["images"]), default=-1)
    max_ann_id = max((ann["id"] for ann in base["annotations"]), default=-1)

    extra_images = []
    id_remap = {}   # old image id → new image id
    for img in extra["images"]:
        new_id = max_img_id + 1 + extra["images"].index(img)
        id_remap[img["id"]] = new_id
        extra_images.append({
            **img,
            "id": new_id,
            "file_name": filename_remap.get(img["file_name"], img["file_name"]),
        })

    extra_anns = []
    for i, ann in enumerate(extra["annotations"]):
        extra_anns.append({
            **ann,
            "id": max_ann_id + 1 + i,
            "image_id": id_remap[ann["image_id"]],
        })

    merged = {
        "info": {"description": "Merged COCO dataset"},
        "images":      base["images"] + extra_images,
        "annotations": base["annotations"] + extra_anns,
        "categories":  base["categories"],
    }

    out_ann = out_dir / "_annotations.coco.json"
    with open(out_ann, "w") as f:
        json.dump(merged, f, indent=2)

    n_base  = len(base["images"])
    n_extra = len(extra_images)
    print(f"  {out_dir.name:6s}: {n_base} base + {n_extra} extra = {n_base + n_extra} images")
    print(f"           → {out_ann}")


def main():
    parser = argparse.ArgumentParser(description="Merge two COCO dataset directories")
    parser.add_argument("--base",   required=True, help="Base COCO dataset dir (e.g. data/lisc_medsam3)")
    parser.add_argument("--extra",  required=True, help="Extra COCO dataset dir (e.g. data/pseudo_labels)")
    parser.add_argument("--output", required=True, help="Output merged dataset dir")
    args = parser.parse_args()

    base   = Path(args.base).expanduser()
    extra  = Path(args.extra).expanduser()
    output = Path(args.output).expanduser()

    print(f"Merging:")
    print(f"  base  : {base}")
    print(f"  extra : {extra}")
    print(f"  output: {output}\n")

    for split in ["train", "valid"]:
        merge_split(base / split, extra / split, output / split)

    print("\nDone! Update data_dir in your config to:", output)


if __name__ == "__main__":
    main()
