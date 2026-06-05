from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("best.pt")
VALID_SUFFIXES = (".tif", ".tiff", ".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a YOLO detector and export one JSON object per detection."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--path-prefix",
        default=None,
        help="Only process files whose dataset-relative path starts with this prefix.",
    )
    parser.add_argument(
        "--case-id-mode",
        choices=["parent", "grandparent", "top_level", "none"],
        default="parent",
        help="How to derive case_id from the relative source path.",
    )
    parser.add_argument(
        "--save-patches",
        action="store_true",
        help="Save one tight patch per detection under output-root/patches.",
    )
    parser.add_argument(
        "--per-image-json",
        action="store_true",
        help="Write one JSON file per source image under output-root/per_image_json.",
    )
    return parser.parse_args()


def iter_images(dataset_root: Path) -> list[Path]:
    return sorted(
        p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )


def derive_case_id(rel_path: Path, mode: str) -> str | None:
    parts = rel_path.parts
    if mode == "none":
        return None
    if mode == "parent":
        return parts[-2] if len(parts) >= 2 else rel_path.stem
    if mode == "grandparent":
        return parts[-3] if len(parts) >= 3 else (parts[-2] if len(parts) >= 2 else rel_path.stem)
    if mode == "top_level":
        return parts[0] if parts else rel_path.stem
    raise ValueError(f"Unsupported case-id mode: {mode}")


def ensure_output_dirs(output_root: Path, save_patches: bool, per_image_json: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if save_patches:
        (output_root / "patches").mkdir(parents=True, exist_ok=True)
    if per_image_json:
        (output_root / "per_image_json").mkdir(parents=True, exist_ok=True)


def normalize_prefix(path_prefix: str | None) -> str | None:
    if not path_prefix:
        return None
    return path_prefix.strip("/").replace("\\", "/")


def filter_paths(image_paths: Iterable[Path], dataset_root: Path, path_prefix: str | None) -> list[Path]:
    normalized_prefix = normalize_prefix(path_prefix)
    if not normalized_prefix:
        return list(image_paths)
    return [
        path
        for path in image_paths
        if path.relative_to(dataset_root).as_posix().startswith(normalized_prefix)
    ]


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def save_patch(
    source_path: Path,
    output_root: Path,
    rel_path: Path,
    detection_id: str,
    box_xyxy: tuple[int, int, int, int],
) -> str:
    patch_rel = Path("patches") / rel_path.parent / f"{rel_path.stem}_{detection_id}{rel_path.suffix}"
    patch_path = output_root / patch_rel
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        crop = image.crop(box_xyxy)
        crop.save(patch_path)
    return patch_rel.as_posix()


def build_image_json_path(output_root: Path, rel_path: Path) -> Path:
    return output_root / "per_image_json" / rel_path.with_suffix(".json")


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_root, args.save_patches, args.per_image_json)

    image_paths = iter_images(args.dataset_root)
    image_paths = filter_paths(image_paths, args.dataset_root, args.path_prefix)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No supported images found under {args.dataset_root}")

    model = YOLO(str(args.model_path))
    detections_out = args.output_root / "detections.jsonl"
    images_out = args.output_root / "images.jsonl"
    summary_out = args.output_root / "summary.json"

    detection_counter = 0
    image_counter = 0
    class_counts: dict[str, int] = {}

    with detections_out.open("w") as detections_handle, images_out.open("w") as images_handle:
        total = len(image_paths)
        for start in range(0, total, args.batch_size):
            batch = image_paths[start : start + args.batch_size]
            results = model.predict(
                source=[str(path) for path in batch],
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                verbose=False,
            )

            for source_path, result in zip(batch, results):
                rel_path = source_path.relative_to(args.dataset_root)
                image_counter += 1
                image_id = rel_path.stem
                case_id = derive_case_id(rel_path, args.case_id_mode)
                names = result.names

                with Image.open(source_path) as image:
                    image_width, image_height = image.size

                per_image_count = 0
                per_image_detections = []
                if result.boxes is not None:
                    xyxy_list = result.boxes.xyxy.cpu().numpy().tolist()
                    conf_list = result.boxes.conf.cpu().numpy().tolist()
                    cls_list = result.boxes.cls.cpu().numpy().tolist()

                    for box_index, (xyxy, confidence, cls_id) in enumerate(
                        zip(xyxy_list, conf_list, cls_list), start=1
                    ):
                        detection_counter += 1
                        per_image_count += 1
                        cls_id = int(cls_id)
                        class_label = names[cls_id]
                        class_counts[class_label] = class_counts.get(class_label, 0) + 1
                        detection_id = f"det_{detection_counter:06d}"
                        clipped = clip_box(*xyxy, image_width, image_height)
                        patch_path = None
                        if args.save_patches:
                            patch_path = save_patch(
                                source_path=source_path,
                                output_root=args.output_root,
                                rel_path=rel_path,
                                detection_id=detection_id,
                                box_xyxy=clipped,
                            )

                        record = {
                            "detection_id": detection_id,
                            "case_id": case_id,
                            "image_id": image_id,
                            "source_image_path": str(source_path),
                            "source_image_relative_path": rel_path.as_posix(),
                            "bbox_xyxy_original": [round(float(v), 3) for v in xyxy],
                            "bbox_xyxy_clipped": list(clipped),
                            "confidence": round(float(confidence), 6),
                            "class_id": cls_id,
                            "class_label": class_label,
                            "image_width": image_width,
                            "image_height": image_height,
                            "detector_checkpoint": str(args.model_path),
                            "patch_path": patch_path,
                            "box_index_in_image": box_index,
                        }
                        detections_handle.write(json.dumps(record) + "\n")
                        per_image_detections.append(record)

                image_record = {
                    "image_id": image_id,
                    "case_id": case_id,
                    "source_image_path": str(source_path),
                    "source_image_relative_path": rel_path.as_posix(),
                    "image_width": image_width,
                    "image_height": image_height,
                    "num_detections": per_image_count,
                    "detector_checkpoint": str(args.model_path),
                }
                images_handle.write(json.dumps(image_record) + "\n")
                if args.per_image_json:
                    per_image_payload = {
                        **image_record,
                        "detections": per_image_detections,
                    }
                    image_json_path = build_image_json_path(args.output_root, rel_path)
                    image_json_path.parent.mkdir(parents=True, exist_ok=True)
                    image_json_path.write_text(json.dumps(per_image_payload, indent=2))

            processed = min(start + args.batch_size, total)
            print(f"Processed {processed}/{total}")

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "detector_checkpoint": str(args.model_path),
        "num_images": image_counter,
        "num_detections": detection_counter,
        "class_counts": class_counts,
        "save_patches": args.save_patches,
        "per_image_json": args.per_image_json,
        "case_id_mode": args.case_id_mode,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "batch_size": args.batch_size,
        "path_prefix": args.path_prefix,
        "detections_manifest": str(detections_out),
        "images_manifest": str(images_out),
    }
    summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
