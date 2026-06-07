#!/usr/bin/env python3
"""Full local pipeline runner for user sessions.

Input full image(s):
  YOLO -> ROI TIFFs -> MedSAM clean patches -> classifier -> agent DB summary/report

This is an orchestration script. It keeps checkpoints swappable and writes all
intermediate artifacts under one session output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ymca_agent.storage import connect  # noqa: E402
from ymca_agent.tools import AgentTools  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("/home/yucheng/Desktop/techbio_pipeline_output/full_agent_sessions")
DEFAULT_YOLO = REPO_ROOT / "best.pt"
DEFAULT_MEDSAM_CONFIG = REPO_ROOT / "MedSAM3" / "configs" / "lisc_lora_config.yaml"
DEFAULT_MEDSAM_DIR = REPO_ROOT / "MedSAM3"
DEFAULT_CLASSIFIER = Path("/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/dinobloom_ce_uniform/best.pth")
DEFAULT_GUIDELINES = WORKSPACE_ROOT / "reporting_guidelines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full local YOLO->MedSAM->classifier->agent pipeline.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Full smear/cell image file or folder of user-uploaded images.")
    parser.add_argument("--session-id", required=True,
                        help="Stable session directory id, e.g. case_001 or uploaded filename stem.")
    parser.add_argument("--case-id", default=None,
                        help="Logical DB case id. Defaults to --session-id. Use this when rerunning an imported demo case whose case_id differs from its session folder.")
    parser.add_argument("--user-id", default="local_user")
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit input images for smoke tests.")
    parser.add_argument("--context-scale", type=float, default=1.3)
    parser.add_argument("--medsam-config", type=Path, default=DEFAULT_MEDSAM_CONFIG)
    parser.add_argument("--medsam3-dir", type=Path, default=DEFAULT_MEDSAM_DIR)
    parser.add_argument("--medsam-threshold", type=float, default=0.5)
    parser.add_argument("--medsam-nms-iou", type=float, default=0.5)
    parser.add_argument("--medsam-max-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--classifier-ckpt", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--classifier-topk", type=int, default=5)
    parser.add_argument("--logit-adjustment", action="store_true", default=True)
    parser.add_argument("--no-logit-adjustment", dest="logit_adjustment", action="store_false")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--guidelines-dir", type=Path, default=DEFAULT_GUIDELINES)
    parser.add_argument("--python-executable", default=sys.executable,
                        help="Default Python executable for pipeline stages.")
    parser.add_argument("--yolo-python", default=None,
                        help="Python executable for YOLO stage; use an env with ultralytics.")
    parser.add_argument("--medsam-python", default=None,
                        help="Python executable for MedSAM stage.")
    parser.add_argument("--classifier-python", default=None,
                        help="Python executable for classifier/agent bridge stage.")
    parser.add_argument("--classifier-worker-url", default=os.environ.get("YMCA_CLASSIFIER_WORKER_URL"),
                        help="Optional local classifier worker URL, e.g. http://127.0.0.1:8777.")
    parser.add_argument("--yolo-gate-conf", type=float, default=0.50,
                        help="YOLO confidence threshold for downstream gating. "
                             "Detections below this are imported but marked downstream_eligible=0 "
                             "and queued for manual review. Set to 0 to disable gating.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only. Does not run models or write DB updates.")
    parser.add_argument("--start-at", choices=["yolo", "roi", "medsam", "classifier", "agent"], default="yolo",
                        help="Resume from a stage if earlier artifacts already exist.")
    parser.add_argument("--stop-after", choices=["yolo", "roi", "medsam", "classifier", "agent"], default="agent")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path = REPO_ROOT, dry_run: bool = False) -> None:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def stage_index(name: str) -> int:
    return ["yolo", "roi", "medsam", "classifier", "agent"].index(name)


def should_run(args: argparse.Namespace, stage: str) -> bool:
    return stage_index(args.start_at) <= stage_index(stage) <= stage_index(args.stop_after)


def effective_case_id(args: argparse.Namespace) -> str:
    return args.case_id or args.session_id


def reset_stage_dir(path: Path) -> None:
    """Clear stale stage artifacts before regenerating a downstream input folder."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_review_exclude_file(args: argparse.Namespace, session: dict[str, Path]) -> Path | None:
    """Merge low-confidence gate ids with reviewer-excluded cells for ROI reruns."""
    gated_file = session["root"] / "gated_detections.json"
    if args.dry_run:
        return gated_file if gated_file.exists() else None

    excluded_ids: set[str] = set()
    if gated_file.exists():
        try:
            excluded_ids.update(json.loads(gated_file.read_text()).get("gated_ids", []))
        except json.JSONDecodeError:
            pass

    db_path = args.db or (session["root"] / "ymca_agent.db")
    if db_path.exists():
        with connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(detection_id, cell_id) AS detection_id
                FROM cells
                WHERE case_id = ? AND is_current = 1 AND review_status = 'excluded'
                """,
                (effective_case_id(args),),
            ).fetchall()
        excluded_ids.update(row["detection_id"] for row in rows if row["detection_id"])

    if not excluded_ids:
        return None
    out = session["root"] / "rerun_excluded_detections.json"
    out.write_text(json.dumps({"gated_ids": sorted(excluded_ids)}, indent=2))
    return out


def input_dataset_root(input_path: Path, session_dir: Path, dry_run: bool) -> tuple[Path, str | None]:
    """Return dataset root and optional path-prefix for YOLO export.

    YOLO exporter expects a folder. For a single user image, create/use a small
    manifest_input folder containing a symlink so relative paths remain stable.
    """
    if input_path.is_dir():
        return input_path, None
    dataset_root = session_dir / "input_images"
    link_path = dataset_root / input_path.name
    if not dry_run:
        dataset_root.mkdir(parents=True, exist_ok=True)
        if not link_path.exists():
            try:
                link_path.symlink_to(input_path.resolve())
            except OSError:
                import shutil
                shutil.copy2(input_path, link_path)
    return dataset_root, None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def wbc_nms(detections: list[dict[str, Any]], iou_threshold: float = 0.45) -> list[dict[str, Any]]:
    """Per-source-image IoU/NMS for WBC detections only. Non-WBC pass through unchanged."""
    non_wbc = [d for d in detections if d.get("class_label") != "WBC"]
    wbc = [d for d in detections if d.get("class_label") == "WBC"]

    by_image: dict[str, list[dict[str, Any]]] = {}
    for det in wbc:
        by_image.setdefault(str(det.get("source_image_path", "")), []).append(det)

    kept: list[dict[str, Any]] = []
    suppressed_total = 0
    for img_dets in by_image.values():
        sorted_dets = sorted(img_dets, key=lambda d: float(d.get("confidence", 0)), reverse=True)
        suppressed = [False] * len(sorted_dets)
        for i, det_i in enumerate(sorted_dets):
            if suppressed[i]:
                continue
            kept.append(det_i)
            bbox_i = det_i.get("bbox_xyxy_original", [])
            if len(bbox_i) != 4:
                continue
            for j in range(i + 1, len(sorted_dets)):
                if suppressed[j]:
                    continue
                bbox_j = sorted_dets[j].get("bbox_xyxy_original", [])
                if len(bbox_j) == 4 and _iou(bbox_i, bbox_j) > iou_threshold:
                    suppressed[j] = True
                    suppressed_total += 1

    if suppressed_total:
        print(f"WBC NMS: suppressed {suppressed_total} duplicate WBC detections (IoU>{iou_threshold})")
    return non_wbc + kept


def ensure_case_and_import_yolo(args: argparse.Namespace, session: dict[str, Path]) -> AgentTools:
    db_path = args.db or (session["root"] / "ymca_agent.db")
    tools = AgentTools(db_path, guidelines_dir=args.guidelines_dir)
    conversation_id = args.conversation_id or f"conv_{args.session_id}"
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO cases (case_id, user_id, original_image_path, status, pipeline_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (effective_case_id(args), args.user_id, str(args.input.resolve()), "pipeline_completed", "full_agent_pipeline_v1"),
        )
    tools.start_conversation(conversation_id, user_id=args.user_id)
    tools.set_active_case(conversation_id, effective_case_id(args), user_id=args.user_id)

    raw_detections = read_jsonl(session["yolo"] / "detections.jsonl")
    # Resolve relative patch_path values against the yolo output directory so that
    # the stored clean_patch_path is always an absolute path that exists on disk.
    yolo_dir = session["yolo"]
    raw_detections = [
        {**det, "patch_path": str(yolo_dir / det["patch_path"])}
        if det.get("patch_path") and not Path(det["patch_path"]).is_absolute()
        else det
        for det in raw_detections
    ]
    # Prefix detection_id with image stem so IDs are unique across images in a
    # multi-image session.  This ensures re-uploading the same image replaces its
    # cells (ON CONFLICT UPDATE) while a new image adds distinct cells.
    raw_detections = [
        {**det, "detection_id": f"{Path(det.get('source_image_path', 'unknown')).stem}_{det['detection_id']}"}
        for det in raw_detections
    ]
    detections = wbc_nms(raw_detections, iou_threshold=0.30)
    imported = 0
    gated = 0
    gate_conf = getattr(args, "yolo_gate_conf", 0.50)
    for rec in detections:
        if rec.get("class_label") != "WBC":
            continue
        rec = {**rec, "case_id": effective_case_id(args)}
        conf = float(rec.get("confidence", 0))
        if gate_conf > 0 and conf < gate_conf:
            # Low confidence — import but mark as not downstream-eligible
            rec["downstream_eligible"] = False
            tools.import_yolo_detection(effective_case_id(args), rec, cell_id=rec["detection_id"])
            # Mark for manual review
            try:
                tools.update_cell_review(
                    rec["detection_id"],
                    review_status="queued_for_review",
                    note=f"YOLO confidence {conf:.3f} below gate threshold {gate_conf}",
                    reviewer_id="pipeline_gate",
                )
            except Exception:
                pass
            gated += 1
        else:
            tools.import_yolo_detection(effective_case_id(args), rec, cell_id=rec["detection_id"])
        imported += 1
    print(f"Imported WBC YOLO detections: {imported} (gated: {gated} below conf={gate_conf})")
    # Write gated detection IDs so ROI stage can skip them
    if gated > 0:
        gated_ids = []
        for rec in detections:
            if rec.get("class_label") == "WBC" and float(rec.get("confidence", 0)) < gate_conf:
                gated_ids.append(rec["detection_id"])
        (session["root"] / "gated_detections.json").write_text(
            json.dumps({"gate_conf": gate_conf, "gated_ids": gated_ids}, indent=2)
        )
    return tools


def load_cell_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = row["image"]
            cell_id = row["cell_id"]
            mapping[image] = cell_id
            mapping[str((path.parent / image).resolve())] = cell_id
    return mapping


def apply_medsam_summary(tools: AgentTools, session: dict[str, Path]) -> int:
    summary = session["medsam_output"] / "inference_summary.csv"
    mapping = load_cell_map(session["cell_map"])
    if not summary.exists():
        print("MedSAM summary not found (stage was skipped) — no MedSAM results to apply")
        return 0
    # Load gated detection IDs so we can skip cells that were never imported into DB
    gated_ids: set[str] = set()
    for fname in ("gated_detections.json", "rerun_excluded_detections.json"):
        gated_file = session["root"] / fname
        if gated_file.exists():
            try:
                gated_ids.update(json.loads(gated_file.read_text()).get("gated_ids", []))
            except json.JSONDecodeError:
                pass
    applied = 0
    skipped_gated = 0
    with summary.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mask_path = row.get("mask_path", "")
            if row.get("status", "").upper() == "OK":
                try_keys = []
                if mask_path:
                    mask = Path(mask_path)
                    try_keys.extend([str(mask), str(mask.resolve())])
                    try:
                        rel = str(mask.relative_to(session["medsam_output"]))
                        try_keys.append(rel)
                        try_keys.append(str((session["cell_map"].parent / rel).resolve()))
                    except ValueError:
                        pass
                    image_name = Path(row.get("image", "")).with_suffix("").name + "_mask.png"
                    if image_name:
                        rel_from_row = str(Path(row.get("category", "")) / row.get("cell_type", "") / image_name)
                        try_keys.append(rel_from_row)
                        try_keys.append(str((session["cell_map"].parent / rel_from_row).resolve()))
                cell_id = next((mapping[k] for k in try_keys if k in mapping), None)
            else:
                stem = Path(row.get("image", "")).with_suffix("").name + "_mask.png"
                cell_id = mapping.get(str(Path(row.get("category", "")) / row.get("cell_type", "") / stem))
            if not cell_id:
                continue
            if cell_id in gated_ids:
                skipped_gated += 1
                continue
            try:
                tools.apply_medsam_result(cell_id, row)
                applied += 1
            except KeyError:
                # Cell not in DB — NMS-suppressed or otherwise excluded upstream
                skipped_gated += 1
    if skipped_gated:
        print(f"Skipped {skipped_gated} MedSAM rows for gated/NMS-suppressed detections")
    print(f"Applied MedSAM summary rows to DB: {applied}")
    return applied


def write_report_payload(args: argparse.Namespace, tools: AgentTools, session: dict[str, Path]) -> None:
    summary = tools.summarize_case(effective_case_id(args))
    report = tools.generate_case_report(effective_case_id(args))
    payload = {
        "case_id": effective_case_id(args),
        "conversation_id": args.conversation_id or f"conv_{args.session_id}",
        "db": str(args.db or (session["root"] / "ymca_agent.db")),
        "paths": {k: str(v) for k, v in session.items()},
        "summary": summary,
        "report_id": report["report_id"],
        "report_content": report["content"],
        "report_safety": report["safety"],
    }
    (session["root"] / "agent_pipeline_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (session["root"] / "agent_report.txt").write_text(report["content"] + "\n")
    print(json.dumps({
        "case_id": effective_case_id(args),
        "db": payload["db"],
        "summary_json": str(session["root"] / "agent_pipeline_summary.json"),
        "report_txt": str(session["root"] / "agent_report.txt"),
        "total_cells": summary["total_cells"],
        "review_needed_count": summary["review_needed_count"],
        "hard_counts": summary["hard_counts"],
        "model_counts_raw": summary["model_counts_raw"],
    }, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    import re
    if not re.match(r'^[A-Za-z0-9_-]+$', args.session_id):
        raise SystemExit(f"Invalid session-id: {args.session_id!r} (must match [A-Za-z0-9_-]+)")
    root = args.output_root / args.session_id
    session = {
        "root": root,
        "yolo": root / "01_yolo",
        "medsam_input": root / "02_medsam_input",
        "medsam_output": root / "03_medsam_output",
        "classifier": root / "04_classifier",
        "cell_map": root / "cell_map.csv",
    }
    if not args.dry_run:
        for path in session.values():
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)

    dataset_root, path_prefix = input_dataset_root(args.input, root, args.dry_run)

    if should_run(args, "yolo"):
        cmd = [
            args.yolo_python or args.python_executable, REPO_ROOT / "export_yolo_detection_manifest.py",
            "--dataset-root", dataset_root,
            "--output-root", session["yolo"],
            "--model-path", args.yolo_model,
            "--device", args.yolo_device,
            "--imgsz", args.yolo_imgsz,
            "--batch-size", args.yolo_batch_size,
            "--conf", args.yolo_conf,
            "--case-id-mode", "none",
            "--save-patches",
            "--per-image-json",
        ]
        if args.limit is not None:
            cmd.extend(["--limit", args.limit])
        if path_prefix:
            cmd.extend(["--path-prefix", path_prefix])
        run(cmd, dry_run=args.dry_run)

    if should_run(args, "roi"):
        if not args.dry_run:
            reset_stage_dir(session["medsam_input"])
        roi_cmd = [
            args.python_executable, REPO_ROOT / "yolo_to_medsam_patches.py",
            "--detections", session["yolo"] / "detections.jsonl",
            "--output-root", session["medsam_input"],
            "--context-scale", args.context_scale,
            "--selection", "all",
            "--mapping-csv", session["cell_map"],
        ]
        exclude_file = write_review_exclude_file(args, session)
        if exclude_file is not None:
            roi_cmd.extend(["--exclude-ids", exclude_file])
        run(roi_cmd, dry_run=args.dry_run)

    if should_run(args, "medsam"):
        tiff_count = sum(1 for ext in ("*.tiff", "*.tif") for _ in session["medsam_input"].rglob(ext)) if not args.dry_run else 1
        if tiff_count == 0:
            print("No .tiff patches in medsam_input — skipping MedSAM stage")
        else:
            if not args.dry_run:
                reset_stage_dir(session["medsam_output"])
            cmd = [
                args.medsam_python or args.python_executable, REPO_ROOT / "MedSAM3" / "tiff_wbc_inference.py",
                "--data-root", session["medsam_input"],
                "--config", args.medsam_config,
                "--output-dir", session["medsam_output"],
                "--medsam3-dir", args.medsam3_dir,
                "--threshold", args.medsam_threshold,
                "--nms-iou", args.medsam_nms_iou,
                "--masked-output",
                "--fill-holes",
                "--erythroid-categories",
            ]
            if args.skip_existing:
                cmd.append("--skip-existing")
            if args.medsam_max_images is not None:
                cmd.extend(["--max-images", args.medsam_max_images])
            run(cmd, cwd=args.medsam3_dir, dry_run=args.dry_run)

    if args.dry_run:
        return

    if should_run(args, "yolo"):
        tools = ensure_case_and_import_yolo(args, session)
    else:
        # Rerun: use existing DB without re-importing YOLO detections
        db_path = args.db or (session["root"] / "ymca_agent.db")
        tools = AgentTools(db_path, guidelines_dir=args.guidelines_dir)
    apply_medsam_summary(tools, session)

    if should_run(args, "classifier"):
        png_count = sum(1 for _ in session["medsam_output"].rglob("*.png")) if session["medsam_output"].exists() else 0
        if png_count == 0:
            print("No mask images in medsam_output — skipping classifier stage")
        else:
            cmd = [
                args.classifier_python or args.python_executable, REPO_ROOT / "scripts" / "run_classifier_agent_pipeline.py",
                "--image", session["medsam_output"],
                "--ckpt", args.classifier_ckpt,
                "--db", args.db or (session["root"] / "ymca_agent.db"),
                "--output-dir", session["classifier"],
                "--case-id", effective_case_id(args),
                "--conversation-id", args.conversation_id or f"conv_{args.session_id}",
                "--user-id", args.user_id,
                "--topk", args.classifier_topk,
                "--cell-map-csv", session["cell_map"],
            ]
            if args.classifier_worker_url:
                cmd.extend(["--classifier-worker-url", args.classifier_worker_url])
            if args.logit_adjustment:
                cmd.append("--logit-adjustment")
            run(cmd, dry_run=False)

    # The classifier bridge already generates a report. Regenerate once from the
    # full runner root so all stage paths are summarized together.
    tools = AgentTools(args.db or (session["root"] / "ymca_agent.db"), guidelines_dir=args.guidelines_dir)
    write_report_payload(args, tools, session)


if __name__ == "__main__":
    main()
