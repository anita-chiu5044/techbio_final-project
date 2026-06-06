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
                        help="Stable session/case id, e.g. case_001 or uploaded filename stem.")
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
    parser.add_argument("--python-executable", default=sys.executable)
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
            (args.session_id, args.user_id, str(args.input.resolve()), "pipeline_completed", "full_agent_pipeline_v1"),
        )
    tools.start_conversation(conversation_id, user_id=args.user_id)
    tools.set_active_case(conversation_id, args.session_id, user_id=args.user_id)

    detections = read_jsonl(session["yolo"] / "detections.jsonl")
    imported = 0
    for rec in detections:
        if rec.get("class_label") != "WBC":
            continue
        rec = {**rec, "case_id": args.session_id}
        tools.import_yolo_detection(args.session_id, rec, cell_id=rec["detection_id"])
        imported += 1
    print(f"Imported WBC YOLO detections into DB: {imported}")
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
        raise FileNotFoundError(f"MedSAM summary not found: {summary}")
    applied = 0
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
                        try_keys.append(str(mask.relative_to(session["medsam_output"])))
                    except ValueError:
                        pass
                cell_id = next((mapping[k] for k in try_keys if k in mapping), None)
            else:
                stem = Path(row.get("image", "")).with_suffix("").name + "_mask.png"
                cell_id = mapping.get(str(Path(row.get("category", "")) / row.get("cell_type", "") / stem))
            if not cell_id:
                continue
            tools.apply_medsam_result(cell_id, row)
            applied += 1
    print(f"Applied MedSAM summary rows to DB: {applied}")
    return applied


def write_report_payload(args: argparse.Namespace, tools: AgentTools, session: dict[str, Path]) -> None:
    summary = tools.summarize_case(args.session_id)
    report = tools.generate_case_report(args.session_id)
    payload = {
        "case_id": args.session_id,
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
        "case_id": args.session_id,
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
            args.python_executable, REPO_ROOT / "export_yolo_detection_manifest.py",
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
        run([
            args.python_executable, REPO_ROOT / "yolo_to_medsam_patches.py",
            "--detections", session["yolo"] / "detections.jsonl",
            "--output-root", session["medsam_input"],
            "--context-scale", args.context_scale,
            "--selection", "all",
            "--mapping-csv", session["cell_map"],
        ], dry_run=args.dry_run)

    if should_run(args, "medsam"):
        cmd = [
            args.python_executable, REPO_ROOT / "MedSAM3" / "tiff_wbc_inference.py",
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

    tools = ensure_case_and_import_yolo(args, session)
    apply_medsam_summary(tools, session)

    if should_run(args, "classifier"):
        cmd = [
            args.python_executable, REPO_ROOT / "scripts" / "run_classifier_agent_pipeline.py",
            "--image", session["medsam_output"],
            "--ckpt", args.classifier_ckpt,
            "--db", args.db or (session["root"] / "ymca_agent.db"),
            "--output-dir", session["classifier"],
            "--case-id", args.session_id,
            "--conversation-id", args.conversation_id or f"conv_{args.session_id}",
            "--user-id", args.user_id,
            "--topk", args.classifier_topk,
            "--cell-map-csv", session["cell_map"],
        ]
        if args.logit_adjustment:
            cmd.append("--logit-adjustment")
        run(cmd, dry_run=False)

    # The classifier bridge already generates a report. Regenerate once from the
    # full runner root so all stage paths are summarized together.
    tools = AgentTools(args.db or (session["root"] / "ymca_agent.db"), guidelines_dir=args.guidelines_dir)
    write_report_payload(args, tools, session)


if __name__ == "__main__":
    main()
