#!/usr/bin/env python3
"""Run classifier inference and push results through the YMCA agent DB/report flow.

MVP bridge:
  classifier_inference.py --ckpt .../dinobloom_ce_uniform/best.pth --logit-adjustment
    -> AgentTools.apply_classifier_result()
    -> AgentTools.summarize_case()
    -> AgentTools.generate_case_report()

The classifier checkpoint is intentionally swappable. Current default is
dinobloom_ce_uniform (macro_F1=0.864, DinoBloom-B backbone).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ymca_agent.storage import connect  # noqa: E402
from ymca_agent.tools import AgentTools  # noqa: E402

DEFAULT_CKPT = Path("/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/dinobloom_ce_uniform/best.pth")
DEFAULT_DB = Path("/home/yucheng/Desktop/techbio_pipeline_output/agent_pipeline_smoke/ymca_agent.db")
DEFAULT_OUTPUT_DIR = Path("/home/yucheng/Desktop/techbio_pipeline_output/agent_pipeline_smoke")
DEFAULT_GUIDELINES = WORKSPACE_ROOT / "reporting_guidelines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifier -> agent DB -> summary -> report smoke pipeline.")
    parser.add_argument("--image", type=Path, default=None,
                        help="Clean patch image file or folder. Required unless --classifier-json is provided.")
    parser.add_argument("--classifier-json", type=Path, default=None,
                        help="Existing classifier_inference.py JSON output to ingest instead of rerunning inference.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                        help="Classifier checkpoint. Default is dinobloom_ce_uniform/best.pth.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--case-id", default="case_agent_pipeline_smoke")
    parser.add_argument("--conversation-id", default="conv_agent_pipeline_smoke")
    parser.add_argument("--user-id", default="local_user")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--guidelines-dir", type=Path, default=DEFAULT_GUIDELINES)
    parser.add_argument("--cell-map-csv", type=Path, default=None,
                        help="Optional CSV with columns image,cell_id. Useful when image paths must map to existing DB cells.")
    parser.add_argument("--logit-adjustment", action="store_true",
                        help="Apply checkpoint class-frequency logit adjustment during classifier inference.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--classifier-worker-url", default=os.environ.get("YMCA_CLASSIFIER_WORKER_URL"),
                        help="Optional local model worker URL, e.g. http://127.0.0.1:8777. Falls back to classifier subprocess when unset.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip cells that already have model_label set. Default true.")
    return parser.parse_args()


def _worker_endpoint(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def run_classifier_via_worker(args: argparse.Namespace, output_json: Path) -> Path:
    if args.image is None:
        raise ValueError("--image is required when --classifier-json is not provided")
    if not args.classifier_worker_url:
        raise ValueError("classifier_worker_url is not set")
    payload = {
        "image": str(args.image),
        "ckpt": str(args.ckpt),
        "topk": int(args.topk),
        "image_size": 224,
        "logit_adjustment": bool(args.logit_adjustment),
    }
    data = json.dumps(payload).encode("utf-8")
    url = _worker_endpoint(args.classifier_worker_url, "/classify")
    print(f"$ POST {url}  # classifier worker", flush=True)
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"classifier worker unavailable at {args.classifier_worker_url}: {exc}") from exc
    if "error" in response:
        raise RuntimeError(f"classifier worker error: {response['error']}")
    records = response.get("records")
    if not isinstance(records, list):
        raise RuntimeError("classifier worker response missing records list")
    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Saved → {output_json}  ({len(records)} records) via classifier worker", flush=True)
    return output_json


def run_classifier(args: argparse.Namespace, output_json: Path) -> Path:
    if args.classifier_worker_url:
        return run_classifier_via_worker(args, output_json)
    if args.image is None:
        raise ValueError("--image is required when --classifier-json is not provided")
    script = REPO_ROOT / "checkpoints_classifier" / "classifier_inference.py"
    cmd = [
        args.python_executable,
        str(script),
        "--image", str(args.image),
        "--ckpt", str(args.ckpt),
        "--topk", str(args.topk),
        "--output", str(output_json),
        "--format", "json",
    ]
    if args.logit_adjustment:
        cmd.append("--logit-adjustment")
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return output_json


def load_classifier_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError("classifier JSON must be a record or list of records")
    return payload


def load_cell_map(path: Path | None, image_dir: Path | None = None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "cell_id"}.issubset(reader.fieldnames or []):
            raise ValueError("--cell-map-csv must contain image and cell_id columns")
        for row in reader:
            img_rel = row["image"]
            cell_id = row["cell_id"]
            mapping[img_rel] = cell_id
            # Resolve relative to cell_map parent (session root) and to image_dir
            # (medsam_output), since mask paths from the classifier are absolute
            # paths inside medsam_output.
            mapping[str((path.parent / img_rel).resolve())] = cell_id
            if image_dir is not None:
                mapping[str((image_dir / img_rel).resolve())] = cell_id
    return mapping


def ensure_case(tools: AgentTools, args: argparse.Namespace) -> None:
    original = str(args.image or args.classifier_json or "classifier_json")
    with connect(tools.db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO cases (case_id, user_id, original_image_path, status, pipeline_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (args.case_id, args.user_id, original, "classifier_completed", "classifier_agent_bridge_v1"),
        )
    tools.start_conversation(args.conversation_id, user_id=args.user_id)
    tools.set_active_case(args.conversation_id, args.case_id, user_id=args.user_id)


def existing_cell_for_patch(tools: AgentTools, clean_patch_path: str) -> str | None:
    with connect(tools.db_path) as conn:
        row = conn.execute(
            """
            SELECT cell_id FROM cells
            WHERE clean_patch_path = ? AND is_current = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (clean_patch_path,),
        ).fetchone()
    return None if row is None else row["cell_id"]


def sanitize_cell_id(image_path: str, index: int) -> str:
    path = Path(image_path)
    stem = path.stem.replace(" ", "_")
    keep = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
    return f"cell_{index:05d}_{keep[:80]}"


def ensure_cell(tools: AgentTools, case_id: str, cell_id: str, clean_patch_path: str) -> dict[str, Any]:
    with connect(tools.db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO cells (
                cell_id, case_id, bbox_xyxy_original, downstream_eligible,
                clean_patch_path, mask_path, segmentation_status, segmentation_quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cell_id,
                case_id,
                json.dumps([0, 0, 1, 1]),
                1,
                clean_patch_path,
                clean_patch_path,
                "ok",
                1.0,
            ),
        )
    try:
        return tools.get_cell(cell_id)
    except KeyError:
        # Cell exists but is_current=0 (previously excluded). Restore it so the
        # classifier can update its label without re-creating an orphan.
        with connect(tools.db_path) as conn:
            conn.execute("UPDATE cells SET is_current=1 WHERE cell_id=?", (cell_id,))
        return tools.get_cell(cell_id)


def apply_records(tools: AgentTools, args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    cell_map = load_cell_map(args.cell_map_csv, image_dir=args.image if args.image and args.image.is_dir() else None)
    applied: list[str] = []
    skipped_existing: list[str] = []
    created_or_reused: list[str] = []

    for index, record in enumerate(records, start=1):
        image = str(record["image"])
        resolved_image = str(Path(image).resolve())
        cell_id = cell_map.get(image) or cell_map.get(resolved_image) or existing_cell_for_patch(tools, image)
        if cell_id is None:
            cell_id = sanitize_cell_id(image, index)
        cell = ensure_cell(tools, args.case_id, cell_id, image)
        created_or_reused.append(cell_id)
        if cell.get("model_label") is not None and args.skip_existing:
            skipped_existing.append(cell_id)
            continue
        tools.apply_classifier_result(
            cell_id,
            record,
            classifier_checkpoint=str(args.ckpt),
            label_map_version="classifier_flat15_lmu_v1",
            preprocess_version="convnet_224_imagenet_v1",
        )
        applied.append(cell_id)

    return {
        "created_or_reused_cell_ids": created_or_reused,
        "applied_cell_ids": applied,
        "skipped_existing_cell_ids": skipped_existing,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier_json = args.classifier_json or (args.output_dir / "classifier_results.json")

    if args.classifier_json is None:
        run_classifier(args, classifier_json)

    records = load_classifier_records(classifier_json)
    tools = AgentTools(args.db, guidelines_dir=args.guidelines_dir)
    ensure_case(tools, args)
    apply_result = apply_records(tools, args, records)
    summary = tools.summarize_case(args.case_id)
    report = tools.generate_case_report(args.case_id)

    payload = {
        "case_id": args.case_id,
        "conversation_id": args.conversation_id,
        "db": str(args.db),
        "classifier_checkpoint": str(args.ckpt),
        "classifier_json": str(classifier_json),
        "record_count": len(records),
        "apply_result": apply_result,
        "summary": summary,
        "report_id": report["report_id"],
        "report_content": report["content"],
        "report_safety": report["safety"],
    }
    summary_path = args.output_dir / "agent_pipeline_summary.json"
    report_path = args.output_dir / "agent_report.txt"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    report_path.write_text(report["content"] + "\n")

    print(json.dumps({
        "case_id": args.case_id,
        "db": str(args.db),
        "record_count": len(records),
        "applied": len(apply_result["applied_cell_ids"]),
        "skipped_existing": len(apply_result["skipped_existing_cell_ids"]),
        "review_needed_count": summary["review_needed_count"],
        "hard_counts": summary["hard_counts"],
        "summary_json": str(summary_path),
        "report_txt": str(report_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
