#!/usr/bin/env python3
"""Local HTML UI for YMCA morphology-review demo v0.

No external web framework required. This server exposes a small local UI and JSON
API over the existing AgentTools DB. It is intended for localhost demos only.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sqlite3
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qa_agent_cli import infer_intent  # noqa: E402
from ymca_agent.tools import AgentTools  # noqa: E402
from ymca_agent.qwen_agent import QwenAgent, AgentResponse  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("/home/yucheng/Desktop/techbio_pipeline_output/full_agent_sessions")
DEFAULT_UPLOAD_ROOT = Path("/home/yucheng/Desktop/techbio_pipeline_output/full_agent_uploads")
DEFAULT_CLASSIFIER = Path("/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/dinobloom_ce_uniform/best.pth")
DEFAULT_GUIDELINES = WORKSPACE_ROOT / "reporting_guidelines"
DEFAULT_YOLO_PYTHON = os.environ.get("YMCA_YOLO_PYTHON", "/home/yucheng/miniconda3/envs/AICUP/bin/python")
DEFAULT_MEDSAM_PYTHON = os.environ.get("YMCA_MEDSAM_PYTHON", sys.executable)
DEFAULT_CLASSIFIER_PYTHON = os.environ.get("YMCA_CLASSIFIER_PYTHON", sys.executable)
LABELS = ["BAS", "EBO", "EOS", "KSC", "LYA", "LYT", "MMZ", "MOB", "MON", "MYB", "MYO", "NGB", "NGS", "PMB", "PMO"]


def safe_session_id(value: str) -> str:
    value = (value or "demo_case_ngs").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return value[:80] or "demo_case_ngs"


def json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_body(handler: BaseHTTPRequestHandler) -> bytes:
    n = int(handler.headers.get("Content-Length", "0") or 0)
    return handler.rfile.read(n) if n else b""


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict:
    raw = read_body(handler)
    return json.loads(raw.decode("utf-8") or "{}")


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    if "boundary=" not in content_type:
        raise ValueError("multipart boundary missing")
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    delimiter = ("--" + boundary).encode()
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, sep, content = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = next((h for h in headers if h.lower().startswith("content-disposition:")), "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        content = content.rstrip(b"\r\n")
        if filename_match and filename_match.group(1):
            files.append((Path(filename_match.group(1)).name, content))
        else:
            fields[name_match.group(1)] = content.decode("utf-8", errors="replace")
    return fields, files


class DemoContext:
    def __init__(self, output_root: Path, upload_root: Path, guidelines_dir: Path, classifier_ckpt: Path,
                 yolo_python: str, medsam_python: str, classifier_python: str,
                 agent_mode: str = "fallback", qwen_model_path: str | None = None,
                 qwen_load_in_4bit: bool = True) -> None:
        self.output_root = output_root
        self.upload_root = upload_root
        self.guidelines_dir = guidelines_dir
        self.classifier_ckpt = classifier_ckpt
        self.yolo_python = yolo_python
        self.medsam_python = medsam_python
        self.classifier_python = classifier_python
        self.agent_mode = agent_mode
        self.qwen_agent: QwenAgent | None = None
        if agent_mode == "qwen" and qwen_model_path:
            self.qwen_agent = QwenAgent(
                model_path=qwen_model_path,
                load_in_4bit=qwen_load_in_4bit,
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.output_root / safe_session_id(session_id)

    def db_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "ymca_agent.db"

    def tools(self, session_id: str) -> AgentTools:
        return AgentTools(self.db_path(session_id), guidelines_dir=self.guidelines_dir)

    def case_id(self, session_id: str) -> str:
        db = self.db_path(session_id)
        if not db.exists():
            raise FileNotFoundError(f"No DB for session: {session_id}")
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT case_id FROM cases ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            raise KeyError(f"No case row in session DB: {session_id}")
        return str(row[0])

    def allowed_file(self, path: Path) -> bool:
        try:
            rp = path.resolve()
            roots = [self.output_root.resolve(), self.upload_root.resolve(), Path("/tmp/e2e_smoke").resolve()]
            # Also allow original dataset images (read-only serving)
            if rp.suffix.lower() in {".tiff", ".tif", ".png", ".jpg", ".jpeg"}:
                roots.append(Path("/mnt2").resolve())
            return any(rp == root or root in rp.parents for root in roots)
        except OSError:
            return False


def load_case(ctx: DemoContext, session_id: str) -> dict:
    session_id = safe_session_id(session_id)
    db = ctx.db_path(session_id)
    if not db.exists():
        raise FileNotFoundError(f"No DB for session: {session_id}")
    case_id = ctx.case_id(session_id)
    tools = ctx.tools(session_id)
    summary = tools.summarize_case(case_id)
    cells = tools.list_cells(case_id)
    report = tools.generate_case_report(case_id)
    # Get original image path from cases table
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT original_image_path FROM cases WHERE case_id=?", (case_id,)).fetchone()
    original_image = str(row[0]) if row else None
    # Try session input_images symlink first (more reliable)
    session_input = ctx.session_dir(session_id) / "input_images"
    if session_input.exists():
        for f in session_input.iterdir():
            if f.suffix.lower() in {".tiff", ".tif", ".png", ".jpg", ".jpeg"}:
                original_image = str(f.resolve())
                break
    return {
        "session_id": session_id,
        "case_id": case_id,
        "db": str(db),
        "summary": summary,
        "cells": cells,
        "report": report["content"],
        "original_image": original_image,
        "fallback_mode": "deterministic_tool_router",
        "classifier_checkpoint": str(ctx.classifier_ckpt),
    }


def apply_review(ctx: DemoContext, session_id: str, payload: dict) -> dict:
    tools = ctx.tools(session_id)
    action = payload.get("action")
    cell_id = payload.get("cell_id")
    label = payload.get("label")
    note = payload.get("note")
    reviewer_id = payload.get("reviewer_id") or "demo_clinician"
    if not cell_id:
        raise ValueError("cell_id is required")
    if action == "approve_gated":
        # Validate cell is actually gated
        cell = tools.get_cell(cell_id)
        if cell.get("downstream_eligible") or cell.get("review_status") != "queued_for_review":
            raise ValueError(f"Cell {cell_id} is not gated (status={cell.get('review_status')})")
        # Update downstream_eligible flag
        from ymca_agent.storage import connect as db_connect
        with db_connect(ctx.db_path(session_id)) as conn:
            conn.execute("UPDATE cells SET downstream_eligible=1 WHERE cell_id=?", (cell_id,))
        # Log via proper audit trail
        result = tools.update_cell_review(
            cell_id, review_status="unreviewed",
            note="YOLO gating overridden — cell approved for downstream processing",
            reviewer_id=reviewer_id,
        )
        return {"review": result, "case": load_case(ctx, session_id)}
    elif action == "accept":
        result = tools.update_cell_review(cell_id, review_status="accepted_model_label", note=note, reviewer_id=reviewer_id)
    elif action == "correct":
        if label not in LABELS:
            raise ValueError(f"label must be one of: {', '.join(LABELS)}")
        result = tools.update_cell_review(cell_id, review_label=label, review_status="corrected", note=note, reviewer_id=reviewer_id)
    elif action == "exclude":
        result = tools.update_cell_review(cell_id, review_status="excluded", note=note, reviewer_id=reviewer_id)
    elif action == "unclassifiable":
        result = tools.update_cell_review(cell_id, review_status="unclassifiable", note=note, reviewer_id=reviewer_id)
    else:
        raise ValueError(f"unsupported review action: {action}")
    return {"review": result, "case": load_case(ctx, session_id)}


def _fb(intent: dict, answer: str, case: dict, **extra) -> dict:
    """Build a consistent fallback chat response."""
    return {"mode": "fallback", "intent": intent, "answer": answer, "tool_trace": [], "case": case, **extra}


def _handle_chat_fallback(ctx: DemoContext, session_id: str, message: str) -> dict:
    """Deterministic intent-based chat handler (no LLM)."""
    intent = infer_intent(message)
    tools = ctx.tools(session_id)
    case_id = ctx.case_id(session_id)
    action = intent["action"]
    if action == "summary":
        summary = tools.summarize_case(case_id)
        answer = (
            f"Case {case_id}: {summary['total_cells']} cells total, "
            f"{summary['hard_count_total']} review-ready, "
            f"{summary['review_needed_count']} need review, "
            f"{summary['excluded_count']} excluded."
        )
        if summary.get("hard_counts"):
            answer += "\nHard counts: " + ", ".join(f"{k}:{v}" for k, v in summary["hard_counts"].items())
        if summary.get("disease_warnings"):
            answer += "\nWarnings: " + "; ".join(summary["disease_warnings"])
        return _fb(intent, answer, load_case(ctx, session_id), data=summary)
    if action == "report":
        report = tools.generate_case_report(case_id)
        return _fb(intent, "已重新產生 morphology-review report。", load_case(ctx, session_id), data=report["content"])
    if action == "uncertain":
        cells = tools.list_uncertain_cells(case_id)
        if cells:
            lines = [f"找到 {len(cells)} 顆需要複核的細胞:"]
            for c in cells[:5]:
                reasons = ", ".join(c.get("review_reasons", []))
                lines.append(f"  {c['cell_id']}: {c.get('model_label','?')} ({(c.get('top_probability',0)*100):.0f}%) — {reasons}")
            if len(cells) > 5:
                lines.append(f"  ... and {len(cells)-5} more")
            answer = "\n".join(lines)
        else:
            answer = "目前沒有需要複核的細胞。"
        return _fb(intent, answer, load_case(ctx, session_id), data=cells)
    if action == "cells":
        cells = tools.list_cells(case_id)
        answer = f"目前 case 有 {len(cells)} 顆 cell。"
        if cells:
            by_label = {}
            for c in cells:
                lbl = c.get("model_label") or "unknown"
                by_label[lbl] = by_label.get(lbl, 0) + 1
            answer += " Model predictions: " + ", ".join(f"{k}:{v}" for k, v in sorted(by_label.items()))
        return _fb(intent, answer, load_case(ctx, session_id), data=cells)
    if action == "cell":
        cell_id = intent.get("cell_id")
        if not cell_id:
            raise ValueError("請指定 cell id")
        cell = tools.get_cell(cell_id)
        answer = f"{cell_id}: model={cell.get('model_label','?')} ({(cell.get('top_probability',0)*100):.1f}%), status={cell.get('review_status','?')}"
        if cell.get("review_label"):
            answer += f", review_label={cell['review_label']}"
        return _fb(intent, answer, load_case(ctx, session_id), data=cell)
    if action in {"accept", "correct", "exclude", "unclassifiable"}:
        review_payload = {"action": action, "cell_id": intent.get("cell_id"), "label": intent.get("label"), "reviewer_id": "demo_clinician"}
        result = apply_review(ctx, session_id, review_payload)
        return {**_fb(intent, "已依照你的指示更新人工複核結果。", result.get("case", load_case(ctx, session_id))), **result}
    return _fb(intent, "我目前只能處理 summary/report/uncertain/cell/review 指令。", load_case(ctx, session_id))


def handle_chat(ctx: DemoContext, session_id: str, message: str) -> dict:
    """Route chat to Qwen agent or deterministic fallback."""
    if ctx.qwen_agent is not None:
        try:
            tools = ctx.tools(session_id)
            case_id = ctx.case_id(session_id)
            response = ctx.qwen_agent.chat(tools, case_id, message)
            result = response.to_dict()
            result["case"] = load_case(ctx, session_id)
            return result
        except Exception as exc:
            print(f"[QwenAgent] error: {exc}, falling back to deterministic handler")
            traceback.print_exc()
            fallback = _handle_chat_fallback(ctx, session_id, message)
            fallback["qwen_error"] = str(exc)
            fallback["mode"] = "fallback (qwen error)"
            return fallback
    return _handle_chat_fallback(ctx, session_id, message)


def run_pipeline(ctx: DemoContext, fields: dict[str, str], files: list[tuple[str, bytes]]) -> dict:
    session_id = safe_session_id(fields.get("session_id", "uploaded_case"))
    upload_dir = ctx.upload_root / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        raise ValueError("At least one image file is required")
    # Save all uploaded files (deduplicate names)
    for filename, content in files:
        base = Path(filename or "upload.png")
        dest = upload_dir / base.name
        counter = 1
        while dest.exists():
            dest = upload_dir / f"{base.stem}_{counter:03d}{base.suffix}"
            counter += 1
        dest.write_bytes(content)
    # Input is the upload folder (YOLO processes all images in it)
    input_path = upload_dir
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_full_agent_pipeline.py"),
        "--input", str(input_path),
        "--session-id", session_id,
        "--user-id", fields.get("user_id", "demo_user"),
        "--yolo-model", str(REPO_ROOT / "best.pt"),
        "--medsam-config", str(REPO_ROOT / "MedSAM3" / "configs" / "lisc_lora_config.yaml"),
        "--medsam3-dir", str(REPO_ROOT / "MedSAM3"),
        "--classifier-ckpt", str(ctx.classifier_ckpt),
        "--logit-adjustment",
        "--output-root", str(ctx.output_root),
        "--start-at", fields.get("start_at", "yolo"),
        "--yolo-python", ctx.yolo_python,
        "--medsam-python", ctx.medsam_python,
        "--classifier-python", ctx.classifier_python,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "session_id": session_id,
        "returncode": proc.returncode,
        "log": proc.stdout[-8000:],
        "case": load_case(ctx, session_id) if proc.returncode == 0 else None,
    }


HTML = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<title>YMCA Morphology Review</title>
<style>
:root{color-scheme:light;--ink:#1a1a2e;--muted:#6b7280;--line:#e5e7eb;--bg:#f3f4f6;--panel:#fff;--accent:#0d6efd;--accent-light:#e7f1ff;--warn:#d97706;--warn-bg:#fffbeb;--bad:#dc2626;--ok:#059669;--ok-bg:#ecfdf5}
*{box-sizing:border-box;margin:0}
body{font-family:'Inter',system-ui,sans-serif;color:var(--ink);background:var(--bg);font-size:14px;overflow:hidden;height:100vh}
/* --- header with tabs --- */
header{height:44px;display:flex;align-items:center;padding:0 16px;background:var(--panel);border-bottom:1px solid var(--line);gap:16px}
header h1{font-size:15px;font-weight:700;white-space:nowrap}
.tab-bar{display:flex;gap:2px}
.tab-bar button{background:none;border:none;border-bottom:2px solid transparent;padding:8px 14px;font-size:13px;font-weight:500;color:var(--muted);cursor:pointer;border-radius:0}
.tab-bar button.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-bar button:hover{color:var(--ink)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
/* --- pages --- */
.page{display:none;height:calc(100vh - 44px);overflow:hidden}
.page.active{display:flex}
/* --- review page: 4 columns --- */
#reviewPage{display:none;gap:8px;padding:8px}
#reviewPage.active{display:grid;grid-template-columns:240px 220px 1fr 340px}
.col{background:var(--panel);border:1px solid var(--line);border-radius:8px;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.col-hdr{padding:10px 12px 6px;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:var(--muted)}
.col-body{flex:1;overflow-y:auto;padding:0 12px 12px}
/* --- shared --- */
h3{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);margin:12px 0 4px}
button,select,input,textarea{font:inherit;font-size:13px}
button{border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:5px 10px;cursor:pointer;font-weight:500}
button:hover{background:var(--bg)}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button.danger{color:var(--bad);border-color:#fca5a5}
input,select,textarea{width:100%;border:1px solid var(--line);border-radius:6px;padding:6px 8px;background:var(--panel)}
textarea{resize:vertical}
.chip{display:inline-flex;align-items:center;gap:3px;border:1px solid var(--line);border-radius:999px;padding:1px 7px;font-size:10px;font-weight:500}
.chip.warn{border-color:#fbbf24;color:var(--warn);background:var(--warn-bg)}
.chip.ok{border-color:#6ee7b7;color:var(--ok);background:var(--ok-bg)}
.chip.info{border-color:#93c5fd;color:#2563eb;background:#eff6ff}
.stat-row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f5f5f5;font-size:12px}
.muted{color:var(--muted);font-size:12px}
pre{white-space:pre-wrap;word-break:break-word;background:#f8fafc;padding:8px;border-radius:6px;border:1px solid var(--line);max-height:260px;overflow:auto;font-size:11px}
.row{display:flex;gap:5px;align-items:center}
.row>*{flex:1}
/* --- cell list --- */
.cell-card{display:grid;grid-template-columns:52px 1fr;gap:8px;padding:8px;border-bottom:1px solid #f5f5f5;cursor:pointer}
.cell-card:hover{background:#f8fafc}
.cell-card.active{background:var(--accent-light);border-left:3px solid var(--accent)}
.thumb{width:52px;height:52px;object-fit:contain;background:#f9fafb;border:1px solid var(--line);border-radius:6px}
/* --- detail --- */
.detail-img{width:100%;max-height:200px;object-fit:contain;background:#f9fafb;border:1px solid var(--line);border-radius:8px;margin-bottom:8px}
.bbox-canvas{width:100%;max-height:200px;object-fit:contain;border:1px solid var(--line);border-radius:8px;margin-bottom:6px;background:#f9fafb}
/* --- chat --- */
.chatlog{flex:1;min-height:80px;overflow-y:auto;border:1px solid var(--line);border-radius:6px;padding:8px;background:#fafbfc;font-size:12px;line-height:1.5}
.chatlog div{margin-bottom:5px}
.chatlog details{margin:3px 0}
.chatlog summary{cursor:pointer;font-size:10px;color:var(--muted)}
/* --- tutorial --- */
#tutorialPage{flex-direction:column;padding:20px 40px;overflow-y:auto}
#tutorialPage.active{display:flex}
.tut{max-width:800px;line-height:1.7;font-size:14px}
.tut h2{font-size:18px;margin:20px 0 8px;color:var(--ink);text-transform:none;letter-spacing:0}
.tut h3{font-size:14px;margin:14px 0 4px;color:var(--ink);text-transform:none;letter-spacing:0}
.tut table{border-collapse:collapse;width:100%;margin:8px 0}
.tut th,.tut td{border:1px solid var(--line);padding:6px 10px;text-align:left;font-size:13px}
.tut th{background:#f8fafc;font-weight:600}
.tut code{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:12px}
</style>
</head>
<body>
<header>
  <h1>YMCA</h1>
  <div class='tab-bar'>
    <button class='active' onclick='showPage("review")'>Review</button>
    <button onclick='showPage("tutorial")'>Tutorial</button>
  </div>
  <div class='hdr-right'>
    <span class='muted' id='agentStatus'>loading...</span>
  </div>
</header>

<!-- ===== REVIEW PAGE ===== -->
<div id='reviewPage' class='page active'>

<!-- Col 1: Session + Summary -->
<div class='col'>
  <div class='col-hdr'>Session</div>
  <div class='col-body'>
    <div class='row' style='margin-bottom:6px'><select id='sessionSelect' style='flex:2'></select><button onclick='loadSelected()'>Load</button></div>
    <details style='margin-bottom:10px'>
      <summary class='muted' style='cursor:pointer'>Upload & Run Pipeline</summary>
      <form id='uploadForm' style='margin-top:6px'>
        <input name='session_id' placeholder='session id (e.g. patient_001)' value='uploaded_demo' style='margin-bottom:3px'/>
        <input name='image' type='file' accept='image/*,.tif,.tiff' multiple style='margin-bottom:3px'/>
        <p class='muted' style='margin:2px 0 4px'>Select one or more blood smear images. All detected WBC will be processed automatically.</p>
        <details>
          <summary class='muted' style='font-size:10px;cursor:pointer'>Advanced options</summary>
          <select name='start_at' style='margin-top:3px'>
            <option value='yolo'>Full run (YOLO → MedSAM → Classifier)</option>
            <option value='roi'>Re-extract ROI patches</option>
            <option value='medsam'>Re-run MedSAM segmentation</option>
            <option value='classifier'>Re-classify existing masks</option>
            <option value='agent'>Re-ingest into DB only</option>
          </select>
        </details>
        <button class='primary' type='submit' style='width:100%;margin-top:6px'>Run Pipeline</button>
      </form>
    </details>
    <h3>Summary</h3>
    <div id='summary'></div>
    <h3>Disease Screening</h3>
    <div id='warnings' class='muted'>No case loaded.</div>
    <details style='margin-top:10px'>
      <summary class='muted' style='cursor:pointer'>Report</summary>
      <pre id='report' style='margin-top:6px'></pre>
    </details>
  </div>
</div>

<!-- Col 2: Cell List -->
<div class='col'>
  <div class='col-hdr'>Cells</div>
  <div id='cells' class='col-body' style='padding:0'></div>
</div>

<!-- Col 3: Cell Detail + Original Image -->
<div class='col'>
  <div class='col-hdr'>Cell Detail</div>
  <div id='detail' class='col-body'><span class='muted'>Select a cell.</span></div>
</div>

<!-- Col 4: Chat (always visible) -->
<div class='col' style='display:flex;flex-direction:column'>
  <div class='col-hdr'>Chat Assistant</div>
  <div style='flex:1;display:flex;flex-direction:column;padding:0 12px 12px;gap:6px;min-height:0'>
    <div id='chatlog' class='chatlog'></div>
    <textarea id='chatInput' rows='2' placeholder='summary / 哪些細胞最不確定？ / 把 det_000001 改成 LYT / report'></textarea>
    <div class='row'><button class='primary' onclick='sendChat()' style='flex:1'>Send</button><button onclick='document.getElementById("chatlog").innerHTML=""'>Clear</button></div>
  </div>
</div>

</div><!-- /reviewPage -->

<!-- ===== TUTORIAL PAGE ===== -->
<div id='tutorialPage' class='page'>
<div class='tut'>
<h2>YMCA Morphology Review — Tutorial</h2>
<p>YMCA is a local morphology-review assistant for white blood cell (WBC) classification. It helps clinical reviewers verify AI-generated cell labels, flag uncertain predictions, and produce research-draft reports.</p>

<h2>Quick Start</h2>
<ol>
<li><b>Load a session</b> — Select an existing session from the dropdown and click <code>Load</code>.</li>
<li><b>Browse cells</b> — The cell list shows each detected WBC with its model prediction and confidence. Click any cell to see detail.</li>
<li><b>Review cells</b> — In the detail panel, you can:
  <ul>
    <li><b>Accept</b> — Confirm the model's label is correct.</li>
    <li><b>Correct</b> — Select the correct label from the dropdown, then click Correct. The original model label is preserved; your correction is stored as <code>review_label</code>.</li>
    <li><b>Exclude</b> — Remove a cell from the analysis (e.g., artifact, overlapping, poor quality).</li>
    <li><b>Unclassifiable</b> — Mark a cell that cannot be reliably identified.</li>
  </ul>
</li>
<li><b>Chat</b> — Use the chat panel (always visible on the right) to ask questions or give commands in natural language.</li>
<li><b>Generate report</b> — Type <code>report</code> in chat, or expand the Report section in the left panel.</li>
</ol>

<h2>Upload &amp; Run Pipeline</h2>
<p>To analyze a new blood smear image:</p>
<ol>
<li>Expand <b>"Upload &amp; Run Pipeline"</b> in the left panel.</li>
<li><b>Session ID</b> — Give a unique name for this analysis session (e.g., <code>patient_001_slide_A</code>).</li>
<li><b>Image file(s)</b> — Select one or more blood smear images (.tiff, .png, .jpg). You can select multiple files at once. YOLO will auto-detect all WBC in each image and all detected cells are processed through MedSAM and the classifier automatically.</li>
<li><b>Start from stage</b> (Advanced) — If you need to re-run only part of the pipeline:
  <ul>
    <li><code>YOLO (full run)</code> — Run everything from scratch</li>
    <li><code>ROI</code> — Skip YOLO, use existing detections to re-extract ROI patches</li>
    <li><code>MedSAM</code> — Skip YOLO+ROI, re-run segmentation on existing patches</li>
    <li><code>Classifier</code> — Skip YOLO+ROI+MedSAM, re-classify existing masks</li>
    <li><code>Agent only</code> — Re-ingest existing results into DB and regenerate report</li>
  </ul>
</li>
<li>Click <b>"Run Pipeline"</b>. Progress appears in the chat panel. This may take 1-5 minutes depending on image size and number of cells.</li>
</ol>

<h2>Chat Commands</h2>
<table>
<tr><th>Command</th><th>Example</th><th>Action</th></tr>
<tr><td>Summary</td><td><code>summary</code></td><td>Show case summary with cell counts and disease warnings</td></tr>
<tr><td>Uncertain cells</td><td><code>哪些細胞最不確定？</code></td><td>List cells needing review, sorted by uncertainty</td></tr>
<tr><td>Inspect cell</td><td><code>看 det_000006</code></td><td>Show details for a specific cell</td></tr>
<tr><td>Accept</td><td><code>接受 det_000006</code></td><td>Accept the model's label for a cell</td></tr>
<tr><td>Correct</td><td><code>把 det_000006 改成 LYT</code></td><td>Correct a cell's label (model label preserved)</td></tr>
<tr><td>Exclude</td><td><code>det_000003 不要用</code></td><td>Exclude a cell from analysis</td></tr>
<tr><td>Unclassifiable</td><td><code>det_000004 無法分類</code></td><td>Mark as unclassifiable</td></tr>
<tr><td>Report</td><td><code>report</code> / <code>產生報告</code></td><td>Generate morphology-review report</td></tr>
<tr><td>List cells</td><td><code>list cells</code> / <code>細胞</code></td><td>List all cells in the case</td></tr>
</table>

<h2>Cell Types (15 Classes)</h2>
<table>
<tr><th>Code</th><th>Name</th><th>Clinical Note</th></tr>
<tr><td>NGS</td><td>Neutrophil (segmented)</td><td>Most common WBC</td></tr>
<tr><td>NGB</td><td>Neutrophil (band)</td><td>Left shift indicator</td></tr>
<tr><td>LYT</td><td>Lymphocyte (typical)</td><td>Normal mature lymphocyte</td></tr>
<tr><td>LYA</td><td>Lymphocyte (atypical)</td><td>Reactive / ALL blast</td></tr>
<tr><td>MON</td><td>Monocyte</td><td>Elevated in AML-M4/M5</td></tr>
<tr><td>EOS</td><td>Eosinophil</td><td></td></tr>
<tr><td>BAS</td><td>Basophil</td><td>Elevated may suggest CML</td></tr>
<tr><td>EBO</td><td>Erythroblast</td><td>Elevated in AML-M6</td></tr>
<tr><td>MYO</td><td>Myelocyte</td><td>Granulocyte precursor</td></tr>
<tr><td>MMZ</td><td>Metamyelocyte</td><td>Between myelocyte and band</td></tr>
<tr><td>MYB</td><td>Myeloblast</td><td style='color:var(--bad)'>Clinically critical (AML-M1/M2)</td></tr>
<tr><td>PMO</td><td>Promyelocyte</td><td>Large azurophilic granules</td></tr>
<tr><td>PMB</td><td>Promonocyte</td><td style='color:var(--bad)'>APL-relevant (M3), URGENT</td></tr>
<tr><td>MOB</td><td>Monoblast</td><td style='color:var(--bad)'>Clinically critical (AML-M5)</td></tr>
<tr><td>KSC</td><td>Smudge cell</td><td>CLL artifact</td></tr>
</table>

<h2>QC Flags</h2>
<p>Cells are automatically flagged for review when:</p>
<ul>
<li><b>rare_or_immature_class</b> — PMB, MYB, MOB, MMZ, KSC, PMO are always review-required</li>
<li><b>low_classifier_probability</b> — Top prediction confidence &lt; 70%</li>
<li><b>small_top1_top2_margin</b> — Difference between top-1 and top-2 &lt; 15%</li>
<li><b>low_yolo_confidence</b> — YOLO detection confidence &lt; 50%</li>
<li><b>high_entropy</b> — Prediction is spread across many classes</li>
<li><b>high_bbox_overlap</b> — Overlapping detections</li>
</ul>

<h2>Disease Screening Warnings</h2>
<p>The system automatically screens cell distributions for morphological patterns:</p>
<ul>
<li><b>Blast-like cells &ge; 20%</b> — Consider AML workup</li>
<li><b>PMB elevated &ge; 10%</b> — Consider urgent APL (M3) screening</li>
<li><b>MOB elevated &ge; 20%</b> — Profile consistent with M5</li>
<li><b>EBO elevated &ge; 50%</b> — Profile consistent with M6</li>
</ul>

<h2>Important Rules</h2>
<ul>
<li>This system provides <b>morphology-level suggestions only</b>, not final clinical diagnoses.</li>
<li>M0 and M7 subtypes <b>cannot be determined by morphology alone</b> — flow cytometry / immunophenotyping is required.</li>
<li>All rare/immature class predictions are <b>review-required by default</b>.</li>
<li>The model's original label (<code>model_label</code>) is <b>never overwritten</b>. Human corrections are stored separately as <code>review_label</code>.</li>
<li>Reports are marked as <b>"Research draft — not for clinical diagnosis"</b>.</li>
</ul>

<h2>Pipeline Architecture</h2>
<p><code>YOLO detection → ROI extraction → MedSAM segmentation → DinoBloom-B classifier → Agent DB → QA/Review</code></p>
<p>Classifier: DinoBloom-B (ViT-B/14, pretrained on 380K WBC images), macro-F1 = 0.864 on 15-class LMU AML dataset.</p>
</div>
</div><!-- /tutorialPage -->

<script>
let currentSession='demo_case_ngs',currentCase=null,currentCell=null;
const labels=__LABELS__;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function fileUrl(p){return'/api/file?path='+encodeURIComponent(p||'')}
async function api(u,o={}){const r=await fetch(u,o);const t=await r.text();let j;try{j=JSON.parse(t)}catch{throw new Error(t)}if(!r.ok)throw new Error(j.error||t);return j}

/* --- page tabs --- */
function showPage(name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById(name+'Page').classList.add('active');
  document.querySelectorAll('.tab-bar button').forEach((b,i)=>{b.classList.toggle('active',b.textContent.toLowerCase()===name)});
}

/* --- session --- */
async function listSessions(){const d=await api('/api/sessions');const s=document.getElementById('sessionSelect');s.innerHTML='';d.sessions.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;if(n===currentSession)o.selected=true;s.appendChild(o)})}
async function loadSelected(){currentSession=document.getElementById('sessionSelect').value;await loadCase(currentSession)}
async function loadCase(session){try{const d=await api('/api/case?session_id='+encodeURIComponent(session));currentCase=d;currentSession=session;render()}catch(e){alert(e.message)}}

function render(){renderSummary();renderCells();renderReport();if(currentCell){currentCell=(currentCase.cells||[]).find(c=>c.cell_id===currentCell.cell_id);renderDetail()}}

function renderSummary(){const s=currentCase.summary;document.getElementById('summary').innerHTML=[['Total cells',s.total_cells],['Review needed',s.review_needed_count],['Review-ready',s.hard_count_total],['Excluded',s.excluded_count]].map(([k,v])=>`<div class="stat-row"><span>${k}</span><b>${v}</b></div>`).join('')+'<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:3px">'+Object.entries(s.hard_counts||{}).map(([k,v])=>`<span class="chip info">${esc(k)}:${v}</span>`).join('')+'</div>';const w=s.disease_warnings||[];document.getElementById('warnings').innerHTML=w.length?w.map(x=>`<div class="chip warn" style="margin:2px 0">${esc(x)}</div>`).join(''):'<span class="chip ok">OK</span>'}

function renderCells(){const box=document.getElementById('cells');box.innerHTML=(currentCase.cells||[]).map(c=>{const act=currentCell&&currentCell.cell_id===c.cell_id?' active':'';const reasons=(c.review_reasons||[]).map(r=>`<span class="chip warn">${esc(r)}</span>`).join('');const prob=c.top_probability!=null?(c.top_probability*100).toFixed(1)+'%':'';const isGated=!c.downstream_eligible&&c.review_status==='queued_for_review';const st=isGated?'<span class="chip warn">YOLO gated</span>':c.review_status==='accepted_model_label'?'<span class="chip ok">accepted</span>':c.review_status==='corrected'?`<span class="chip info">&rarr;${esc(c.review_label)}</span>`:c.review_status==='excluded'?'<span class="chip" style="color:var(--bad)">excluded</span>':'';const yoloConf=c.yolo_confidence!=null?` YOLO:${(c.yolo_confidence*100).toFixed(0)}%`:'';return`<div class="cell-card${act}" onclick="selectCell('${esc(c.cell_id)}')"><img class="thumb" src="${fileUrl(c.clean_patch_path||c.mask_path)}"/><div><div style="font-weight:600;font-size:12px">${esc(c.model_label||'pending')} <span class="muted">${prob}${isGated?yoloConf:''}</span></div><div class="muted" style="font-size:11px">${esc(c.cell_id)}</div><div style="margin-top:2px">${st}${reasons}</div></div></div>`}).join('')}

function renderReport(){document.getElementById('report').textContent=currentCase.report||''}

function selectCell(id){currentCell=(currentCase.cells||[]).find(c=>c.cell_id===id);renderCells();renderDetail()}

function renderDetail(){
  const d=document.getElementById('detail');const c=currentCell;
  if(!c){d.innerHTML='<span class="muted">Select a cell.</span>';return}
  const probs=c.probabilities||{};
  const bars=Object.entries(probs).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k,v])=>`<div style="display:flex;align-items:center;gap:4px;margin:1px 0"><span style="width:32px;font-size:11px;text-align:right;font-weight:500">${esc(k)}</span><div style="flex:1;height:12px;background:#f3f4f6;border-radius:2px;overflow:hidden"><div style="height:100%;width:${(v*100).toFixed(1)}%;background:${k===c.model_label?'var(--accent)':'#cbd5e1'};border-radius:2px"></div></div><span class="muted" style="width:36px;font-size:10px">${(v*100).toFixed(1)}%</span></div>`).join('');
  /* bbox overlay on original YOLO input image */
  const bbox=c.bbox_xyxy_original;
  const origImg=currentCase.original_image;
  let bboxHtml='';
  if(bbox&&bbox.length===4&&origImg){
    bboxHtml=`<h3>Position on Original Image</h3><canvas id="bboxCanvas" class="bbox-canvas" width="400" height="400"></canvas>`;
  }
  d.innerHTML=`<img class="detail-img" src="${fileUrl(c.clean_patch_path||c.mask_path)}"/>
<div style="font-weight:600;font-size:14px;margin-bottom:3px">${esc(c.cell_id)}</div>
<div style="margin-bottom:2px;font-size:13px">Model: <b>${esc(c.model_label)}</b></div>
<div class="muted" style="margin-bottom:4px">Status: ${esc(c.review_status)}${c.review_label?' &rarr; <b>'+esc(c.review_label)+'</b>':''}</div>
<h3>Top Probabilities</h3>${bars}
<h3>QC Flags</h3>
<div style="margin-bottom:6px">${(c.review_reasons||[]).map(r=>`<span class="chip warn">${esc(r)}</span>`).join('')||'<span class="chip ok">none</span>'}</div>
${bboxHtml}
<h3>Actions</h3>
${!c.downstream_eligible&&c.review_status==='queued_for_review'?`
<div style="background:var(--warn-bg);border:1px solid #fbbf24;border-radius:6px;padding:8px;margin-bottom:6px;font-size:12px">
  <b>YOLO Gated</b>: confidence ${(c.yolo_confidence*100).toFixed(1)}% is below threshold.<br>
  This cell was NOT sent to MedSAM/classifier. Approve to include it in the next pipeline run, or exclude it.
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
<button class="primary" onclick="review('approve_gated')">Approve &amp; Include</button>
<button onclick="review('exclude')" class="danger">Exclude</button>
</div>`:`
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
<button onclick="review('accept')">Accept</button>
<button onclick="review('exclude')" class="danger">Exclude</button>
<button onclick="review('unclassifiable')">Unclassifiable</button>
</div>
<div class="row" style="margin-top:5px"><select id="labelSelect">${labels.map(l=>`<option value="${l}">${l}</option>`).join('')}</select><button class="primary" onclick="review('correct')">Correct</button></div>`}`;
  /* draw bbox on the ORIGINAL YOLO input image */
  if(bbox&&bbox.length===4&&origImg){
    const cvs=document.getElementById('bboxCanvas');
    if(cvs){
      const ctx2=cvs.getContext('2d');
      const img=new Image();
      img.onerror=()=>{ctx2.font='13px sans-serif';ctx2.fillStyle='#999';ctx2.fillText('Original image not available',10,30)};
      img.onload=()=>{
        cvs.width=img.naturalWidth;cvs.height=img.naturalHeight;
        cvs.style.maxWidth='100%';cvs.style.maxHeight='240px';
        ctx2.drawImage(img,0,0);
        /* draw all cell bboxes in grey first */
        (currentCase.cells||[]).forEach(oc=>{
          if(oc.cell_id!==c.cell_id&&oc.bbox_xyxy_original&&oc.bbox_xyxy_original.length===4){
            const ob=oc.bbox_xyxy_original;
            ctx2.strokeStyle='rgba(150,150,150,0.5)';ctx2.lineWidth=1;
            ctx2.strokeRect(ob[0],ob[1],ob[2]-ob[0],ob[3]-ob[1]);
          }
        });
        /* draw selected cell bbox in red */
        ctx2.strokeStyle='#ff3333';ctx2.lineWidth=3;
        ctx2.strokeRect(bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]);
        ctx2.font='bold 14px sans-serif';ctx2.fillStyle='#ff3333';
        const label=c.model_label+' '+(c.top_probability*100).toFixed(0)+'%';
        const ty=bbox[1]>20?bbox[1]-6:bbox[3]+16;
        ctx2.fillText(label,bbox[0],ty);
      };
      img.src=fileUrl(origImg);
    }
  }
}

async function review(action){if(!currentCell){alert('Select a cell first');return}const body={session_id:currentSession,action,cell_id:currentCell.cell_id};if(action==='correct')body.label=document.getElementById('labelSelect').value;const data=await api('/api/review',{method:'POST',body:JSON.stringify(body)});currentCase=data.case;currentCell=(currentCase.cells||[]).find(c=>c.cell_id===body.cell_id);render();renderDetail()}

let isSending=false;
async function sendChat(){if(isSending)return;const msg=document.getElementById('chatInput').value.trim();if(!msg)return;isSending=true;const btn=document.querySelector('#reviewPage .col:last-child button.primary');if(btn){btn.disabled=true;btn.textContent='Sending...'}document.getElementById('chatInput').value='';const log=document.getElementById('chatlog');log.innerHTML+=`<div><b>You:</b> ${esc(msg)}</div>`;log.innerHTML+=`<div class="muted" id="typing">Agent is thinking...</div>`;log.scrollTop=log.scrollHeight;try{const data=await api('/api/chat',{method:'POST',body:JSON.stringify({session_id:currentSession,message:msg})});document.getElementById('typing')?.remove();const tag=data.mode==='qwen'?'<span class="chip ok">Qwen</span>':'<span class="chip">fallback</span>';log.innerHTML+=`<div><b>Agent</b> ${tag}: ${esc(data.answer||'done')}</div>`;if(data.tool_trace&&data.tool_trace.length){log.innerHTML+=`<details><summary class="muted">${data.tool_trace.length} tool call(s)</summary><pre>${esc(JSON.stringify(data.tool_trace,null,2))}</pre></details>`}if(data.case){currentCase=data.case;render()}}catch(e){document.getElementById('typing')?.remove();log.innerHTML+=`<div style="color:var(--bad)"><b>Error:</b> ${esc(e.message)}</div>`}finally{isSending=false;if(btn){btn.disabled=false;btn.textContent='Send'}}log.scrollTop=log.scrollHeight}

/* Enter key to send */
document.getElementById('chatInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat()}});

document.getElementById('uploadForm').addEventListener('submit',async e=>{e.preventDefault();const fd=new FormData(e.target);const sid=fd.get('session_id')||'uploaded_demo';const log=document.getElementById('chatlog');log.innerHTML+='<div class="muted">Running pipeline...</div>';try{const r=await fetch('/api/run_pipeline',{method:'POST',body:fd});const data=await r.json();if(!r.ok)throw new Error(data.error||'pipeline failed');currentSession=data.session_id||sid;await listSessions();if(data.case){currentCase=data.case;render()}log.innerHTML+=`<div class="muted">Pipeline done (code ${data.returncode}).</div>`}catch(err){alert(err.message)}});

async function fetchAgentStatus(){try{const s=await api('/api/agent_status');const el=document.getElementById('agentStatus');if(s.agent_mode==='qwen'&&s.qwen_available){el.textContent='Qwen3-14B'+(s.qwen_loaded?' (ready)':' (lazy)');el.style.color='#1e8449'}else{el.textContent='Fallback mode';el.style.color='#5d6d7e'}}catch{}}
fetchAgentStatus();
listSessions().then(()=>loadCase(currentSession)).catch(()=>{
  document.getElementById('summary').innerHTML='<span class="muted">No sessions found. Upload an image to start.</span>';
});
</script>
</body>
</html>""".replace("__LABELS__", json.dumps(LABELS))


class Handler(BaseHTTPRequestHandler):
    ctx: DemoContext

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/":
                text_response(self, HTML)
            elif parsed.path == "/api/sessions":
                sessions = sorted(p.name for p in self.ctx.output_root.iterdir() if (p / "ymca_agent.db").exists())
                json_response(self, {"sessions": sessions})
            elif parsed.path == "/api/agent_status":
                mode = self.ctx.agent_mode
                qwen_loaded = self.ctx.qwen_agent is not None and self.ctx.qwen_agent._model is not None
                json_response(self, {
                    "agent_mode": mode,
                    "qwen_available": self.ctx.qwen_agent is not None,
                    "qwen_loaded": qwen_loaded,
                })
            elif parsed.path == "/api/case":
                session_id = safe_session_id(qs.get("session_id", ["demo_case_ngs"])[0])
                json_response(self, load_case(self.ctx, session_id))
            elif parsed.path == "/api/file":
                raw = unquote(qs.get("path", [""])[0])
                path = Path(raw)
                if not raw or not path.exists() or not self.ctx.allowed_file(path):
                    json_response(self, {"error": "file not found or not allowed"}, 404)
                    return
                # Browsers can't render TIFF — convert to PNG on the fly
                if path.suffix.lower() in {".tiff", ".tif"}:
                    if path.stat().st_size > 100_000_000:
                        json_response(self, {"error": "Image too large (>100MB)"}, 413)
                        return
                    from PIL import Image as PILImage
                    import io
                    with PILImage.open(path) as img:
                        # Handle 16-bit grayscale (common in medical imaging)
                        if img.mode in ("I;16", "I"):
                            import numpy as np
                            arr = np.array(img, dtype=np.float64)
                            lo, hi = arr.min(), arr.max()
                            arr = ((arr - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
                            img = PILImage.fromarray(arr)
                        img = img.convert("RGB")
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        data = buf.getvalue()
                    ctype = "image/png"
                else:
                    data = path.read_bytes()
                    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                json_response(self, {"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            json_response(self, {"error": str(exc)}, 500)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/review":
                payload = parse_json_body(self)
                session_id = safe_session_id(payload.get("session_id", "demo_case_ngs"))
                json_response(self, apply_review(self.ctx, session_id, payload))
            elif parsed.path == "/api/chat":
                payload = parse_json_body(self)
                session_id = safe_session_id(payload.get("session_id", "demo_case_ngs"))
                json_response(self, handle_chat(self.ctx, session_id, payload.get("message", "")))
            elif parsed.path == "/api/run_pipeline":
                body = read_body(self)
                fields, files = parse_multipart(body, self.headers.get("Content-Type", ""))
                result = run_pipeline(self.ctx, fields, files)
                json_response(self, result, status=200 if result.get("returncode") == 0 else 500)
            else:
                json_response(self, {"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            json_response(self, {"error": str(exc)}, 500)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[demo-ui] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve YMCA local morphology-review demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--upload-root", type=Path, default=DEFAULT_UPLOAD_ROOT)
    parser.add_argument("--guidelines-dir", type=Path, default=DEFAULT_GUIDELINES)
    parser.add_argument("--classifier-ckpt", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--yolo-python", default=DEFAULT_YOLO_PYTHON)
    parser.add_argument("--medsam-python", default=DEFAULT_MEDSAM_PYTHON)
    parser.add_argument("--classifier-python", default=DEFAULT_CLASSIFIER_PYTHON)
    parser.add_argument("--agent-mode", choices=["qwen", "fallback"], default="fallback",
                        help="Chat agent mode: 'qwen' for Qwen3-14B tool agent, 'fallback' for deterministic router.")
    parser.add_argument("--qwen-model-path", default=os.environ.get("YMCA_QWEN_MODEL_PATH"),
                        help="Path to local Qwen3-14B model directory.")
    parser.add_argument("--qwen-load-in-4bit", action="store_true", default=True,
                        help="Load Qwen in 4-bit quantization (default: True).")
    parser.add_argument("--no-qwen-4bit", dest="qwen_load_in_4bit", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Handler.ctx = DemoContext(
        args.output_root, args.upload_root, args.guidelines_dir, args.classifier_ckpt,
        args.yolo_python, args.medsam_python, args.classifier_python,
        agent_mode=args.agent_mode, qwen_model_path=args.qwen_model_path,
        qwen_load_in_4bit=args.qwen_load_in_4bit,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"YMCA demo UI: http://{args.host}:{args.port}")
    print(f"Output root: {args.output_root}")
    server.serve_forever()


if __name__ == "__main__":
    main()
