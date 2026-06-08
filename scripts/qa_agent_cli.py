#!/usr/bin/env python3
"""Small DB-backed QA CLI for the YMCA morphology-review agent.

This is not the final Qwen chat UI. It is the deterministic tool layer that a
local LLM should call: summarize_case, generate_case_report, list_uncertain_cells,
get_cell, and review updates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ymca_agent.tools import AgentTools  # noqa: E402

DEFAULT_GUIDELINES = WORKSPACE_ROOT / "reporting_guidelines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask simple QA questions over an agent case DB.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--guidelines-dir", type=Path, default=DEFAULT_GUIDELINES)
    parser.add_argument("--question", default=None,
                        help="Natural-ish shortcut. Examples: summary, report, uncertain, cell det_000001")
    parser.add_argument("--action", choices=["summary", "report", "uncertain", "cell", "cells", "accept", "correct", "exclude", "unclassifiable"], default=None)
    parser.add_argument("--cell-id", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--note", default=None)
    parser.add_argument("--reviewer-id", default="local_reviewer")
    return parser.parse_args()


LABEL_PATTERN = r"[A-Za-z][A-Za-z0-9_/-]*"
CELL_PATTERN = r"(?:det|cell)[A-Za-z0-9_-]*"


def infer_intent(question: str | None) -> dict[str, str | None]:
    """Parse explicit review/QA shortcuts from a chat-like sentence.

    This is intentionally conservative. Ambiguous free text should be handled by
    the future Qwen layer, which can call this CLI/tool with explicit arguments.
    """
    if not question:
        return {"action": "summary", "cell_id": None, "label": None}
    q = question.strip()
    lower = q.lower()
    parts = q.split()

    cell_match = re.search(CELL_PATTERN, q, flags=re.IGNORECASE)
    cell_id = cell_match.group(0) if cell_match else None

    # Examples:
    #   把 det_000001 改成 LYT
    #   det_000001 改為 LYT
    #   correct det_000001 to LYT
    #   set cell_001 as PMO
    correct_patterns = [
        rf"(?:把\s*)?({CELL_PATTERN})\s*(?:改成|改為|改到|標成|標為)\s*({LABEL_PATTERN})",
        rf"(?:correct|set|change)\s+({CELL_PATTERN})\s+(?:to|as)\s+({LABEL_PATTERN})",
    ]
    for pattern in correct_patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            return {"action": "correct", "cell_id": match.group(1), "label": match.group(2).upper()}

    # Bulk-accept: "accept all", "confirm all", "confirm everything", etc.
    if re.search(r"\b(accept|confirm)\s+(all|everything)\b", lower) and not cell_id:
        return {"action": "accept_all", "cell_id": None, "label": None}

    if cell_id and ("接受" in q or "accept" in lower or "同意" in q or "正確" in q):
        return {"action": "accept", "cell_id": cell_id, "label": None}

    if cell_id and ("不要用" in q or "排除" in q or "exclude" in lower or "remove" in lower):
        return {"action": "exclude", "cell_id": cell_id, "label": None}

    if cell_id and ("無法分類" in q or "不能分類" in q or "unclassifiable" in lower):
        return {"action": "unclassifiable", "cell_id": cell_id, "label": None}

    if lower.startswith("cell ") and len(parts) >= 2:
        return {"action": "cell", "cell_id": parts[1], "label": None}
    if cell_id and ("看" in q or "show" in lower or "inspect" in lower):
        return {"action": "cell", "cell_id": cell_id, "label": None}
    if "uncertain" in lower or "review" in lower or "不確定" in q or "複核" in q:
        return {"action": "uncertain", "cell_id": None, "label": None}
    if "report" in lower or "報告" in q:
        return {"action": "report", "cell_id": None, "label": None}
    if "list" in lower or "cells" in lower or "細胞" in q:
        return {"action": "cells", "cell_id": None, "label": None}
    return {"action": "summary", "cell_id": None, "label": None}


def main() -> None:
    args = parse_args()
    tools = AgentTools(args.db, guidelines_dir=args.guidelines_dir)
    action = args.action
    inferred = {"action": action, "cell_id": None, "label": None}
    if action is None:
        inferred = infer_intent(args.question)
        action = inferred["action"]
    cell_id = args.cell_id or inferred.get("cell_id")
    label = args.label or inferred.get("label")

    if action == "summary":
        print(json.dumps(tools.summarize_case(args.case_id), indent=2, ensure_ascii=False))
    elif action == "report":
        report = tools.generate_case_report(args.case_id)
        print(report["content"])
    elif action == "uncertain":
        cells = tools.list_uncertain_cells(args.case_id, label=args.label, limit=args.limit)
        print(json.dumps(cells, indent=2, ensure_ascii=False))
    elif action == "cells":
        cells = tools.list_cells(args.case_id, label=args.label)
        print(json.dumps(cells[: args.limit], indent=2, ensure_ascii=False))
    elif action == "cell":
        if not cell_id:
            raise SystemExit("--cell-id is required for cell lookup")
        print(json.dumps(tools.get_cell(cell_id), indent=2, ensure_ascii=False))
    elif action == "accept":
        if not cell_id:
            raise SystemExit("--cell-id is required for accept")
        print(json.dumps(tools.update_cell_review(cell_id, review_status="accepted_model_label", note=args.note, reviewer_id=args.reviewer_id), indent=2, ensure_ascii=False))
    elif action == "correct":
        if not cell_id or not label:
            raise SystemExit("--cell-id and --label are required for correct")
        print(json.dumps(tools.update_cell_review(cell_id, review_label=label, review_status="corrected", note=args.note, reviewer_id=args.reviewer_id), indent=2, ensure_ascii=False))
    elif action == "exclude":
        if not cell_id:
            raise SystemExit("--cell-id is required for exclude")
        print(json.dumps(tools.update_cell_review(cell_id, review_status="excluded", note=args.note, reviewer_id=args.reviewer_id), indent=2, ensure_ascii=False))
    elif action == "unclassifiable":
        if not cell_id:
            raise SystemExit("--cell-id is required for unclassifiable")
        print(json.dumps(tools.update_cell_review(cell_id, review_status="unclassifiable", note=args.note, reviewer_id=args.reviewer_id), indent=2, ensure_ascii=False))
    else:
        raise SystemExit(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
