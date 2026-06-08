"""Backend-controlled tools exposed to the local conversation agent."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from .guidelines import load_reporting_guidelines, validate_report_safety
from .model_contracts import classifier_result_to_cell_fields, medsam_summary_to_cell_fields, yolo_detection_to_cell_fields
from .disease_cell_validation import screen_cell_profile
from .qc import BLAST_LIKE_LABELS, RARE_CLASSES, review_reasons, uncertainty_score
from .storage import connect, init_db

# Canonical labels accepted for review_label validation.
# Matches the 16-class task_combine classifier label set.
CANONICAL_LABELS = frozenset({
    "apl_suspect", "artifact", "basophil", "early_pre_b", "eosinophil",
    "erythroid", "hematogone", "mature_lymphocyte", "monoblast", "monocyte",
    "myeloblast", "myelocyte", "neutrophil", "other_immature", "pre_b", "pro_b",
})

REVIEW_BLOCKING_STATUSES = {"queued_for_review", "needs_senior_review", "unclassifiable", "excluded"}
REVIEW_RESOLVED_STATUSES = {"accepted_model_label", "corrected", "excluded", "unclassifiable"}


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _meaningful_guideline_items(items: list[str]) -> list[str]:
    """Drop YAML scaffolding/blank placeholders from lightweight guideline line loading."""
    ignored_exact = {
        "critical_flags:",
        "review_triggers:",
        "severity_levels:",
        'id: ""',
        'label: ""',
        'severity: ""',
        'criteria: ""',
        'required_action: ""',
        'approved_wording: ""',
        'notify_role: ""',
        'notes: ""',
    }
    meaningful: list[str] = []
    for item in items:
        stripped = item.strip()
        if not stripped or stripped in ignored_exact:
            continue
        if stripped.endswith(':') and len(stripped.split()) == 1:
            continue
        meaningful.append(stripped)
    return meaningful


def _fill_zh_template(
    case_id: str,
    summary: dict[str, Any],
    guidelines: Any,
    guidelines_dir: Path,
) -> str:
    """Fill the Chinese report template with available DB data.

    Fields knowable from the pipeline are filled; clinical fields (specimen type,
    dates, history, treatment) are left as '—' for the clinician to complete.
    """
    template_path = guidelines_dir / "report_template_zh.md"
    if template_path.exists():
        raw = template_path.read_text()
        # Extract text inside the first ```text ... ``` block
        import re as _re
        m = _re.search(r"```text\s*(.*?)```", raw, _re.DOTALL)
        template_body = m.group(1).strip() if m else raw
    else:
        template_body = (
            "I. Specimen Information\n- Specimen type:\n- Collection date/time:\n- Received date/time:\n- Examination:\n\n"
            "II. Clinical Data\n- Clinical diagnosis/question:\n- Relevant history:\n- Recent treatment or tests:\n\n"
            "III. Findings\n- Peripheral blood morphology:\n- Bone marrow smear/biopsy findings:\n- Special stains / flow cytometry / molecular or cytogenetics:\n\n"
            "IV. Interpretation\n- Main morphological findings:\n- Relevance to clinical question:\n- Items to exclude or confirm:\n\n"
            "V. Conclusion/Opinion\n- Conclusion:\n- Recommendation:\n- Urgent review or notification required:"
        )

    # Build per-field fill values from summary
    hard_counts: dict[str, int] = summary.get("hard_counts", {})
    disease_warnings: list[str] = summary.get("disease_warnings", [])
    total_cells: int = summary.get("total_cells", 0)
    hard_total: int = summary.get("hard_count_total", 0)
    review_needed: int = summary.get("review_needed_count", 0)
    excluded: int = summary.get("excluded_count", 0)
    unclassifiable: int = summary.get("unclassifiable_count", 0)
    blast_like_ratio: float = summary.get("blast_like_ratio") or 0.0

    # III: bone marrow smear findings — cell distribution table
    if hard_counts:
        count_lines = ["(Morphology screening result — not a complete differential count)"]
        for label, n in sorted(hard_counts.items(), key=lambda x: -x[1]):
            pct = 100 * n / hard_total if hard_total else 0
            count_lines.append(f"  {label}: {n} ({pct:.1f}%)")
        count_lines.append(f"  Total cells: {total_cells} (confirmed: {hard_total}, pending review: {review_needed}, excluded: {excluded}, unclassifiable: {unclassifiable})")
        bone_marrow_val = "\n".join(count_lines)
    else:
        bone_marrow_val = "No confirmed cell labels available"

    # IV: main morphological findings
    if disease_warnings:
        main_finding = "; ".join(disease_warnings)
    elif hard_counts:
        top = sorted(hard_counts.items(), key=lambda x: -x[1])[:3]
        main_finding = "Predominant cell types: " + ", ".join(f"{l} ({n})" for l, n in top)
    else:
        main_finding = "—"

    # IV: items to exclude or confirm
    if review_needed > 0:
        exclude_note = f"{review_needed} cell(s) still pending manual review; recommend human confirmation before issuing final report"
    else:
        exclude_note = "All cells have completed initial screening; recommend correlation with other clinical tests"

    # V: conclusion
    if blast_like_ratio > 0:
        conclusion = f"Morphology screening indicates suspected blast-like cell ratio ~{blast_like_ratio*100:.1f}%; recommend clinical review. (Research draft — not for clinical diagnosis)"
    else:
        conclusion = "Morphology screening complete; no significant blast-like proportion detected. (Research draft — not for clinical diagnosis)"

    # V: recommendation
    recommendation = "Recommend correlation with CBC, flow cytometry, and molecular/cytogenetic testing; if morphology is abnormal, refer to a hematopathologist for review"

    # V: urgent review
    critical = _meaningful_guideline_items(guidelines.critical_flags)
    if disease_warnings or blast_like_ratio > 0.2:
        urgent = "Yes — suspected high-risk morphology detected; recommend urgent review"
    elif critical:
        urgent = f"Depends on clinical context — see: {critical[0]}"
    else:
        urgent = "No — no significant urgent flags in current screening"

    # Field substitution map: field prefix (after "- ") → value
    fill_map = {
        "Examination:": f"Case ID: {case_id}",
        "Bone marrow smear/biopsy findings:": bone_marrow_val,
        "Main morphological findings:": main_finding,
        "Items to exclude or confirm:": exclude_note,
        "Conclusion:": conclusion,
        "Recommendation:": recommendation,
        "Urgent review or notification required:": urgent,
    }

    out_lines: list[str] = []
    for line in template_body.splitlines():
        stripped = line.lstrip("- ").strip()
        matched = False
        for prefix, value in fill_map.items():
            if stripped.startswith(prefix):
                indent = "- " if line.lstrip().startswith("-") else ""
                # Multi-line values: indent continuation lines
                value_lines = value.splitlines()
                out_lines.append(f"{indent}{prefix}{value_lines[0]}")
                for vl in value_lines[1:]:
                    out_lines.append(f"    {vl}")
                matched = True
                break
        if not matched:
            # Leave blank fields as "—"
            if (line.endswith(":") or line.endswith("：")) and "- " in line:
                out_lines.append(line + " —")
            else:
                out_lines.append(line)

    header = "[Research draft — not for clinical diagnosis]\n"
    return header + "\n".join(out_lines)


def _row_to_cell(row: Any) -> dict[str, Any]:
    probabilities = _loads(row["probabilities_json"], {})
    rare_class = row["model_label"] in RARE_CLASSES if row["model_label"] else False
    downstream_eligible = bool(row["downstream_eligible"]) if row["downstream_eligible"] is not None else True
    reasons = review_reasons(
        yolo_confidence=row["yolo_confidence"],
        segmentation_quality=row["segmentation_quality"],
        segmentation_status=row["segmentation_status"],
        top_probability=row["top_probability"],
        top2_probability=row["top2_probability"],
        probabilities=probabilities,
        overlap_score=row["overlap_score"],
        rare_class=rare_class,
        downstream_eligible=downstream_eligible,
    )
    score = uncertainty_score(
        yolo_confidence=row["yolo_confidence"],
        segmentation_quality=row["segmentation_quality"],
        top_probability=row["top_probability"],
        top2_probability=row["top2_probability"],
        probabilities=probabilities,
        overlap_score=row["overlap_score"],
    )
    data = dict(row)
    data["bbox_xyxy_original"] = _loads(row["bbox_xyxy_original"], row["bbox_xyxy_original"])
    data["roi_xyxy_original"] = _loads(row["roi_xyxy_original"], row["roi_xyxy_original"])
    data["probabilities"] = probabilities
    data["review_reasons"] = reasons
    data["uncertainty_score"] = score
    data["qc_passed"] = not reasons
    return data


class AgentTools:
    def __init__(
        self,
        db_path: str | Path = "ymca_agent.db",
        guidelines_dir: str | Path = "reporting_guidelines",
    ) -> None:
        self.db_path = Path(db_path)
        self.guidelines_dir = Path(guidelines_dir)
        init_db(self.db_path)


    def start_conversation(self, conversation_id: str, user_id: str | None = None) -> dict[str, Any]:
        """Create a conversation row and its state if they do not exist.

        If the conversation already exists, an optional user_id must match the
        stored owner. This is the MVP access-control boundary for local demos.
        """
        # NOTE: user_id=None bypasses ownership check — permitted for local single-user MVP only.
        # Must be hardened before any multi-user or production deployment.
        if user_id is None:
            warnings.warn(
                "start_conversation called with user_id=None: ownership check bypassed. "
                "Not safe for multi-user or production use.",
                stacklevel=2,
            )
        with connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT user_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if existing:
                stored_user_id = existing["user_id"]
                if user_id is not None and stored_user_id is not None and user_id != stored_user_id:
                    raise PermissionError("conversation does not belong to this user")
            else:
                conn.execute(
                    "INSERT INTO conversations (conversation_id, user_id) VALUES (?, ?)",
                    (conversation_id, user_id),
                )
            conn.execute(
                "INSERT OR IGNORE INTO conversation_state (conversation_id, state_json) VALUES (?, ?)",
                (conversation_id, "{}"),
            )
        return self.get_conversation_state(conversation_id, user_id=user_id)

    def record_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist one chat or tool message for audit/recovery."""
        if role not in {"user", "agent", "tool", "system"}:
            raise ValueError("role must be one of: user, agent, tool, system")
        self.start_conversation(conversation_id)
        if case_id is not None:
            self.assert_case_access(conversation_id, case_id)
        with connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, case_id, role, content) VALUES (?, ?, ?, ?)",
                (conversation_id, case_id, role, content),
            )
        return {"message_id": cur.lastrowid, "conversation_id": conversation_id, "case_id": case_id, "role": role}

    def set_active_case(self, conversation_id: str, case_id: str, user_id: str | None = None) -> dict[str, Any]:
        """Bind a conversation to the one full-image case currently being reviewed."""
        case = self.get_case(case_id)
        state = self.start_conversation(conversation_id, user_id=user_id)
        conversation_user_id = self._get_conversation_user_id(conversation_id)
        if conversation_user_id is not None and case.get("user_id") is not None and conversation_user_id != case["user_id"]:
            raise PermissionError("case does not belong to this conversation user")
        if state.get("active_case_id") not in {None, case_id}:
            raise PermissionError("conversation already has a different active case")
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE conversation_state
                SET active_case_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (case_id, conversation_id),
            )
        return self.get_conversation_state(conversation_id, user_id=user_id)

    def _get_conversation_user_id(self, conversation_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT user_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return None if row is None else row["user_id"]

    def get_conversation_state(self, conversation_id: str, user_id: str | None = None) -> dict[str, Any]:
        """Return the active case/cell/report context for a conversation."""
        # NOTE: user_id=None bypasses ownership check — permitted for local single-user MVP only.
        # Must be hardened before any multi-user or production deployment.
        if user_id is None:
            warnings.warn(
                "get_conversation_state called with user_id=None: ownership check bypassed. "
                "Not safe for multi-user or production use.",
                stacklevel=2,
            )
        stored_user_id = self._get_conversation_user_id(conversation_id)
        if user_id is not None and stored_user_id is not None and user_id != stored_user_id:
            raise PermissionError("conversation does not belong to this user")
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM conversation_state WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                return {"conversation_id": conversation_id, "active_case_id": None, "state": {}}
        data = dict(row)
        data["state"] = _loads(row["state_json"], {})
        return data

    def get_active_case(self, conversation_id: str, user_id: str | None = None) -> dict[str, Any]:
        """Resolve the case for a single-image conversation."""
        state = self.get_conversation_state(conversation_id, user_id=user_id)
        case_id = state.get("active_case_id")
        if not case_id:
            raise KeyError(f"conversation has no active case: {conversation_id}")
        self.assert_case_access(conversation_id, case_id, user_id=user_id)
        return self.get_case(case_id)

    def assert_case_access(self, conversation_id: str, case_id: str, user_id: str | None = None) -> None:
        """Ensure a case belongs to the active one-image conversation scope."""
        # NOTE: user_id=None bypasses ownership check — permitted for local single-user MVP only.
        # Must be hardened before any multi-user or production deployment.
        if user_id is None and case_id is not None:
            warnings.warn(
                "assert_case_access called with user_id=None: ownership check bypassed. "
                "Not safe for multi-user or production use.",
                stacklevel=2,
            )
        state = self.get_conversation_state(conversation_id, user_id=user_id)
        active_case_id = state.get("active_case_id")
        if active_case_id != case_id:
            raise PermissionError("case is not active for this conversation")
        case = self.get_case(case_id)
        conversation_user_id = self._get_conversation_user_id(conversation_id)
        if conversation_user_id is not None and case.get("user_id") is not None and conversation_user_id != case["user_id"]:
            raise PermissionError("case does not belong to this conversation user")

    def get_case(self, case_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            case = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if not case:
                raise KeyError(f"case not found: {case_id}")
            counts = conn.execute("SELECT COUNT(*) AS n FROM cells WHERE case_id = ? AND is_current = 1", (case_id,)).fetchone()
        data = dict(case)
        data["cell_count"] = counts["n"]
        return data

    def list_cells(self, case_id: str, label: str | None = None, review_status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cells WHERE case_id = ? AND is_current = 1"
        args: list[Any] = [case_id]
        if label:
            query += " AND (model_label = ? OR review_label = ?)"
            args.extend([label, label])
        if review_status:
            query += " AND review_status = ?"
            args.append(review_status)
        query += " ORDER BY cell_id"
        with connect(self.db_path) as conn:
            rows = conn.execute(query, args).fetchall()
        return [_row_to_cell(row) for row in rows]

    def get_cell(self, cell_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM cells WHERE cell_id = ? AND is_current = 1", (cell_id,)).fetchone()
            if not row:
                raise KeyError(f"cell not found: {cell_id}")
        return _row_to_cell(row)

    def list_uncertain_cells(self, case_id: str, label: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        cells = self.list_cells(case_id, label=label)
        unresolved_statuses = {"unreviewed", "queued_for_review", "needs_senior_review"}
        uncertain = [
            cell
            for cell in cells
            if cell["review_status"] in unresolved_statuses
            and (cell["review_reasons"] or cell["review_status"] in {"queued_for_review", "needs_senior_review"})
        ]
        uncertain.sort(key=lambda c: c["uncertainty_score"], reverse=True)
        return uncertain[:limit]

    def update_cell_review(
        self,
        cell_id: str,
        review_label: str | None = None,
        review_status: str = "corrected",
        note: str | None = None,
        reviewer_id: str | None = None,
    ) -> dict[str, Any]:
        valid_statuses = {
            "unreviewed",
            "queued_for_review",
            "accepted_model_label",
            "corrected",
            "unclassifiable",
            "excluded",
            "needs_senior_review",
        }
        if review_status not in valid_statuses:
            raise ValueError(f"invalid review_status: {review_status}")
        if review_status == "corrected" and not review_label:
            raise ValueError("review_label is required when review_status='corrected'")
        if review_label is not None and review_label not in CANONICAL_LABELS:
            raise ValueError(f"unknown review_label: {review_label!r}; must be one of {sorted(CANONICAL_LABELS)}")
        if review_status == "accepted_model_label" and review_label is not None:
            raise ValueError("review_label should be None when accepting the model label")
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM cells WHERE cell_id = ? AND is_current = 1", (cell_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"cell not found: {cell_id}")
            before = dict(row)
            if review_status == "accepted_model_label" and before.get("model_label") is None:
                raise ValueError(
                    f"cannot accept model label for cell {cell_id}: model_label is not set"
                )
            conn.execute(
                """
                UPDATE cells
                SET review_status = ?, review_label = ?, review_note = ?, reviewer_id = ?, reviewed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE cell_id = ?
                """,
                (review_status, review_label, note, reviewer_id, cell_id),
            )
            conn.execute(
                """
                INSERT INTO review_events (cell_id, previous_review_status, previous_review_label, new_review_status, new_review_label, note, reviewer_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cell_id, before["review_status"], before["review_label"], review_status, review_label, note, reviewer_id),
            )
            after = dict(conn.execute(
                "SELECT * FROM cells WHERE cell_id = ?", (cell_id,)
            ).fetchone())
        return {"before": before, "after": after}

    def deactivate_cell(self, cell_id: str, note: str | None = None, reviewer_id: str | None = None) -> dict[str, Any]:
        """Soft-delete a cell by setting is_current=0 and recording a review_event."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM cells WHERE cell_id = ? AND is_current = 1", (cell_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"cell not found: {cell_id}")
            before = dict(row)
            conn.execute(
                "UPDATE cells SET is_current = 0, updated_at = CURRENT_TIMESTAMP WHERE cell_id = ?",
                (cell_id,),
            )
            conn.execute(
                """
                INSERT INTO review_events (cell_id, previous_review_status, previous_review_label, new_review_status, new_review_label, note, reviewer_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cell_id, before["review_status"], before["review_label"], before["review_status"], None, note, reviewer_id),
            )
        return {"cell_id": cell_id, "is_current": 0, "previous_review_status": before["review_status"]}

    def import_yolo_detection(
        self,
        case_id: str,
        detection: dict[str, Any],
        *,
        cell_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or update a cell candidate from a YOLO detection manifest record."""
        self.get_case(case_id)
        fields = yolo_detection_to_cell_fields(detection)
        resolved_cell_id = cell_id or fields["detection_id"]
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO cells (
                    cell_id, case_id, detection_id, bbox_xyxy_original,
                    yolo_class_id, yolo_class_name, downstream_eligible,
                    yolo_confidence, clean_patch_path, roi_image_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cell_id) DO UPDATE SET
                    detection_id = excluded.detection_id,
                    bbox_xyxy_original = excluded.bbox_xyxy_original,
                    yolo_class_id = excluded.yolo_class_id,
                    yolo_class_name = excluded.yolo_class_name,
                    downstream_eligible = excluded.downstream_eligible,
                    yolo_confidence = excluded.yolo_confidence,
                    clean_patch_path = excluded.clean_patch_path,
                    roi_image_path = excluded.roi_image_path,
                    is_current = 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    resolved_cell_id,
                    case_id,
                    fields["detection_id"],
                    fields["bbox_xyxy_original"],
                    fields["yolo_class_id"],
                    fields["yolo_class_name"],
                    fields["downstream_eligible"],
                    fields["yolo_confidence"],
                    fields["clean_patch_path"],
                    fields["roi_image_path"],
                ),
            )
        return self.get_cell(resolved_cell_id)

    def apply_medsam_result(
        self,
        cell_id: str,
        medsam_result: dict[str, Any],
        *,
        preprocess_version: str = "medsam3_lisc_wbc_v1",
    ) -> dict[str, Any]:
        """Attach MedSAM summary output to an existing cell candidate."""
        before = self.get_cell(cell_id)
        fields = medsam_summary_to_cell_fields(medsam_result, preprocess_version=preprocess_version)
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE cells
                SET mask_path = ?,
                    clean_patch_path = COALESCE(?, clean_patch_path),
                    segmentation_status = ?,
                    segmentation_quality = ?,
                    preprocess_version = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE cell_id = ?
                """,
                (
                    fields["mask_path"],
                    fields["clean_patch_path"],
                    fields["segmentation_status"],
                    fields["segmentation_quality"],
                    fields["preprocess_version"],
                    cell_id,
                ),
            )
        after = self.get_cell(cell_id)
        return {"before": before, "after": after, "medsam_status": fields["medsam_status"]}

    def apply_classifier_result(
        self,
        cell_id: str,
        classifier_result: dict[str, Any],
        *,
        classifier_checkpoint: str,
        label_map_version: str = "classifier_flat16_v1",
        preprocess_version: str = "convnet_224_imagenet_v1",
    ) -> dict[str, Any]:
        """Attach ConvNet output to an existing cell candidate."""
        before = self.get_cell(cell_id)
        if before.get("model_label") is not None:
            raise RuntimeError(
                f"model_label already set for cell {cell_id}: cannot overwrite without explicit reset"
            )
        fields = classifier_result_to_cell_fields(
            classifier_result,
            classifier_checkpoint=classifier_checkpoint,
            label_map_version=label_map_version,
            preprocess_version=preprocess_version,
        )
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE cells
                SET clean_patch_path = ?,
                    model_label = ?,
                    top_probability = ?,
                    top2_label = ?,
                    top2_probability = ?,
                    probability_margin = ?,
                    probabilities_json = ?,
                    classifier_checkpoint = ?,
                    label_map_version = ?,
                    preprocess_version = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE cell_id = ?
                """,
                (
                    fields["clean_patch_path"],
                    fields["model_label"],
                    fields["top_probability"],
                    fields["top2_label"],
                    fields["top2_probability"],
                    fields["probability_margin"],
                    fields["probabilities_json"],
                    fields["classifier_checkpoint"],
                    fields["label_map_version"],
                    fields["preprocess_version"],
                    cell_id,
                ),
            )
        after = self.get_cell(cell_id)
        return {"before": before, "after": after}

    def summarize_case(self, case_id: str, use_review_labels: bool = True) -> dict[str, Any]:
        cells = self.list_cells(case_id)
        hard_counts: Counter[str] = Counter()
        model_counts: Counter[str] = Counter()
        review_needed = []
        excluded = []
        unclassifiable = []
        for cell in cells:
            if cell["model_label"]:
                model_counts[cell["model_label"]] += 1
            status = cell["review_status"]
            if status == "excluded":
                excluded.append(cell["cell_id"])
                continue
            if status == "unclassifiable":
                unclassifiable.append(cell["cell_id"])
                continue
            if status == "needs_senior_review":
                review_needed.append(cell["cell_id"])
                continue
            accepted_label = cell["review_label"] if use_review_labels and cell["review_label"] else cell["model_label"]
            if cell["review_reasons"] and status not in {"accepted_model_label", "corrected"}:
                review_needed.append(cell["cell_id"])
                continue
            if accepted_label:
                hard_counts[accepted_label] += 1
        hard_total = sum(hard_counts.values())
        percentages = {label: round(count / hard_total * 100, 2) for label, count in hard_counts.items()} if hard_total else {}
        blast_like = sum(hard_counts.get(label, 0) for label in BLAST_LIKE_LABELS)
        blast_like_ratio = round(blast_like / hard_total, 4) if hard_total else None
        disease_warnings = screen_cell_profile(dict(hard_counts))
        return {
            "case_id": case_id,
            "total_cells": len(cells),
            "hard_count_total": hard_total,
            "hard_counts": dict(hard_counts),
            "hard_percentages": percentages,
            "model_counts_raw": dict(model_counts),
            "review_needed_count": len(review_needed),
            "review_needed_cell_ids": review_needed,
            "excluded_count": len(excluded),
            "unclassifiable_count": len(unclassifiable),
            "blast_like_ratio": blast_like_ratio,
            "disease_warnings": disease_warnings,
            "interpretation_note": "Hard count uses review_label when available and excludes unresolved QC-failed cells.",
        }

    def generate_case_report(self, case_id: str) -> dict[str, Any]:
        summary = self.summarize_case(case_id)
        guidelines = load_reporting_guidelines(self.guidelines_dir)
        content = _fill_zh_template(case_id, summary, guidelines, Path(self.guidelines_dir))
        safety = validate_report_safety(content, guidelines)
        if not safety["safe"]:
            raise ValueError(f"report violates prohibited claims: {safety['violations']}")
        with connect(self.db_path) as conn:
            cur = conn.execute("INSERT INTO reports (case_id, content) VALUES (?, ?)", (case_id, content))
            report_id = cur.lastrowid
        return {
            "report_id": report_id,
            "case_id": case_id,
            "content": content,
            "safety": safety,
            "guidelines_loaded": {
                "allowed_phrases": len(guidelines.allowed_phrases),
                "prohibited_claims": len(guidelines.prohibited_claims),
                "review_triggers": len(guidelines.review_triggers),
                "critical_flags": len(guidelines.critical_flags),
            },
        }
