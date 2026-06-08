"""Guideline loading and report-safety helpers for the local agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        lines.append(line)
    return lines


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text().strip()


def _parse_yaml_triggers(path: Path) -> list[dict[str, Any]]:
    """Parse review_triggers.yaml or critical_flags.yaml into a list of dicts."""
    if not path.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            return []
        # Support both "review_triggers" and "critical_flags" top-level keys
        for key in ("review_triggers", "critical_flags"):
            if key in data and isinstance(data[key], list):
                return [e for e in data[key] if isinstance(e, dict) and e.get("id")]
    except Exception:
        pass
    return []


def _parse_abbreviation_map(path: Path) -> dict[str, str]:
    """Parse cell_abbreviation_canonical_map.md table → {abbrev: full_term}."""
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in path.read_text().splitlines():
        # Match markdown table rows: | abbrev | full term | ... |
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2 and parts[0] not in ("Canonical abbreviation", "---", ""):
            abbrev = parts[0]
            full_term = parts[1]
            if abbrev and full_term and not abbrev.startswith("-"):
                mapping[abbrev] = full_term
    return mapping


def _parse_allowed_phrases(path: Path) -> list[str]:
    """Extract non-empty approved phrases from allowed_phrases.md code blocks."""
    if not path.exists():
        return []
    phrases: list[str] = []
    in_block = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_block = not in_block
            continue
        if in_block and stripped and not stripped.startswith("Add "):
            phrases.append(stripped)
    return phrases


@dataclass(frozen=True)
class ReportingGuidelines:
    root: Path
    report_template: str
    allowed_phrases: list[str]
    prohibited_claims: list[str]
    review_triggers: list[str]          # raw lines (legacy)
    critical_flags: list[str]           # raw lines (legacy)
    abbreviation_map: str               # raw markdown text
    qc_review_template: str
    source_notes: str
    # Structured data (used by _fill_report_template)
    review_trigger_items: list[dict[str, Any]] = field(default_factory=list)
    critical_flag_items: list[dict[str, Any]] = field(default_factory=list)
    abbreviation_lookup: dict[str, str] = field(default_factory=dict)
    approved_phrases: list[str] = field(default_factory=list)


def load_reporting_guidelines(root: str | Path = "reporting_guidelines") -> ReportingGuidelines:
    base = Path(root)
    return ReportingGuidelines(
        root=base,
        report_template=_read_text(base / "report_template.md"),
        allowed_phrases=_read_lines(base / "allowed_phrases.md"),
        prohibited_claims=_read_lines(base / "prohibited_claims.md"),
        review_triggers=_read_lines(base / "review_triggers.yaml"),
        critical_flags=_read_lines(base / "critical_flags.yaml"),
        abbreviation_map=_read_text(base / "cell_abbreviation_canonical_map.md"),
        qc_review_template=_read_text(base / "qc_review_template.md"),
        source_notes=_read_text(base / "source_notes.md"),
        # Structured
        review_trigger_items=_parse_yaml_triggers(base / "review_triggers.yaml"),
        critical_flag_items=_parse_yaml_triggers(base / "critical_flags.yaml"),
        abbreviation_lookup=_parse_abbreviation_map(base / "cell_abbreviation_canonical_map.md"),
        approved_phrases=_parse_allowed_phrases(base / "allowed_phrases.md"),
    )


def _is_negated(content_lower: str, start: int) -> bool:
    prefix = content_lower[max(0, start - 120):start]
    negation_markers = [
        "not ", "no ", "cannot ", "does not ", "do not ",
        "without ", "never ", "non-",
    ]
    return any(
        re.search(r'\b' + re.escape(marker.rstrip()) + r'\b', prefix, re.IGNORECASE)
        for marker in negation_markers
    )


def find_prohibited_claims(content: str, guidelines: ReportingGuidelines) -> list[str]:
    content_lower = content.lower()
    matches: list[str] = []
    for phrase in guidelines.prohibited_claims:
        phrase_lower = phrase.lower()
        start = content_lower.find(phrase_lower)
        while start != -1:
            if not _is_negated(content_lower, start):
                matches.append(phrase)
                break
            start = content_lower.find(phrase_lower, start + len(phrase_lower))
    return matches


def validate_report_safety(content: str, guidelines: ReportingGuidelines) -> dict[str, object]:
    violations = find_prohibited_claims(content, guidelines)
    return {
        "safe": not violations,
        "violations": violations,
    }
