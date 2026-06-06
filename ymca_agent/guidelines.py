"""Guideline loading and report-safety helpers for the local agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class ReportingGuidelines:
    root: Path
    report_template: str
    allowed_phrases: list[str]
    prohibited_claims: list[str]
    review_triggers: list[str]
    critical_flags: list[str]
    abbreviation_map: str
    qc_review_template: str
    source_notes: str


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
